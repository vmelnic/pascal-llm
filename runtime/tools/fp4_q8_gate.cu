#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kThreads = 256U;
constexpr std::uint32_t kWarp = 32U;
constexpr std::uint32_t kWarpsPerBlock = kThreads / kWarp;
constexpr std::uint32_t kQwenHidden = 5120U;
constexpr std::uint32_t kQwenIntermediate = 17408U;
constexpr std::uint32_t kQwenLayers = 64U;
constexpr int kRepetitions = 20;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class DeviceAllocation final {
 public:
  DeviceAllocation(int device, std::size_t bytes) : device_(device) {
    check(cudaSetDevice(device_), "select allocation device");
    check(cudaMalloc(&pointer_, bytes), "allocate device memory");
  }

  ~DeviceAllocation() {
    if (pointer_ != nullptr) {
      cudaSetDevice(device_);
      cudaFree(pointer_);
    }
  }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  template <typename T>
  T* as() const {
    return static_cast<T*>(pointer_);
  }

 private:
  int device_{};
  void* pointer_{};
};

__device__ std::int8_t device_fp4_twice(std::uint8_t code) {
  const auto magnitude_index = code & 7U;
  const auto magnitude = magnitude_index <= 4U ? static_cast<int>(magnitude_index)
      : magnitude_index == 5U ? 6
      : magnitude_index == 6U ? 8
                              : 12;
  return static_cast<std::int8_t>((code & 8U) != 0U ? -magnitude : magnitude);
}

__device__ float device_ue8m0(std::uint8_t code) {
  return code == 0U ? __uint_as_float(0x00400000U)
                    : __uint_as_float(static_cast<unsigned>(code) << 23U);
}

__device__ float warp_sum(float value) {
  for (int offset = 16; offset != 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

__global__ void quantize_q8(const float* input, std::int8_t* output,
                            float* output_scale, std::uint32_t columns) {
  __shared__ float maxima[kThreads];
  float maximum = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    maximum = fmaxf(maximum, fabsf(input[column]));
  }
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x],
                                  maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float scale = maxima[0] > 0.0F ? maxima[0] / 127.0F : 1.0F;
  if (threadIdx.x == 0U) output_scale[0] = scale;
  for (std::uint32_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    int value = __float2int_rn(input[column] / scale);
    value = max(-127, min(127, value));
    output[column] = static_cast<std::int8_t>(value);
  }
}

__device__ float fp4_q8_dot(const std::uint8_t* weights,
                            const std::uint8_t* scales,
                            const std::int8_t* input,
                            const float* input_scale, std::uint32_t row,
                            std::uint32_t columns) {
  const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U;
  const auto blocks = columns / 32U;
  const auto* row_weights = weights +
      static_cast<std::size_t>(row) * (columns / 2U);
  const auto* row_scales = scales +
      static_cast<std::size_t>(row) * blocks;
  float total = 0.0F;
  for (std::uint32_t block = lane; block < blocks; block += kWarp) {
    const auto* packed = row_weights + static_cast<std::size_t>(block) * 16U;
    const auto* activation = input + static_cast<std::size_t>(block) * 32U;
    int block_total = 0;
#pragma unroll
    for (std::uint32_t byte = 0; byte < 16U; ++byte) {
      const auto codes = packed[byte];
      block_total +=
          static_cast<int>(device_fp4_twice(codes & 0x0fU)) *
              static_cast<int>(activation[2U * byte]) +
          static_cast<int>(device_fp4_twice(codes >> 4U)) *
              static_cast<int>(activation[2U * byte + 1U]);
    }
    total += static_cast<float>(block_total) * device_ue8m0(row_scales[block]);
  }
  return warp_sum(total) * input_scale[0] * 0.5F;
}

__global__ void fp4_q8_gemv(const std::uint8_t* weights,
                            const std::uint8_t* scales,
                            const std::int8_t* input,
                            const float* input_scale, float* output,
                            std::uint32_t rows, std::uint32_t columns) {
  const auto warp = static_cast<std::uint32_t>(threadIdx.x) / kWarp;
  const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U;
  const auto row = static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;
  const auto value = fp4_q8_dot(weights, scales, input, input_scale, row, columns);
  if (lane == 0U) output[row] = value;
}

std::size_t packed_bytes(std::uint32_t rows, std::uint32_t columns) {
  return static_cast<std::size_t>(rows) * columns / 2U;
}

std::size_t scale_bytes(std::uint32_t rows, std::uint32_t columns) {
  return static_cast<std::size_t>(rows) * columns / 32U;
}

struct Result final {
  int device{};
  std::string name;
  std::uint32_t local_intermediate{};
  double gate_up_ms{};
  double gate_up_gib_s{};
  double down_ms{};
  double down_gib_s{};
};

class Operation final {
 public:
  Operation(int device, std::uint32_t rows, std::uint32_t columns,
            std::uint8_t pattern)
      : device_(device), rows_(rows), columns_(columns),
        weight_bytes_(packed_bytes(rows, columns)),
        scale_bytes_(::scale_bytes(rows, columns)),
        weights_(device, weight_bytes_), scales_(device, scale_bytes_),
        input_(device, static_cast<std::size_t>(columns) * sizeof(float)),
        quantized_(device, columns), input_scale_(device, sizeof(float)),
        output_(device, static_cast<std::size_t>(rows) * sizeof(float)) {
    check(cudaSetDevice(device_), "select operation device");
    check(cudaMemset(weights_.as<void>(), pattern, weight_bytes_),
          "initialize packed weights");
    check(cudaMemset(scales_.as<void>(), 127, scale_bytes_),
          "initialize UE8M0 scales");
    std::vector<float> input(columns);
    for (std::uint32_t index = 0; index < columns; ++index) {
      input[index] = std::sin(static_cast<float>(index + 1U) * 0.013F) * 0.5F;
    }
    check(cudaMemcpy(input_.as<void>(), input.data(),
                     input.size() * sizeof(float), cudaMemcpyHostToDevice),
          "upload operation input");
    check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
          "create operation stream");
    check(cudaEventCreate(&begin_), "create operation begin event");
    check(cudaEventCreate(&end_), "create operation end event");
    launch();
    check(cudaStreamSynchronize(stream_), "warm operation");
  }

  ~Operation() {
    cudaSetDevice(device_);
    if (end_ != nullptr) cudaEventDestroy(end_);
    if (begin_ != nullptr) cudaEventDestroy(begin_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  Operation(const Operation&) = delete;
  Operation& operator=(const Operation&) = delete;

  void begin() {
    check(cudaSetDevice(device_), "select timed operation device");
    check(cudaEventRecord(begin_, stream_), "record operation begin");
    for (int repetition = 0; repetition < kRepetitions; ++repetition) launch();
    check(cudaEventRecord(end_, stream_), "record operation end");
  }

  std::pair<double, double> finish() {
    check(cudaSetDevice(device_), "select finishing operation device");
    check(cudaEventSynchronize(end_), "wait for operation");
    float total_ms = 0.0F;
    check(cudaEventElapsedTime(&total_ms, begin_, end_), "measure operation");
    const double milliseconds = total_ms / kRepetitions;
    const double bytes = static_cast<double>(weight_bytes_ + scale_bytes_);
    const double gib_per_second =
        bytes / static_cast<double>(1ULL << 30U) / (milliseconds / 1000.0);
    return {milliseconds, gib_per_second};
  }

 private:
  void launch() {
    quantize_q8<<<1U, kThreads, 0, stream_>>>(
        input_.as<float>(), quantized_.as<std::int8_t>(),
        input_scale_.as<float>(), columns_);
    check(cudaPeekAtLastError(), "launch Q8 quantization");
    const auto blocks = (rows_ + kWarpsPerBlock - 1U) / kWarpsPerBlock;
    fp4_q8_gemv<<<blocks, kThreads, 0, stream_>>>(
        weights_.as<std::uint8_t>(), scales_.as<std::uint8_t>(),
        quantized_.as<std::int8_t>(), input_scale_.as<float>(),
        output_.as<float>(), rows_, columns_);
    check(cudaPeekAtLastError(), "launch FP4 Q8 GEMV");
  }

  int device_{};
  std::uint32_t rows_{};
  std::uint32_t columns_{};
  std::size_t weight_bytes_{};
  std::size_t scale_bytes_{};
  DeviceAllocation weights_;
  DeviceAllocation scales_;
  DeviceAllocation input_;
  DeviceAllocation quantized_;
  DeviceAllocation input_scale_;
  DeviceAllocation output_;
  cudaStream_t stream_{};
  cudaEvent_t begin_{};
  cudaEvent_t end_{};
};

int host_fp4_twice(std::uint8_t code) {
  constexpr int values[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                              0, -1, -2, -3, -4, -6, -8, -12};
  return values[code & 15U];
}

float host_ue8m0(std::uint8_t code) {
  return std::ldexp(1.0F, code == 0U ? -127 : static_cast<int>(code) - 127);
}

void validate_oracle() {
  constexpr std::uint32_t rows = 17U;
  constexpr std::uint32_t columns = 64U;
  std::vector<std::uint8_t> weights(packed_bytes(rows, columns));
  std::vector<std::uint8_t> scales(scale_bytes(rows, columns));
  std::vector<float> input(columns);
  for (std::size_t index = 0; index < weights.size(); ++index) {
    weights[index] = static_cast<std::uint8_t>((index * 37U + 19U) & 0xffU);
  }
  for (std::size_t index = 0; index < scales.size(); ++index) {
    scales[index] = static_cast<std::uint8_t>(122U + index % 9U);
  }
  for (std::uint32_t index = 0; index < columns; ++index) {
    input[index] = std::sin(static_cast<float>(index + 3U) * 0.071F) * 0.7F;
  }

  float maximum = 0.0F;
  for (const auto value : input) maximum = std::max(maximum, std::abs(value));
  const float input_scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
  std::vector<std::int8_t> quantized(columns);
  for (std::uint32_t column = 0; column < columns; ++column) {
    auto value = static_cast<int>(std::nearbyint(input[column] / input_scale));
    quantized[column] = static_cast<std::int8_t>(std::clamp(value, -127, 127));
  }
  std::vector<float> expected(rows, 0.0F);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t block = 0; block < columns / 32U; ++block) {
      int block_total = 0;
      for (std::uint32_t column = 0; column < 32U; ++column) {
        const auto offset = static_cast<std::size_t>(row) * (columns / 2U) +
                            block * 16U + column / 2U;
        const auto packed = weights[offset];
        const auto code = static_cast<std::uint8_t>(
            (column & 1U) == 0U ? packed & 15U : packed >> 4U);
        block_total += host_fp4_twice(code) *
            static_cast<int>(quantized[block * 32U + column]);
      }
      expected[row] += static_cast<float>(block_total) *
          host_ue8m0(scales[static_cast<std::size_t>(row) *
                            (columns / 32U) + block]);
    }
    expected[row] *= input_scale * 0.5F;
  }

  check(cudaSetDevice(0), "select oracle GPU");
  DeviceAllocation d_weights(0, weights.size());
  DeviceAllocation d_scales(0, scales.size());
  DeviceAllocation d_input(0, input.size() * sizeof(float));
  DeviceAllocation d_quantized(0, quantized.size());
  DeviceAllocation d_input_scale(0, sizeof(float));
  DeviceAllocation d_output(0, expected.size() * sizeof(float));
  check(cudaMemcpy(d_weights.as<void>(), weights.data(), weights.size(),
                   cudaMemcpyHostToDevice), "upload oracle weights");
  check(cudaMemcpy(d_scales.as<void>(), scales.data(), scales.size(),
                   cudaMemcpyHostToDevice), "upload oracle scales");
  check(cudaMemcpy(d_input.as<void>(), input.data(), input.size() * sizeof(float),
                   cudaMemcpyHostToDevice), "upload oracle input");
  quantize_q8<<<1U, kThreads>>>(d_input.as<float>(),
                               d_quantized.as<std::int8_t>(),
                               d_input_scale.as<float>(), columns);
  fp4_q8_gemv<<<(rows + kWarpsPerBlock - 1U) / kWarpsPerBlock, kThreads>>>(
      d_weights.as<std::uint8_t>(), d_scales.as<std::uint8_t>(),
      d_quantized.as<std::int8_t>(), d_input_scale.as<float>(),
      d_output.as<float>(), rows, columns);
  check(cudaPeekAtLastError(), "launch numerical oracle");
  std::vector<float> actual(rows);
  check(cudaMemcpy(actual.data(), d_output.as<void>(),
                   actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download oracle output");

  double maximum_error = 0.0;
  double squared_error = 0.0;
  double squared_reference = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double error = static_cast<double>(actual[index]) - expected[index];
    maximum_error = std::max(maximum_error, std::abs(error));
    squared_error += error * error;
    squared_reference += static_cast<double>(expected[index]) * expected[index];
  }
  const double relative_l2 =
      std::sqrt(squared_error / std::max(squared_reference, 1.0e-30));
  std::cout << std::scientific << "oracle_max_abs=" << maximum_error
            << " oracle_relative_l2=" << relative_l2 << '\n';
  if (maximum_error > 1.0e-3 || relative_l2 > 1.0e-5) {
    throw std::runtime_error("independent FP4 Q8 oracle failed");
  }
}

}  // namespace

int main() try {
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "enumerate devices");
  if (device_count < 3) {
    throw std::runtime_error("three P100 devices are required");
  }
  for (int device = 0; device < 3; ++device) {
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "read device properties");
    if (properties.major != 6 || properties.minor != 0) {
      throw std::runtime_error("CUDA devices 0-2 must be SM60");
    }
  }

  validate_oracle();
  constexpr std::uint32_t shards[3] = {5824U, 5792U, 5792U};
  static_assert(shards[0] + shards[1] + shards[2] == kQwenIntermediate);
  std::vector<std::unique_ptr<Operation>> gate_up;
  std::vector<std::unique_ptr<Operation>> down;
  for (int device = 0; device < 3; ++device) {
    gate_up.push_back(std::make_unique<Operation>(
        device, 2U * shards[device], kQwenHidden, 0x21U));
    down.push_back(std::make_unique<Operation>(
        device, kQwenHidden, shards[device], 0x43U));
  }

  for (auto& operation : gate_up) operation->begin();
  std::vector<std::pair<double, double>> gate_results;
  for (auto& operation : gate_up) gate_results.push_back(operation->finish());
  for (auto& operation : down) operation->begin();
  std::vector<std::pair<double, double>> down_results;
  for (auto& operation : down) down_results.push_back(operation->finish());

  double maximum_layer_ms = 0.0;
  std::cout << std::fixed << std::setprecision(3);
  for (int device = 0; device < 3; ++device) {
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "read result device");
    const double layer_ms = gate_results[device].first + down_results[device].first;
    maximum_layer_ms = std::max(maximum_layer_ms, layer_ms);
    std::cout << "device=" << device << " name=\"" << properties.name
              << "\" local_intermediate=" << shards[device]
              << " gate_up_ms=" << gate_results[device].first
              << " gate_up_gib_s=" << gate_results[device].second
              << " down_ms=" << down_results[device].first
              << " down_gib_s=" << down_results[device].second
              << " layer_mlp_ms=" << layer_ms << '\n';
  }
  std::cout << "projected_64_layer_mlp_ms="
            << maximum_layer_ms * kQwenLayers << '\n';
  return EXIT_SUCCESS;
} catch (const std::exception& error) {
  std::cerr << "fp4_q8_gate: " << error.what() << '\n';
  return EXIT_FAILURE;
}
