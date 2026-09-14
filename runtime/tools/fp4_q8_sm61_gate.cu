#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kThreads = 256U;
constexpr std::uint32_t kWarp = 32U;
constexpr std::uint32_t kWarpsPerBlock = kThreads / kWarp;
constexpr std::uint32_t kQwenHidden = 5120U;
constexpr std::uint32_t kRepresentativeIntermediate = 5824U;
constexpr int kRepetitions = 20;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class DeviceAllocation final {
 public:
  explicit DeviceAllocation(std::size_t bytes) {
    check(cudaMalloc(&pointer_, bytes), "allocate device memory");
  }
  ~DeviceAllocation() {
    if (pointer_ != nullptr) cudaFree(pointer_);
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  template <typename T>
  T* as() const { return static_cast<T*>(pointer_); }

 private:
  void* pointer_{};
};

__device__ float ue8m0(std::uint8_t code) {
  return code == 0U ? __uint_as_float(0x00400000U)
                    : __uint_as_float(static_cast<unsigned>(code) << 23U);
}

__device__ float warp_sum(float value) {
  for (int offset = 16; offset != 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

__device__ int packed_fp4x4(std::uint8_t first, std::uint8_t second) {
  const auto selectors = static_cast<std::uint32_t>(first) |
                         (static_cast<std::uint32_t>(second) << 8U);
  const auto magnitudes =
      __byte_perm(0x03020100U, 0x0c080604U, selectors & 0x7777U);
  int a = static_cast<int>(magnitudes & 0xffU);
  int b = static_cast<int>((magnitudes >> 8U) & 0xffU);
  int c = static_cast<int>((magnitudes >> 16U) & 0xffU);
  int d = static_cast<int>((magnitudes >> 24U) & 0xffU);
  if ((first & 0x08U) != 0U) a = -a;
  if ((first & 0x80U) != 0U) b = -b;
  if ((second & 0x08U) != 0U) c = -c;
  if ((second & 0x80U) != 0U) d = -d;
  return static_cast<int>(static_cast<std::uint8_t>(a)) |
      (static_cast<int>(static_cast<std::uint8_t>(b)) << 8U) |
      (static_cast<int>(static_cast<std::uint8_t>(c)) << 16U) |
      (static_cast<int>(static_cast<std::uint8_t>(d)) << 24U);
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
                            const std::int8_t* activation,
                            const float* activation_scale,
                            std::uint32_t row, std::uint32_t columns) {
  const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U;
  const auto blocks = columns / 32U;
  const auto* row_weights = reinterpret_cast<const int4*>(
      weights + static_cast<std::size_t>(row) * (columns / 2U));
  const auto* row_scales = scales + static_cast<std::size_t>(row) * blocks;
  const auto* activation_blocks = reinterpret_cast<const int4*>(activation);
  float total = 0.0F;
  for (std::uint32_t block = lane; block < blocks; block += kWarp) {
    const auto packed_weights = row_weights[block];
    const auto activation_low = activation_blocks[2U * block];
    const auto activation_high = activation_blocks[2U * block + 1U];
    const std::uint32_t weight_words[] = {
        static_cast<std::uint32_t>(packed_weights.x),
        static_cast<std::uint32_t>(packed_weights.y),
        static_cast<std::uint32_t>(packed_weights.z),
        static_cast<std::uint32_t>(packed_weights.w),
    };
    const int activation_words[] = {
        activation_low.x, activation_low.y, activation_low.z, activation_low.w,
        activation_high.x, activation_high.y, activation_high.z,
        activation_high.w,
    };
    int block_total = 0;
#pragma unroll
    for (std::uint32_t group = 0; group < 8U; ++group) {
      const auto pair = static_cast<std::uint16_t>(
          weight_words[group / 2U] >> (16U * (group & 1U)));
      block_total = __dp4a(
          packed_fp4x4(static_cast<std::uint8_t>(pair),
                       static_cast<std::uint8_t>(pair >> 8U)),
          activation_words[group], block_total);
    }
    total += static_cast<float>(block_total) * ue8m0(row_scales[block]);
  }
  return warp_sum(total) * activation_scale[0] * 0.5F;
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

std::size_t weight_bytes(std::uint32_t rows, std::uint32_t columns) {
  return static_cast<std::size_t>(rows) * columns / 2U;
}

std::size_t scale_bytes(std::uint32_t rows, std::uint32_t columns) {
  return static_cast<std::size_t>(rows) * columns / 32U;
}

struct Measurement final {
  double milliseconds{};
  double gib_per_second{};
};

Measurement measure(std::uint32_t rows, std::uint32_t columns,
                    std::uint8_t pattern) {
  const auto weights_size = weight_bytes(rows, columns);
  const auto scales_size = scale_bytes(rows, columns);
  DeviceAllocation weights(weights_size);
  DeviceAllocation scales(scales_size);
  DeviceAllocation input(static_cast<std::size_t>(columns) * sizeof(float));
  DeviceAllocation quantized(columns);
  DeviceAllocation input_scale(sizeof(float));
  DeviceAllocation output(static_cast<std::size_t>(rows) * sizeof(float));
  check(cudaMemset(weights.as<void>(), pattern, weights_size),
        "initialize packed weights");
  check(cudaMemset(scales.as<void>(), 127, scales_size),
        "initialize scales");
  std::vector<float> host_input(columns);
  for (std::uint32_t index = 0; index < columns; ++index) {
    host_input[index] = std::sin(static_cast<float>(index + 1U) * 0.013F) * 0.5F;
  }
  check(cudaMemcpy(input.as<void>(), host_input.data(),
                   host_input.size() * sizeof(float), cudaMemcpyHostToDevice),
        "upload input");
  cudaStream_t stream{};
  cudaEvent_t begin{};
  cudaEvent_t end{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create stream");
  check(cudaEventCreate(&begin), "create begin event");
  check(cudaEventCreate(&end), "create end event");
  const auto launch = [&] {
    quantize_q8<<<1U, kThreads, 0, stream>>>(
        input.as<float>(), quantized.as<std::int8_t>(), input_scale.as<float>(),
        columns);
    fp4_q8_gemv<<<(rows + kWarpsPerBlock - 1U) / kWarpsPerBlock,
                    kThreads, 0, stream>>>(
        weights.as<std::uint8_t>(), scales.as<std::uint8_t>(),
        quantized.as<std::int8_t>(), input_scale.as<float>(), output.as<float>(),
        rows, columns);
    check(cudaPeekAtLastError(), "launch SM61 FP4 Q8 operation");
  };
  launch();
  check(cudaStreamSynchronize(stream), "warm SM61 operation");
  check(cudaEventRecord(begin, stream), "record begin");
  for (int repetition = 0; repetition < kRepetitions; ++repetition) launch();
  check(cudaEventRecord(end, stream), "record end");
  check(cudaEventSynchronize(end), "wait for SM61 operation");
  float total_ms = 0.0F;
  check(cudaEventElapsedTime(&total_ms, begin, end), "measure SM61 operation");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  cudaStreamDestroy(stream);
  const double milliseconds = total_ms / kRepetitions;
  return {
      milliseconds,
      static_cast<double>(weights_size + scales_size) /
          static_cast<double>(1ULL << 30U) / (milliseconds / 1000.0),
  };
}

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
  std::vector<std::uint8_t> weights(weight_bytes(rows, columns));
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
  const float input_scale = maximum / 127.0F;
  std::vector<std::int8_t> quantized(columns);
  for (std::uint32_t column = 0; column < columns; ++column) {
    quantized[column] = static_cast<std::int8_t>(std::clamp(
        static_cast<int>(std::nearbyint(input[column] / input_scale)),
        -127, 127));
  }
  std::vector<float> expected(rows, 0.0F);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t block = 0; block < columns / 32U; ++block) {
      int block_total = 0;
      for (std::uint32_t column = 0; column < 32U; ++column) {
        const auto packed = weights[
            static_cast<std::size_t>(row) * (columns / 2U) +
            block * 16U + column / 2U];
        const auto code = static_cast<std::uint8_t>(
            (column & 1U) == 0U ? packed & 15U : packed >> 4U);
        block_total += host_fp4_twice(code) *
            static_cast<int>(quantized[block * 32U + column]);
      }
      expected[row] += static_cast<float>(block_total) * host_ue8m0(
          scales[static_cast<std::size_t>(row) * (columns / 32U) + block]);
    }
    expected[row] *= input_scale * 0.5F;
  }

  DeviceAllocation d_weights(weights.size());
  DeviceAllocation d_scales(scales.size());
  DeviceAllocation d_input(input.size() * sizeof(float));
  DeviceAllocation d_quantized(quantized.size());
  DeviceAllocation d_input_scale(sizeof(float));
  DeviceAllocation d_output(expected.size() * sizeof(float));
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
  check(cudaPeekAtLastError(), "launch SM61 oracle");
  std::vector<float> actual(rows);
  check(cudaMemcpy(actual.data(), d_output.as<void>(),
                   actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download SM61 oracle");
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
    throw std::runtime_error("independent SM61 oracle failed");
  }
}

}  // namespace

int main() try {
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "enumerate devices");
  if (device_count < 4) throw std::runtime_error("P40 device is absent");
  check(cudaSetDevice(3), "select P40");
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, 3), "read P40 properties");
  if (properties.major != 6 || properties.minor != 1) {
    throw std::runtime_error("CUDA device 3 must be SM61");
  }
  validate_oracle();
  const auto gate_up = measure(2U * kRepresentativeIntermediate,
                               kQwenHidden, 0x21U);
  const auto down = measure(kQwenHidden, kRepresentativeIntermediate, 0x43U);
  std::cout << std::fixed << std::setprecision(3)
            << "device=3 name=\"" << properties.name << "\""
            << " representative_intermediate=" << kRepresentativeIntermediate
            << " gate_up_ms=" << gate_up.milliseconds
            << " gate_up_gib_s=" << gate_up.gib_per_second
            << " down_ms=" << down.milliseconds
            << " down_gib_s=" << down.gib_per_second
            << " representative_layer_mlp_ms="
            << gate_up.milliseconds + down.milliseconds << '\n';
  return EXIT_SUCCESS;
} catch (const std::exception& error) {
  std::cerr << "fp4_q8_sm61_gate: " << error.what() << '\n';
  return EXIT_FAILURE;
}
