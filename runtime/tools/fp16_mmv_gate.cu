#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kThreads = 256U;
constexpr std::uint32_t kWarp = 32U;
constexpr std::uint32_t kWarpsPerBlock = kThreads / kWarp;
constexpr std::uint32_t kHidden = 5120U;
constexpr std::uint32_t kIntermediate = 17408U;
constexpr std::uint32_t kLayers = 64U;
constexpr std::array<std::uint32_t, 3> kIntermediateShards = {
    5824U, 5792U, 5792U};
static_assert(kIntermediateShards[0] + kIntermediateShards[1] +
                  kIntermediateShards[2] ==
              kIntermediate);

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
    check(cudaMalloc(&pointer_, bytes), "allocate device buffer");
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

__device__ float warp_sum(float value) {
  for (int offset = 16; offset != 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

__global__ void fill_half(__half* values, std::size_t count, float value) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    values[index] = __float2half_rn(value);
  }
}

// Batch-one resident matrix-vector kernel. The layout and reduction follow the
// mature CUDA MMV strategy used by llama.cpp, specialized here for SM60,
// row-major FP16 weights/activations and FP32 accumulation.
__global__ void fp16_mmv_fp32(const __half* __restrict__ weights,
                              const __half* __restrict__ input,
                              float* __restrict__ output,
                              std::uint32_t rows,
                              std::uint32_t columns) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  if (row >= rows) return;
  const auto* row_weights = reinterpret_cast<const __half2*>(
      weights + static_cast<std::size_t>(row) * columns);
  const auto* input_pairs = reinterpret_cast<const __half2*>(input);
  float total = 0.0F;
  for (std::uint32_t pair = threadIdx.x; pair < columns / 2U;
       pair += blockDim.x) {
    const float2 weight = __half22float2(row_weights[pair]);
    const float2 activation = __half22float2(input_pairs[pair]);
    total = fmaf(weight.x, activation.x, total);
    total = fmaf(weight.y, activation.y, total);
  }
  total = warp_sum(total);

  __shared__ float partial[kWarpsPerBlock];
  const auto lane = static_cast<std::uint32_t>(threadIdx.x) & (kWarp - 1U);
  const auto warp = static_cast<std::uint32_t>(threadIdx.x) / kWarp;
  if (lane == 0U) partial[warp] = total;
  __syncthreads();
  if (warp == 0U) {
    total = lane < kWarpsPerBlock ? partial[lane] : 0.0F;
    total = warp_sum(total);
    if (lane == 0U) output[row] = total;
  }
}

__global__ void swiglu_to_half(const float* gate_up, __half* output,
                               std::uint32_t width) {
  for (std::uint32_t index =
           static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < width; index += blockDim.x * gridDim.x) {
    const float gate = gate_up[index];
    const float value = gate / (1.0F + expf(-gate)) * gate_up[width + index];
    output[index] = __float2half_rn(value);
  }
}

__global__ void sum_three(const float* first, const float* second,
                          const float* third, float* output,
                          std::uint32_t count) {
  for (std::uint32_t index =
           static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    output[index] = first[index] + second[index] + third[index];
  }
}

std::uint32_t blocks_for(std::size_t count) {
  constexpr std::uint32_t maximum_blocks = 4096U;
  const auto required = static_cast<std::uint32_t>(
      (count + kThreads - 1U) / kThreads);
  return std::min(required, maximum_blocks);
}

void enable_peers() {
  for (int device = 0; device < 3; ++device) {
    check(cudaSetDevice(device), "select peer device");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "query P100");
    if (properties.major != 6 || properties.minor != 0) {
      throw std::runtime_error("devices 0-2 must be SM60 P100 GPUs");
    }
    for (int peer = 0; peer < 3; ++peer) {
      if (device == peer) continue;
      int available = 0;
      check(cudaDeviceCanAccessPeer(&available, device, peer),
            "query P100 peer access");
      if (available == 0) {
        throw std::runtime_error("three-P100 peer matrix is incomplete");
      }
      const auto status = cudaDeviceEnablePeerAccess(peer, 0);
      if (status == cudaErrorPeerAccessAlreadyEnabled) {
        static_cast<void>(cudaGetLastError());
      } else {
        check(status, "enable P100 peer access");
      }
    }
  }
}

void validate_oracle() {
  constexpr std::uint32_t rows = 17U;
  constexpr std::uint32_t columns = 64U;
  std::vector<__half> weights(static_cast<std::size_t>(rows) * columns);
  std::vector<__half> input(columns);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t column = 0; column < columns; ++column) {
      weights[static_cast<std::size_t>(row) * columns + column] =
          __float2half_rn(static_cast<float>((row * 7U + column * 3U) % 19U) /
                              64.0F -
                          0.140625F);
    }
  }
  for (std::uint32_t column = 0; column < columns; ++column) {
    input[column] = __float2half_rn(
        std::sin(static_cast<float>(column + 1U) * 0.071F) * 0.25F);
  }

  std::vector<float> expected(rows, 0.0F);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t column = 0; column < columns; ++column) {
      expected[row] += __half2float(
          weights[static_cast<std::size_t>(row) * columns + column]) *
          __half2float(input[column]);
    }
  }

  check(cudaSetDevice(0), "select MMV oracle device");
  DeviceAllocation device_weights(0, weights.size() * sizeof(__half));
  DeviceAllocation device_input(0, input.size() * sizeof(__half));
  DeviceAllocation device_output(0, expected.size() * sizeof(float));
  check(cudaMemcpy(device_weights.as<void>(), weights.data(),
                   weights.size() * sizeof(__half), cudaMemcpyHostToDevice),
        "upload MMV oracle weights");
  check(cudaMemcpy(device_input.as<void>(), input.data(),
                   input.size() * sizeof(__half), cudaMemcpyHostToDevice),
        "upload MMV oracle input");
  fp16_mmv_fp32<<<rows, kThreads>>>(
      device_weights.as<__half>(), device_input.as<__half>(),
      device_output.as<float>(), rows, columns);
  check(cudaPeekAtLastError(), "launch MMV oracle");
  std::vector<float> actual(rows);
  check(cudaMemcpy(actual.data(), device_output.as<void>(),
                   actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download MMV oracle output");

  double maximum_error = 0.0;
  double squared_error = 0.0;
  double squared_reference = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double error = static_cast<double>(actual[index]) - expected[index];
    maximum_error = std::max(maximum_error, std::abs(error));
    squared_error += error * error;
    squared_reference +=
        static_cast<double>(expected[index]) * expected[index];
  }
  const double relative_l2 =
      std::sqrt(squared_error / std::max(squared_reference, 1.0e-30));
  std::cout << std::scientific << "oracle_max_abs=" << maximum_error
            << " oracle_relative_l2=" << relative_l2 << '\n';
  if (maximum_error > 2.0e-4 || relative_l2 > 2.0e-5) {
    throw std::runtime_error("independent FP16 MMV oracle failed");
  }
}

class Shard final {
 public:
  Shard(int device, std::uint32_t intermediate)
      : device_(device), intermediate_(intermediate),
        gate_up_values_(static_cast<std::size_t>(2U * intermediate) * kHidden),
        down_values_(static_cast<std::size_t>(kHidden) * intermediate),
        gate_up_weights_(device, gate_up_values_ * sizeof(__half)),
        down_weights_(device, down_values_ * sizeof(__half)),
        input_(device, kHidden * sizeof(__half)),
        gate_up_(device, 2ULL * intermediate * sizeof(float)),
        intermediate_buffer_(device, intermediate * sizeof(__half)),
        output_(device, kHidden * sizeof(float)) {
    check(cudaSetDevice(device_), "select MMV shard");
    check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
          "create MMV shard stream");
    fill_half<<<blocks_for(gate_up_values_), kThreads, 0, stream_>>>(
        gate_up_weights_.as<__half>(), gate_up_values_, 0.0009765625F);
    fill_half<<<blocks_for(down_values_), kThreads, 0, stream_>>>(
        down_weights_.as<__half>(), down_values_, 0.000244140625F);
    check(cudaPeekAtLastError(), "initialize MMV weights");
    check(cudaStreamSynchronize(stream_), "finish MMV weight initialization");
  }

  ~Shard() {
    cudaSetDevice(device_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  Shard(const Shard&) = delete;
  Shard& operator=(const Shard&) = delete;

  void launch() {
    check(cudaSetDevice(device_), "select launching MMV shard");
    fp16_mmv_fp32<<<2U * intermediate_, kThreads, 0, stream_>>>(
        gate_up_weights_.as<__half>(), input_.as<__half>(),
        gate_up_.as<float>(), 2U * intermediate_, kHidden);
    swiglu_to_half<<<(intermediate_ + kThreads - 1U) / kThreads, kThreads, 0,
                     stream_>>>(gate_up_.as<float>(),
                                intermediate_buffer_.as<__half>(),
                                intermediate_);
    fp16_mmv_fp32<<<kHidden, kThreads, 0, stream_>>>(
        down_weights_.as<__half>(), intermediate_buffer_.as<__half>(),
        output_.as<float>(), kHidden, intermediate_);
    check(cudaPeekAtLastError(), "launch resident FP16 MMV MLP");
  }

  void synchronize() {
    check(cudaSetDevice(device_), "select synchronizing MMV shard");
    check(cudaStreamSynchronize(stream_), "synchronize MMV shard");
  }

  cudaStream_t stream() const { return stream_; }
  __half* input() const { return input_.as<__half>(); }
  float* output() const { return output_.as<float>(); }
  std::size_t weight_bytes() const {
    return (gate_up_values_ + down_values_) * sizeof(__half);
  }

 private:
  int device_{};
  std::uint32_t intermediate_{};
  std::size_t gate_up_values_{};
  std::size_t down_values_{};
  DeviceAllocation gate_up_weights_;
  DeviceAllocation down_weights_;
  DeviceAllocation input_;
  DeviceAllocation gate_up_;
  DeviceAllocation intermediate_buffer_;
  DeviceAllocation output_;
  cudaStream_t stream_{};
};

class Program final {
 public:
  Program()
      : peer_one_(0, kHidden * sizeof(float)),
        peer_two_(0, kHidden * sizeof(float)),
        reduced_(0, kHidden * sizeof(float)) {
    enable_peers();
    for (int device = 0; device < 3; ++device) {
      shards_.push_back(std::make_unique<Shard>(
          device, kIntermediateShards[static_cast<std::size_t>(device)]));
    }
    std::vector<__half> input(kHidden);
    for (std::uint32_t index = 0; index < kHidden; ++index) {
      input[index] = __float2half_rn(
          std::sin(static_cast<float>(index + 1U) * 0.013F) * 0.1F);
    }
    check(cudaSetDevice(0), "select MMV input device");
    check(cudaMemcpy(shards_[0]->input(), input.data(),
                     input.size() * sizeof(__half), cudaMemcpyHostToDevice),
          "upload MMV input");
  }

  double run(std::uint32_t layers) {
    const auto begin = std::chrono::steady_clock::now();
    for (std::uint32_t layer = 0; layer < layers; ++layer) run_layer();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
  }

  std::size_t layer_weight_bytes() const {
    std::size_t total = 0;
    for (const auto& shard : shards_) total += shard->weight_bytes();
    return total;
  }

 private:
  void run_layer() {
    constexpr auto input_bytes = kHidden * sizeof(__half);
    check(cudaSetDevice(1), "select P100 1 MMV broadcast");
    check(cudaMemcpyPeerAsync(shards_[1]->input(), 1, shards_[0]->input(), 0,
                              input_bytes, shards_[1]->stream()),
          "broadcast MMV input to P100 1");
    check(cudaSetDevice(2), "select P100 2 MMV broadcast");
    check(cudaMemcpyPeerAsync(shards_[2]->input(), 2, shards_[0]->input(), 0,
                              input_bytes, shards_[2]->stream()),
          "broadcast MMV input to P100 2");
    for (auto& shard : shards_) shard->launch();
    for (auto& shard : shards_) shard->synchronize();

    constexpr auto output_bytes = kHidden * sizeof(float);
    check(cudaSetDevice(0), "select MMV reduction device");
    check(cudaMemcpyPeerAsync(peer_one_.as<void>(), 0, shards_[1]->output(), 1,
                              output_bytes, shards_[0]->stream()),
          "collect P100 1 MMV result");
    check(cudaMemcpyPeerAsync(peer_two_.as<void>(), 0, shards_[2]->output(), 2,
                              output_bytes, shards_[0]->stream()),
          "collect P100 2 MMV result");
    sum_three<<<(kHidden + kThreads - 1U) / kThreads, kThreads, 0,
                shards_[0]->stream()>>>(
        shards_[0]->output(), peer_one_.as<float>(), peer_two_.as<float>(),
        reduced_.as<float>(), kHidden);
    check(cudaPeekAtLastError(), "launch MMV reduction");
    check(cudaStreamSynchronize(shards_[0]->stream()),
          "complete MMV reduction");
  }

  DeviceAllocation peer_one_;
  DeviceAllocation peer_two_;
  DeviceAllocation reduced_;
  std::vector<std::unique_ptr<Shard>> shards_;
};

}  // namespace

int main() try {
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "enumerate GPUs");
  if (device_count < 3) throw std::runtime_error("three GPUs are required");
  validate_oracle();
  Program program;
  static_cast<void>(program.run(1U));
  const double milliseconds = program.run(kLayers);
  const double gib =
      static_cast<double>(program.layer_weight_bytes()) * kLayers /
      static_cast<double>(1ULL << 30U);
  std::cout << std::fixed << std::setprecision(3)
            << "layers=" << kLayers
            << " integrated_fp16_mmv_ms=" << milliseconds
            << " per_layer_ms=" << milliseconds / kLayers
            << " effective_weight_gib_s=" << gib / (milliseconds / 1000.0)
            << " mlp_only_tok_s=" << 1000.0 / milliseconds << '\n';
  return EXIT_SUCCESS;
} catch (const std::exception& error) {
  std::cerr << "fp16_mmv_gate: " << error.what() << '\n';
  return EXIT_FAILURE;
}
