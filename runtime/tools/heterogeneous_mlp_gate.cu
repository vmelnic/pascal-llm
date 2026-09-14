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
constexpr std::array<std::uint32_t, 4> kIntermediateShards = {
    4416U, 4640U, 4544U, 3808U};
static_assert(kIntermediateShards[0] + kIntermediateShards[1] +
                  kIntermediateShards[2] + kIntermediateShards[3] ==
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
  T* as() const { return static_cast<T*>(pointer_); }

 private:
  int device_{};
  void* pointer_{};
};

class PinnedAllocation final {
 public:
  explicit PinnedAllocation(std::size_t bytes) {
    check(cudaHostAlloc(&pointer_, bytes, cudaHostAllocPortable),
          "allocate pinned host buffer");
  }
  ~PinnedAllocation() {
    if (pointer_ != nullptr) cudaFreeHost(pointer_);
  }
  PinnedAllocation(const PinnedAllocation&) = delete;
  PinnedAllocation& operator=(const PinnedAllocation&) = delete;
  template <typename T>
  T* as() const { return static_cast<T*>(pointer_); }

 private:
  void* pointer_{};
};

__device__ std::int8_t fp4_twice(std::uint8_t code) {
  const auto index = code & 7U;
  const auto magnitude = index <= 4U ? static_cast<int>(index)
      : index == 5U ? 6
      : index == 6U ? 8
                    : 12;
  return static_cast<std::int8_t>((code & 8U) != 0U ? -magnitude : magnitude);
}

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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
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
#endif

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
  const auto scale = maxima[0] > 0.0F ? maxima[0] / 127.0F : 1.0F;
  if (threadIdx.x == 0U) output_scale[0] = scale;
  for (std::uint32_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    int value = __float2int_rn(input[column] / scale);
    value = max(-127, min(127, value));
    output[column] = static_cast<std::int8_t>(value);
  }
}

template <bool UseDp4a>
__device__ float fp4_q8_dot(const std::uint8_t* weights,
                            const std::uint8_t* scales,
                            const std::int8_t* activation,
                            const float* activation_scale,
                            std::uint32_t row, std::uint32_t columns) {
  const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U;
  const auto blocks = columns / 32U;
  const auto* row_weights = weights +
      static_cast<std::size_t>(row) * (columns / 2U);
  const auto* row_scales = scales +
      static_cast<std::size_t>(row) * blocks;
  float total = 0.0F;
  for (std::uint32_t block = lane; block < blocks; block += kWarp) {
    const auto* packed = row_weights + static_cast<std::size_t>(block) * 16U;
    const auto* q = activation + static_cast<std::size_t>(block) * 32U;
    int block_total = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    if constexpr (UseDp4a) {
      const auto packed_weights =
          *reinterpret_cast<const int4*>(packed);
      const auto activation_low =
          *reinterpret_cast<const int4*>(q);
      const auto activation_high =
          *reinterpret_cast<const int4*>(q + 16U);
      const std::uint32_t weight_words[] = {
          static_cast<std::uint32_t>(packed_weights.x),
          static_cast<std::uint32_t>(packed_weights.y),
          static_cast<std::uint32_t>(packed_weights.z),
          static_cast<std::uint32_t>(packed_weights.w),
      };
      const int activation_words[] = {
          activation_low.x, activation_low.y, activation_low.z,
          activation_low.w, activation_high.x, activation_high.y,
          activation_high.z, activation_high.w,
      };
#pragma unroll
      for (std::uint32_t group = 0; group < 8U; ++group) {
        const auto pair = static_cast<std::uint16_t>(
            weight_words[group / 2U] >> (16U * (group & 1U)));
        block_total = __dp4a(
            packed_fp4x4(static_cast<std::uint8_t>(pair),
                         static_cast<std::uint8_t>(pair >> 8U)),
            activation_words[group], block_total);
      }
    } else
#endif
    {
#pragma unroll
      for (std::uint32_t byte = 0; byte < 16U; ++byte) {
        const auto codes = packed[byte];
        block_total +=
            static_cast<int>(fp4_twice(codes & 15U)) *
                static_cast<int>(q[2U * byte]) +
            static_cast<int>(fp4_twice(codes >> 4U)) *
                static_cast<int>(q[2U * byte + 1U]);
      }
    }
    total += static_cast<float>(block_total) * ue8m0(row_scales[block]);
  }
  return warp_sum(total) * activation_scale[0] * 0.5F;
}

template <bool UseDp4a>
__global__ void fp4_q8_gemv(const std::uint8_t* weights,
                            const std::uint8_t* scales,
                            const std::int8_t* input,
                            const float* input_scale, float* output,
                            std::uint32_t rows, std::uint32_t columns) {
  const auto warp = static_cast<std::uint32_t>(threadIdx.x) / kWarp;
  const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U;
  const auto row = static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;
  const auto value = fp4_q8_dot<UseDp4a>(
      weights, scales, input, input_scale, row, columns);
  if (lane == 0U) output[row] = value;
}

__global__ void swiglu(float* gate_up, std::uint32_t width) {
  for (std::uint32_t index =
           static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < width; index += blockDim.x * gridDim.x) {
    const auto gate = gate_up[index];
    gate_up[index] = gate / (1.0F + expf(-gate)) * gate_up[width + index];
  }
}

__global__ void sum_four(const float* first, const float* second,
                         const float* third, const float* fourth,
                         float* output, std::uint32_t count) {
  for (std::uint32_t index =
           static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    output[index] = first[index] + second[index] + third[index] + fourth[index];
  }
}

std::size_t packed_bytes(std::uint32_t rows, std::uint32_t columns) {
  return static_cast<std::size_t>(rows) * columns / 2U;
}

std::size_t scale_bytes(std::uint32_t rows, std::uint32_t columns) {
  return static_cast<std::size_t>(rows) * columns / 32U;
}

class Shard final {
 public:
  Shard(int device, std::uint32_t intermediate)
      : device_(device), intermediate_(intermediate), use_dp4a_(device == 3),
        gate_up_weights_(device, packed_bytes(2U * intermediate, kHidden)),
        gate_up_scales_(device, scale_bytes(2U * intermediate, kHidden)),
        down_weights_(device, packed_bytes(kHidden, intermediate)),
        down_scales_(device, scale_bytes(kHidden, intermediate)),
        input_(device, kHidden * sizeof(float)),
        q_input_(device, kHidden), q_input_scale_(device, sizeof(float)),
        gate_up_(device, 2ULL * intermediate * sizeof(float)),
        q_intermediate_(device, intermediate),
        q_intermediate_scale_(device, sizeof(float)),
        output_(device, kHidden * sizeof(float)) {
    check(cudaSetDevice(device_), "select shard device");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device_), "query shard GPU");
    const bool compatible = device_ == 3
        ? properties.major == 6 && properties.minor == 1
        : properties.major == 6 && properties.minor == 0;
    if (!compatible) throw std::runtime_error("unexpected shard architecture");
    name_ = properties.name;
    check(cudaMemset(gate_up_weights_.as<void>(), 0x21,
                     packed_bytes(2U * intermediate_, kHidden)),
          "initialize gate/up weights");
    check(cudaMemset(gate_up_scales_.as<void>(), 127,
                     scale_bytes(2U * intermediate_, kHidden)),
          "initialize gate/up scales");
    check(cudaMemset(down_weights_.as<void>(), 0x43,
                     packed_bytes(kHidden, intermediate_)),
          "initialize down weights");
    check(cudaMemset(down_scales_.as<void>(), 127,
                     scale_bytes(kHidden, intermediate_)),
          "initialize down scales");
    check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
          "create shard stream");
  }

  ~Shard() {
    cudaSetDevice(device_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  Shard(const Shard&) = delete;
  Shard& operator=(const Shard&) = delete;

  void launch() {
    check(cudaSetDevice(device_), "select launching shard");
    quantize_q8<<<1U, kThreads, 0, stream_>>>(
        input_.as<float>(), q_input_.as<std::int8_t>(),
        q_input_scale_.as<float>(), kHidden);
    const auto gate_blocks =
        (2U * intermediate_ + kWarpsPerBlock - 1U) / kWarpsPerBlock;
    if (use_dp4a_) {
      fp4_q8_gemv<true><<<gate_blocks, kThreads, 0, stream_>>>(
          gate_up_weights_.as<std::uint8_t>(),
          gate_up_scales_.as<std::uint8_t>(), q_input_.as<std::int8_t>(),
          q_input_scale_.as<float>(), gate_up_.as<float>(),
          2U * intermediate_, kHidden);
    } else {
      fp4_q8_gemv<false><<<gate_blocks, kThreads, 0, stream_>>>(
          gate_up_weights_.as<std::uint8_t>(),
          gate_up_scales_.as<std::uint8_t>(), q_input_.as<std::int8_t>(),
          q_input_scale_.as<float>(), gate_up_.as<float>(),
          2U * intermediate_, kHidden);
    }
    swiglu<<<(intermediate_ + kThreads - 1U) / kThreads, kThreads, 0,
              stream_>>>(gate_up_.as<float>(), intermediate_);
    quantize_q8<<<1U, kThreads, 0, stream_>>>(
        gate_up_.as<float>(), q_intermediate_.as<std::int8_t>(),
        q_intermediate_scale_.as<float>(), intermediate_);
    const auto down_blocks =
        (kHidden + kWarpsPerBlock - 1U) / kWarpsPerBlock;
    if (use_dp4a_) {
      fp4_q8_gemv<true><<<down_blocks, kThreads, 0, stream_>>>(
          down_weights_.as<std::uint8_t>(), down_scales_.as<std::uint8_t>(),
          q_intermediate_.as<std::int8_t>(),
          q_intermediate_scale_.as<float>(), output_.as<float>(),
          kHidden, intermediate_);
    } else {
      fp4_q8_gemv<false><<<down_blocks, kThreads, 0, stream_>>>(
          down_weights_.as<std::uint8_t>(), down_scales_.as<std::uint8_t>(),
          q_intermediate_.as<std::int8_t>(),
          q_intermediate_scale_.as<float>(), output_.as<float>(),
          kHidden, intermediate_);
    }
    check(cudaPeekAtLastError(), "launch shard MLP");
  }

  void synchronize() {
    check(cudaSetDevice(device_), "select synchronizing shard");
    check(cudaStreamSynchronize(stream_), "synchronize shard");
  }

  int device() const { return device_; }
  cudaStream_t stream() const { return stream_; }
  float* input() const { return input_.as<float>(); }
  float* output() const { return output_.as<float>(); }
  const std::string& name() const { return name_; }
  std::uint32_t intermediate() const { return intermediate_; }

 private:
  int device_{};
  std::uint32_t intermediate_{};
  bool use_dp4a_{};
  std::string name_;
  DeviceAllocation gate_up_weights_;
  DeviceAllocation gate_up_scales_;
  DeviceAllocation down_weights_;
  DeviceAllocation down_scales_;
  DeviceAllocation input_;
  DeviceAllocation q_input_;
  DeviceAllocation q_input_scale_;
  DeviceAllocation gate_up_;
  DeviceAllocation q_intermediate_;
  DeviceAllocation q_intermediate_scale_;
  DeviceAllocation output_;
  cudaStream_t stream_{};
};

void enable_p100_peers() {
  for (int device = 0; device < 3; ++device) {
    check(cudaSetDevice(device), "select peer device");
    for (int peer = 0; peer < 3; ++peer) {
      if (device == peer) continue;
      const auto status = cudaDeviceEnablePeerAccess(peer, 0);
      if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) {
        check(status, "enable P100 peer access");
      }
      if (status == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
    }
  }
}

class MlpProgram final {
 public:
  MlpProgram()
      : host_input_(kHidden * sizeof(float)),
        host_output_(kHidden * sizeof(float)),
        peer_one_(0, kHidden * sizeof(float)),
        peer_two_(0, kHidden * sizeof(float)),
        p40_result_(0, kHidden * sizeof(float)) {
    enable_p100_peers();
    for (int device = 0; device < 4; ++device) {
      shards_.push_back(std::make_unique<Shard>(
          device, kIntermediateShards[device]));
    }
    std::vector<float> initial(kHidden);
    for (std::uint32_t index = 0; index < kHidden; ++index) {
      initial[index] = std::sin(static_cast<float>(index + 1U) * 0.013F) * 0.1F;
    }
    check(cudaSetDevice(0), "select initial device");
    check(cudaMemcpy(shards_[0]->input(), initial.data(),
                     initial.size() * sizeof(float), cudaMemcpyHostToDevice),
          "upload initial hidden state");
  }

  double run(std::uint32_t layers) {
    const auto begin = std::chrono::steady_clock::now();
    for (std::uint32_t layer = 0; layer < layers; ++layer) run_layer();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
  }

  const std::vector<std::unique_ptr<Shard>>& shards() const { return shards_; }

 private:
  void run_layer() {
    constexpr auto hidden_bytes = kHidden * sizeof(float);
    check(cudaSetDevice(1), "select P100 1 broadcast");
    check(cudaMemcpyPeerAsync(shards_[1]->input(), 1, shards_[0]->input(), 0,
                              hidden_bytes, shards_[1]->stream()),
          "broadcast hidden to P100 1");
    check(cudaSetDevice(2), "select P100 2 broadcast");
    check(cudaMemcpyPeerAsync(shards_[2]->input(), 2, shards_[0]->input(), 0,
                              hidden_bytes, shards_[2]->stream()),
          "broadcast hidden to P100 2");
    check(cudaSetDevice(0), "select P40 staging source");
    check(cudaMemcpyAsync(host_input_.as<void>(), shards_[0]->input(),
                          hidden_bytes, cudaMemcpyDeviceToHost,
                          shards_[0]->stream()),
          "stage P40 input from coordinator");
    check(cudaStreamSynchronize(shards_[0]->stream()),
          "complete P40 input staging");
    check(cudaSetDevice(3), "select P40 input");
    check(cudaMemcpyAsync(shards_[3]->input(), host_input_.as<void>(),
                          hidden_bytes, cudaMemcpyHostToDevice,
                          shards_[3]->stream()),
          "upload P40 input");

    for (auto& shard : shards_) shard->launch();
    for (auto& shard : shards_) shard->synchronize();

    check(cudaSetDevice(0), "select reduction coordinator");
    check(cudaMemcpyPeerAsync(peer_one_.as<void>(), 0, shards_[1]->output(), 1,
                              hidden_bytes, shards_[0]->stream()),
          "collect P100 1 output");
    check(cudaMemcpyPeerAsync(peer_two_.as<void>(), 0, shards_[2]->output(), 2,
                              hidden_bytes, shards_[0]->stream()),
          "collect P100 2 output");
    check(cudaSetDevice(3), "select P40 staging output");
    check(cudaMemcpyAsync(host_output_.as<void>(), shards_[3]->output(),
                          hidden_bytes, cudaMemcpyDeviceToHost,
                          shards_[3]->stream()),
          "download P40 output");
    check(cudaStreamSynchronize(shards_[3]->stream()),
          "complete P40 output staging");
    check(cudaSetDevice(0), "select P40 reduction upload");
    check(cudaMemcpyAsync(p40_result_.as<void>(), host_output_.as<void>(),
                          hidden_bytes, cudaMemcpyHostToDevice,
                          shards_[0]->stream()),
          "upload P40 reduction operand");
    sum_four<<<(kHidden + kThreads - 1U) / kThreads, kThreads, 0,
               shards_[0]->stream()>>>(
        shards_[0]->output(), peer_one_.as<float>(), peer_two_.as<float>(),
        p40_result_.as<float>(), shards_[0]->input(), kHidden);
    check(cudaPeekAtLastError(), "launch MLP reduction");
    check(cudaStreamSynchronize(shards_[0]->stream()),
          "complete MLP reduction");
  }

  PinnedAllocation host_input_;
  PinnedAllocation host_output_;
  DeviceAllocation peer_one_;
  DeviceAllocation peer_two_;
  DeviceAllocation p40_result_;
  std::vector<std::unique_ptr<Shard>> shards_;
};

}  // namespace

int main() try {
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "enumerate GPUs");
  if (device_count < 4) throw std::runtime_error("four GPUs are required");
  MlpProgram program;
  static_cast<void>(program.run(1U));
  const double milliseconds = program.run(kLayers);
  std::cout << std::fixed << std::setprecision(3);
  for (const auto& shard : program.shards()) {
    std::cout << "device=" << shard->device() << " name=\"" << shard->name()
              << "\" intermediate=" << shard->intermediate() << '\n';
  }
  std::cout << "layers=" << kLayers
            << " integrated_mlp_ms=" << milliseconds
            << " per_layer_ms=" << milliseconds / kLayers
            << " mlp_only_tok_s=" << 1000.0 / milliseconds << '\n';
  return EXIT_SUCCESS;
} catch (const std::exception& error) {
  std::cerr << "heterogeneous_mlp_gate: " << error.what() << '\n';
  return EXIT_FAILURE;
}
