#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kThreads = 256U;
constexpr std::uint32_t kWarp = 32U;
constexpr std::uint32_t kWarps = kThreads / kWarp;
constexpr std::uint32_t kQueryHeads = 24U;
constexpr std::uint32_t kKvHeads = 4U;
constexpr std::uint32_t kGroupedHeads = kQueryHeads / kKvHeads;
constexpr std::uint32_t kHeadDim = 256U;
constexpr std::uint32_t kAttentionLayers = 16U;
constexpr std::uint32_t kContextTokens = 262144U;
constexpr std::array<std::uint32_t, 3> kLocalTokens = {
    87382U, 87381U, 87381U};
static_assert(kLocalTokens[0] + kLocalTokens[1] + kLocalTokens[2] ==
              kContextTokens);

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void check(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) +
                             ": cuBLAS status " + std::to_string(status));
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

__device__ float warp_max(float value) {
  for (int offset = 16; offset != 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  return value;
}

__device__ float warp_sum(float value) {
  for (int offset = 16; offset != 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

__global__ void softmax_to_half(
    const float* __restrict__ scores,
    __half* __restrict__ probabilities,
    float* __restrict__ local_maxima,
    float* __restrict__ local_sums,
    std::uint32_t tokens) {
  __shared__ float warp_values[kWarps];
  __shared__ float maximum;
  __shared__ float denominator;
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto lane = static_cast<std::uint32_t>(threadIdx.x) & (kWarp - 1U);
  const auto warp = static_cast<std::uint32_t>(threadIdx.x) / kWarp;
  const auto* head_scores = scores + static_cast<std::size_t>(head) * tokens;
  auto* head_probabilities =
      probabilities + static_cast<std::size_t>(head) * tokens;

  float local_maximum = -FLT_MAX;
  for (std::uint32_t token = threadIdx.x; token < tokens;
       token += blockDim.x) {
    local_maximum = fmaxf(local_maximum, head_scores[token]);
  }
  local_maximum = warp_max(local_maximum);
  if (lane == 0U) warp_values[warp] = local_maximum;
  __syncthreads();
  if (warp == 0U) {
    local_maximum = lane < kWarps
        ? warp_values[lane]
        : -FLT_MAX;
    local_maximum = warp_max(local_maximum);
    if (lane == 0U) maximum = local_maximum;
  }
  __syncthreads();

  float local_sum = 0.0F;
  for (std::uint32_t token = threadIdx.x; token < tokens;
       token += blockDim.x) {
    const __half probability =
        __float2half_rn(expf(head_scores[token] - maximum));
    head_probabilities[token] = probability;
    local_sum += __half2float(probability);
  }
  local_sum = warp_sum(local_sum);
  if (lane == 0U) warp_values[warp] = local_sum;
  __syncthreads();
  if (warp == 0U) {
    local_sum = lane < kWarps ? warp_values[lane] : 0.0F;
    local_sum = warp_sum(local_sum);
    if (lane == 0U) denominator = local_sum;
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    local_maxima[head] = maximum;
    local_sums[head] = denominator;
  }
}

__global__ void combine_devices(
    const float* maxima0, const float* sums0, const float* outputs0,
    const float* maxima1, const float* sums1, const float* outputs1,
    const float* maxima2, const float* sums2, const float* outputs2,
    const float* gate, float* output) {
  __shared__ float maximum;
  __shared__ float denominator;
  __shared__ float scale0;
  __shared__ float scale1;
  __shared__ float scale2;
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (dimension == 0U) {
    maximum = fmaxf(maxima0[head], fmaxf(maxima1[head], maxima2[head]));
    scale0 = expf(maxima0[head] - maximum);
    scale1 = expf(maxima1[head] - maximum);
    scale2 = expf(maxima2[head] - maximum);
    denominator = sums0[head] * scale0 + sums1[head] * scale1 +
                  sums2[head] * scale2;
  }
  __syncthreads();
  if (dimension >= kHeadDim) return;
  const auto index = static_cast<std::size_t>(head) * kHeadDim + dimension;
  const float numerator = outputs0[index] * scale0 + outputs1[index] * scale1 +
                          outputs2[index] * scale2;
  output[index] = numerator / denominator /
                  (1.0F + expf(-gate[index]));
}

void enable_peers() {
  for (int device = 0; device < 3; ++device) {
    check(cudaSetDevice(device), "select BLAS attention peer device");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device),
          "query BLAS attention GPU");
    if (properties.major != 6 || properties.minor != 0) {
      throw std::runtime_error("devices 0-2 must be SM60 P100 GPUs");
    }
    for (int peer = 0; peer < 3; ++peer) {
      if (peer == device) continue;
      int available = 0;
      check(cudaDeviceCanAccessPeer(&available, device, peer),
            "query BLAS attention peer access");
      if (available == 0) {
        throw std::runtime_error("three-P100 peer matrix is incomplete");
      }
      const auto status = cudaDeviceEnablePeerAccess(peer, 0);
      if (status == cudaErrorPeerAccessAlreadyEnabled) {
        static_cast<void>(cudaGetLastError());
      } else {
        check(status, "enable BLAS attention peer access");
      }
    }
  }
}

class AttentionShard final {
 public:
  AttentionShard(int device, std::uint32_t tokens, std::uint32_t layers)
      : device_(device), tokens_(tokens), layers_(layers),
        head_values_(static_cast<std::size_t>(tokens) * kHeadDim),
        layer_values_(static_cast<std::size_t>(kKvHeads) * head_values_),
        kv_(device, 2ULL * layers * layer_values_ * sizeof(__half)),
        query_(device, kQueryHeads * kHeadDim * sizeof(__half)),
        gate_(device, kQueryHeads * kHeadDim * sizeof(float)),
        scores_(device, static_cast<std::size_t>(kQueryHeads) * tokens *
                            sizeof(float)),
        probabilities_(device, static_cast<std::size_t>(kQueryHeads) * tokens *
                                   sizeof(__half)),
        local_maxima_(device, kQueryHeads * sizeof(float)),
        local_sums_(device, kQueryHeads * sizeof(float)),
        local_outputs_(device, kQueryHeads * kHeadDim * sizeof(float)) {
    check(cudaSetDevice(device_), "select BLAS attention shard");
    check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
          "create BLAS attention stream");
    check(cublasCreate(&handle_), "create BLAS attention handle");
    check(cublasSetStream(handle_, stream_), "bind BLAS attention stream");
    check(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH),
          "set BLAS attention math mode");
    check(cudaMemsetAsync(kv_.as<void>(), 0,
                         2ULL * layers_ * layer_values_ * sizeof(__half),
                         stream_),
          "initialize BLAS exact F16 KV");
    check(cudaStreamSynchronize(stream_),
          "finish BLAS exact F16 KV initialization");
  }

  ~AttentionShard() {
    cudaSetDevice(device_);
    if (handle_ != nullptr) cublasDestroy(handle_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  AttentionShard(const AttentionShard&) = delete;
  AttentionShard& operator=(const AttentionShard&) = delete;

  void launch(std::uint32_t layer) {
    check(cudaSetDevice(device_), "select launching BLAS attention shard");
    const auto* keys = kv_.as<__half>() +
        static_cast<std::size_t>(layer) * 2U * layer_values_;
    const auto* values = keys + layer_values_;
    constexpr float score_alpha = 0.0625F;
    constexpr float one = 1.0F;
    constexpr float zero = 0.0F;
    const long long kv_stride = static_cast<long long>(head_values_);
    const long long query_stride =
        static_cast<long long>(kGroupedHeads) * kHeadDim;
    const long long score_stride =
        static_cast<long long>(kGroupedHeads) * tokens_;
    check(cublasGemmStridedBatchedEx(
              handle_, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(tokens_),
              static_cast<int>(kGroupedHeads), static_cast<int>(kHeadDim),
              &score_alpha, keys, CUDA_R_16F, static_cast<int>(kHeadDim),
              kv_stride, query_.as<__half>(), CUDA_R_16F,
              static_cast<int>(kHeadDim), query_stride, &zero,
              scores_.as<float>(), CUDA_R_32F, static_cast<int>(tokens_),
              score_stride, static_cast<int>(kKvHeads), CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT),
          "launch BLAS grouped QK");
    softmax_to_half<<<kQueryHeads, kThreads, 0, stream_>>>(
        scores_.as<float>(), probabilities_.as<__half>(),
        local_maxima_.as<float>(), local_sums_.as<float>(), tokens_);
    check(cudaPeekAtLastError(), "launch BLAS attention softmax");
    check(cublasGemmStridedBatchedEx(
              handle_, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(kHeadDim),
              static_cast<int>(kGroupedHeads), static_cast<int>(tokens_), &one,
              values, CUDA_R_16F, static_cast<int>(kHeadDim), kv_stride,
              probabilities_.as<__half>(), CUDA_R_16F,
              static_cast<int>(tokens_), score_stride, &zero,
              local_outputs_.as<float>(), CUDA_R_32F,
              static_cast<int>(kHeadDim), query_stride,
              static_cast<int>(kKvHeads), CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT),
          "launch BLAS grouped PV");
  }

  void synchronize() {
    check(cudaSetDevice(device_), "select synchronizing BLAS attention shard");
    check(cudaStreamSynchronize(stream_), "synchronize BLAS attention shard");
  }

  void upload_kv(const std::vector<__half>& values) {
    if (values.size() != 2ULL * layers_ * layer_values_) {
      throw std::runtime_error("BLAS attention KV size mismatch");
    }
    check(cudaSetDevice(device_), "select BLAS attention KV upload");
    check(cudaMemcpy(kv_.as<void>(), values.data(),
                     values.size() * sizeof(__half), cudaMemcpyHostToDevice),
          "upload BLAS attention KV");
  }

  void upload_query(const std::vector<__half>& query,
                    const std::vector<float>& gate) {
    if (query.size() != static_cast<std::size_t>(kQueryHeads) * kHeadDim ||
        gate.size() != query.size()) {
      throw std::runtime_error("BLAS attention query size mismatch");
    }
    check(cudaSetDevice(device_), "select BLAS attention query upload");
    check(cudaMemcpy(query_.as<void>(), query.data(),
                     query.size() * sizeof(__half), cudaMemcpyHostToDevice),
          "upload BLAS attention query");
    check(cudaMemcpy(gate_.as<void>(), gate.data(), gate.size() * sizeof(float),
                     cudaMemcpyHostToDevice),
          "upload BLAS attention gate");
  }

  int device() const { return device_; }
  cudaStream_t stream() const { return stream_; }
  __half* query() const { return query_.as<__half>(); }
  float* gate() const { return gate_.as<float>(); }
  float* maxima() const { return local_maxima_.as<float>(); }
  float* sums() const { return local_sums_.as<float>(); }
  float* outputs() const { return local_outputs_.as<float>(); }
  std::size_t kv_bytes() const {
    return 2ULL * layers_ * layer_values_ * sizeof(__half);
  }

 private:
  int device_{};
  std::uint32_t tokens_{};
  std::uint32_t layers_{};
  std::size_t head_values_{};
  std::size_t layer_values_{};
  DeviceAllocation kv_;
  DeviceAllocation query_;
  DeviceAllocation gate_;
  DeviceAllocation scores_;
  DeviceAllocation probabilities_;
  DeviceAllocation local_maxima_;
  DeviceAllocation local_sums_;
  DeviceAllocation local_outputs_;
  cudaStream_t stream_{};
  cublasHandle_t handle_{};
};

class DeviceCombiner final {
 public:
  DeviceCombiner()
      : maxima_one_(0, kQueryHeads * sizeof(float)),
        sums_one_(0, kQueryHeads * sizeof(float)),
        outputs_one_(0, kQueryHeads * kHeadDim * sizeof(float)),
        maxima_two_(0, kQueryHeads * sizeof(float)),
        sums_two_(0, kQueryHeads * sizeof(float)),
        outputs_two_(0, kQueryHeads * kHeadDim * sizeof(float)),
        output_(0, kQueryHeads * kHeadDim * sizeof(float)) {}

  void combine(AttentionShard& first, AttentionShard& second,
               AttentionShard& third) {
    constexpr auto scalar_bytes = kQueryHeads * sizeof(float);
    constexpr auto output_bytes = kQueryHeads * kHeadDim * sizeof(float);
    check(cudaSetDevice(0), "select BLAS attention combiner");
    check(cudaMemcpyPeerAsync(maxima_one_.as<void>(), 0, second.maxima(), 1,
                              scalar_bytes, first.stream()),
          "collect BLAS maxima from P100 1");
    check(cudaMemcpyPeerAsync(sums_one_.as<void>(), 0, second.sums(), 1,
                              scalar_bytes, first.stream()),
          "collect BLAS sums from P100 1");
    check(cudaMemcpyPeerAsync(outputs_one_.as<void>(), 0, second.outputs(), 1,
                              output_bytes, first.stream()),
          "collect BLAS outputs from P100 1");
    check(cudaMemcpyPeerAsync(maxima_two_.as<void>(), 0, third.maxima(), 2,
                              scalar_bytes, first.stream()),
          "collect BLAS maxima from P100 2");
    check(cudaMemcpyPeerAsync(sums_two_.as<void>(), 0, third.sums(), 2,
                              scalar_bytes, first.stream()),
          "collect BLAS sums from P100 2");
    check(cudaMemcpyPeerAsync(outputs_two_.as<void>(), 0, third.outputs(), 2,
                              output_bytes, first.stream()),
          "collect BLAS outputs from P100 2");
    combine_devices<<<kQueryHeads, kThreads, 0, first.stream()>>>(
        first.maxima(), first.sums(), first.outputs(), maxima_one_.as<float>(),
        sums_one_.as<float>(), outputs_one_.as<float>(), maxima_two_.as<float>(),
        sums_two_.as<float>(), outputs_two_.as<float>(), first.gate(),
        output_.as<float>());
    check(cudaPeekAtLastError(), "launch BLAS cross-device combine");
    check(cudaStreamSynchronize(first.stream()),
          "complete BLAS cross-device combine");
  }

  std::vector<float> download() const {
    check(cudaSetDevice(0), "select BLAS attention result download");
    std::vector<float> result(kQueryHeads * kHeadDim);
    check(cudaMemcpy(result.data(), output_.as<void>(),
                     result.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "download BLAS attention result");
    return result;
  }

 private:
  DeviceAllocation maxima_one_;
  DeviceAllocation sums_one_;
  DeviceAllocation outputs_one_;
  DeviceAllocation maxima_two_;
  DeviceAllocation sums_two_;
  DeviceAllocation outputs_two_;
  DeviceAllocation output_;
};

struct Query final {
  std::vector<__half> values;
  std::vector<float> gate;
};

Query make_query() {
  Query query;
  query.values.resize(static_cast<std::size_t>(kQueryHeads) * kHeadDim);
  query.gate.resize(query.values.size());
  for (std::size_t index = 0; index < query.values.size(); ++index) {
    query.values[index] = __float2half_rn(
        std::sin(static_cast<float>(index + 1U) * 0.0031F) * 0.08F);
    query.gate[index] =
        std::cos(static_cast<float>(index + 5U) * 0.0023F) * 0.1F;
  }
  return query;
}

void compare_vectors(const std::vector<float>& actual,
                     const std::vector<float>& expected,
                     const char* label, double max_limit,
                     double relative_limit) {
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
  std::cout << std::scientific << label << "_max_abs=" << maximum_error
            << ' ' << label << "_relative_l2=" << relative_l2 << '\n';
  if (maximum_error > max_limit || relative_l2 > relative_limit) {
    throw std::runtime_error(std::string(label) + " comparison failed");
  }
}

void validate_oracle() {
  constexpr std::uint32_t tokens = 32U;
  const auto query = make_query();
  std::array<std::unique_ptr<AttentionShard>, 3> shards;
  std::array<std::vector<__half>, 3> local_kv;
  for (int device = 0; device < 3; ++device) {
    shards[device] = std::make_unique<AttentionShard>(device, tokens, 1U);
    auto& kv = local_kv[device];
    const auto layer_values =
        static_cast<std::size_t>(kKvHeads) * tokens * kHeadDim;
    kv.resize(2ULL * layer_values);
    for (std::uint32_t head = 0; head < kKvHeads; ++head) {
      for (std::uint32_t token = 0; token < tokens; ++token) {
        const auto global_token = static_cast<std::uint32_t>(device) * tokens +
                                  token;
        for (std::uint32_t dimension = 0; dimension < kHeadDim; ++dimension) {
          const auto local =
              (static_cast<std::size_t>(head) * tokens + token) * kHeadDim +
              dimension;
          const auto seed =
              (static_cast<std::size_t>(global_token) * kKvHeads + head) *
                  kHeadDim +
              dimension;
          kv[local] = __float2half_rn(
              std::sin(static_cast<float>(seed + 3U) * 0.0017F) * 0.1F);
          kv[layer_values + local] = __float2half_rn(
              std::cos(static_cast<float>(seed + 7U) * 0.0013F) * 0.1F);
        }
      }
    }
    shards[device]->upload_kv(kv);
    shards[device]->upload_query(query.values, query.gate);
    shards[device]->launch(0U);
  }
  for (auto& shard : shards) shard->synchronize();
  DeviceCombiner combiner;
  combiner.combine(*shards[0], *shards[1], *shards[2]);
  const auto actual = combiner.download();

  std::vector<float> algorithm(kQueryHeads * kHeadDim, 0.0F);
  std::vector<float> semantic(algorithm.size(), 0.0F);
  for (std::uint32_t query_head = 0; query_head < kQueryHeads; ++query_head) {
    const auto kv_head = query_head / kGroupedHeads;
    std::array<double, 3> local_maxima;
    std::array<double, 3> local_sums{};
    std::array<std::vector<double>, 3> probabilities;
    std::array<std::vector<double>, 3> raw_scores;
    for (int device = 0; device < 3; ++device) {
      probabilities[device].resize(tokens);
      raw_scores[device].resize(tokens);
      local_maxima[device] = -std::numeric_limits<double>::infinity();
      for (std::uint32_t token = 0; token < tokens; ++token) {
        double score = 0.0;
        const auto& kv = local_kv[device];
        for (std::uint32_t dimension = 0; dimension < kHeadDim; ++dimension) {
          const auto local =
              (static_cast<std::size_t>(kv_head) * tokens + token) * kHeadDim +
              dimension;
          score += __half2float(query.values[
                       static_cast<std::size_t>(query_head) * kHeadDim +
                       dimension]) *
                   __half2float(kv[local]);
        }
        probabilities[device][token] = score * 0.0625;
        raw_scores[device][token] = probabilities[device][token];
        local_maxima[device] =
            std::max(local_maxima[device], probabilities[device][token]);
      }
      for (std::uint32_t token = 0; token < tokens; ++token) {
        const auto rounded = __float2half_rn(static_cast<float>(std::exp(
            probabilities[device][token] - local_maxima[device])));
        probabilities[device][token] = __half2float(rounded);
        local_sums[device] += probabilities[device][token];
      }
    }
    const double global_maximum = std::max(
        local_maxima[0], std::max(local_maxima[1], local_maxima[2]));
    double global_sum = 0.0;
    for (int device = 0; device < 3; ++device) {
      global_sum += local_sums[device] *
                    std::exp(local_maxima[device] - global_maximum);
    }
    double semantic_maximum = -std::numeric_limits<double>::infinity();
    for (const auto& scores : raw_scores) {
      for (const auto score : scores) {
        semantic_maximum = std::max(semantic_maximum, score);
      }
    }
    double semantic_sum = 0.0;
    for (const auto& scores : raw_scores) {
      for (const auto score : scores) {
        semantic_sum += std::exp(score - semantic_maximum);
      }
    }
    for (std::uint32_t dimension = 0; dimension < kHeadDim; ++dimension) {
      double numerator = 0.0;
      double semantic_numerator = 0.0;
      for (int device = 0; device < 3; ++device) {
        const auto& kv = local_kv[device];
        double local = 0.0;
        const auto layer_values =
            static_cast<std::size_t>(kKvHeads) * tokens * kHeadDim;
        for (std::uint32_t token = 0; token < tokens; ++token) {
          const auto offset = layer_values +
              (static_cast<std::size_t>(kv_head) * tokens + token) * kHeadDim +
              dimension;
          local += probabilities[device][token] * __half2float(kv[offset]);
          semantic_numerator +=
              std::exp(raw_scores[device][token] - semantic_maximum) *
              __half2float(kv[offset]);
        }
        numerator += local * std::exp(local_maxima[device] - global_maximum);
      }
      const auto index =
          static_cast<std::size_t>(query_head) * kHeadDim + dimension;
      algorithm[index] = static_cast<float>(
          numerator / global_sum / (1.0 + std::exp(-query.gate[index])));
      semantic[index] = static_cast<float>(
          semantic_numerator / semantic_sum /
          (1.0 + std::exp(-query.gate[index])));
    }
  }
  compare_vectors(actual, algorithm, "oracle", 3.0e-5, 3.0e-4);
  compare_vectors(actual, semantic, "semantic", 2.0e-4, 2.0e-3);
}

class AttentionProgram final {
 public:
  AttentionProgram() : query_(make_query()) {
    for (int device = 0; device < 3; ++device) {
      shards_[device] = std::make_unique<AttentionShard>(
          device, kLocalTokens[static_cast<std::size_t>(device)],
          kAttentionLayers);
      shards_[device]->upload_query(query_.values, query_.gate);
    }
  }

  double run() {
    const auto begin = std::chrono::steady_clock::now();
    for (std::uint32_t layer = 0; layer < kAttentionLayers; ++layer) {
      constexpr auto query_bytes = kQueryHeads * kHeadDim * sizeof(__half);
      constexpr auto gate_bytes = kQueryHeads * kHeadDim * sizeof(float);
      for (int device = 1; device < 3; ++device) {
        check(cudaSetDevice(device), "select BLAS attention broadcast target");
        check(cudaMemcpyPeerAsync(
                  shards_[device]->query(), device, shards_[0]->query(), 0,
                  query_bytes, shards_[device]->stream()),
              "broadcast BLAS attention query");
        check(cudaMemcpyPeerAsync(
                  shards_[device]->gate(), device, shards_[0]->gate(), 0,
                  gate_bytes, shards_[device]->stream()),
              "broadcast BLAS attention gate");
      }
      for (auto& shard : shards_) shard->launch(layer);
      for (auto& shard : shards_) shard->synchronize();
      combiner_.combine(*shards_[0], *shards_[1], *shards_[2]);
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
  }

  std::size_t kv_bytes() const {
    std::size_t total = 0;
    for (const auto& shard : shards_) total += shard->kv_bytes();
    return total;
  }

 private:
  Query query_;
  std::array<std::unique_ptr<AttentionShard>, 3> shards_;
  DeviceCombiner combiner_;
};

}  // namespace

int main() try {
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "enumerate GPUs");
  if (device_count < 3) throw std::runtime_error("three GPUs are required");
  enable_peers();
  validate_oracle();
  AttentionProgram program;
  static_cast<void>(program.run());
  const double milliseconds = program.run();
  const double gib = static_cast<double>(program.kv_bytes()) /
                     static_cast<double>(1ULL << 30U);
  std::cout << std::fixed << std::setprecision(3)
            << "context_tokens=" << kContextTokens
            << " attention_layers=" << kAttentionLayers
            << " exact_kv_gib=" << gib
            << " integrated_attention_ms=" << milliseconds
            << " effective_kv_gib_s=" << gib / (milliseconds / 1000.0)
            << " attention_only_tok_s=" << 1000.0 / milliseconds << '\n';
  return milliseconds <= 11.0 ? EXIT_SUCCESS : 2;
} catch (const std::exception& error) {
  std::cerr << "fp16_attention_blas_gate: " << error.what() << '\n';
  return EXIT_FAILURE;
}
