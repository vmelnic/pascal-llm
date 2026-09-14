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
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kThreads = 256U;
constexpr std::uint32_t kWarp = 32U;
constexpr std::uint32_t kTileTokens = 32U;
constexpr std::uint32_t kQueryHeads = 24U;
constexpr std::uint32_t kKvHeads = 4U;
constexpr std::uint32_t kGroupedHeads = kQueryHeads / kKvHeads;
constexpr std::uint32_t kHeadDim = 256U;
constexpr std::uint32_t kAttentionLayers = 16U;
constexpr std::uint32_t kContextTokens = 262144U;
constexpr std::uint32_t kSplitTokens = 1024U;
constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
constexpr std::array<std::uint32_t, 3> kLocalTokens = {
    87382U, 87381U, 87381U};
static_assert(kLocalTokens[0] + kLocalTokens[1] + kLocalTokens[2] ==
              kContextTokens);
static_assert(kGroupedHeads <= kThreads / kWarp);

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

__device__ float warp_max(float value) {
  for (int offset = 16; offset != 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  return value;
}

__global__ void grouped_attention_split(
    const float* __restrict__ query_and_gate,
    const __half* __restrict__ keys,
    const __half* __restrict__ values,
    float* __restrict__ partial_maxima,
    float* __restrict__ partial_sums,
    float* __restrict__ partial_outputs,
    std::uint32_t tokens,
    std::uint32_t split_tokens) {
  __shared__ __align__(16) __half key_tile[kTileTokens][kHeadDim];
  __shared__ __align__(16) __half value_tile[kTileTokens][kHeadDim];

  const auto kv_head = static_cast<std::uint32_t>(blockIdx.x);
  const auto split = static_cast<std::uint32_t>(blockIdx.y);
  const auto warp = static_cast<std::uint32_t>(threadIdx.x) / kWarp;
  const auto lane = static_cast<std::uint32_t>(threadIdx.x) & (kWarp - 1U);
  const auto first_token = split * split_tokens;
  const auto last_token = min(tokens, first_token + split_tokens);
  const bool active = warp < kGroupedHeads;
  const auto query_head = kv_head * kGroupedHeads + warp;
  const auto* query = active
      ? query_and_gate + static_cast<std::size_t>(query_head) * 2U * kHeadDim
      : query_and_gate;
  float maximum = kNegativeInfinity;
  float sum = 0.0F;
  float result[kHeadDim / kWarp]{};

  for (std::uint32_t tile_first = first_token; tile_first < last_token;
       tile_first += kTileTokens) {
    const auto tile_tokens = min(kTileTokens, last_token - tile_first);
    const auto tile_values = tile_tokens * kHeadDim;
    for (std::uint32_t item = threadIdx.x; item < 2U * tile_values;
         item += blockDim.x) {
      const bool is_value = item >= tile_values;
      const auto local_item = item - (is_value ? tile_values : 0U);
      const auto local_token = local_item / kHeadDim;
      const auto dimension = local_item % kHeadDim;
      const auto global_offset =
          (static_cast<std::size_t>(tile_first + local_token) * kKvHeads +
           kv_head) * kHeadDim + dimension;
      if (is_value) {
        value_tile[local_token][dimension] = values[global_offset];
      } else {
        key_tile[local_token][dimension] = keys[global_offset];
      }
    }
    __syncthreads();

    if (active) {
      float lane_score = kNegativeInfinity;
      for (std::uint32_t local_token = 0; local_token < tile_tokens;
           ++local_token) {
        float score = 0.0F;
#pragma unroll
        for (std::uint32_t local = 0; local < kHeadDim / kWarp; ++local) {
          const auto dimension = lane + local * kWarp;
          score = fmaf(query[dimension],
                       __half2float(key_tile[local_token][dimension]), score);
        }
        score = warp_sum(score);
        score = __shfl_sync(0xffffffffU, score, 0) * 0.0625F;
        if (lane == local_token) lane_score = score;
      }

      float tile_maximum = warp_max(lane_score);
      tile_maximum = __shfl_sync(0xffffffffU, tile_maximum, 0);
      const float next_maximum = fmaxf(maximum, tile_maximum);
      const float previous_scale = maximum == kNegativeInfinity
          ? 0.0F
          : expf(maximum - next_maximum);
      const float probability = lane < tile_tokens
          ? expf(lane_score - next_maximum)
          : 0.0F;
      const float tile_sum = warp_sum(probability);
      if (lane == 0U) {
        sum = sum * previous_scale + tile_sum;
        maximum = next_maximum;
      }
#pragma unroll
      for (std::uint32_t local = 0; local < kHeadDim / kWarp; ++local) {
        result[local] *= previous_scale;
      }
      for (std::uint32_t local_token = 0; local_token < tile_tokens;
           ++local_token) {
        const float token_probability =
            __shfl_sync(0xffffffffU, probability, local_token);
#pragma unroll
        for (std::uint32_t local = 0; local < kHeadDim / kWarp; ++local) {
          const auto dimension = lane + local * kWarp;
          result[local] = fmaf(
              token_probability,
              __half2float(value_tile[local_token][dimension]),
              result[local]);
        }
      }
    }
    __syncthreads();
  }

  if (!active) return;
  const auto partial =
      static_cast<std::size_t>(split) * kQueryHeads + query_head;
  if (lane == 0U) {
    partial_maxima[partial] = maximum;
    partial_sums[partial] = sum;
  }
#pragma unroll
  for (std::uint32_t local = 0; local < kHeadDim / kWarp; ++local) {
    const auto dimension = lane + local * kWarp;
    partial_outputs[partial * kHeadDim + dimension] = result[local];
  }
}

__global__ void combine_local_splits(
    const float* __restrict__ partial_maxima,
    const float* __restrict__ partial_sums,
    const float* __restrict__ partial_outputs,
    float* __restrict__ local_maxima,
    float* __restrict__ local_sums,
    float* __restrict__ local_outputs,
    std::uint32_t splits) {
  __shared__ float maximum;
  __shared__ float denominator;
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (dimension == 0U) {
    float found = kNegativeInfinity;
    for (std::uint32_t split = 0; split < splits; ++split) {
      found = fmaxf(found, partial_maxima[
          static_cast<std::size_t>(split) * kQueryHeads + head]);
    }
    float sum = 0.0F;
    for (std::uint32_t split = 0; split < splits; ++split) {
      const auto partial =
          static_cast<std::size_t>(split) * kQueryHeads + head;
      sum += partial_sums[partial] * expf(partial_maxima[partial] - found);
    }
    maximum = found;
    denominator = sum;
    local_maxima[head] = found;
    local_sums[head] = sum;
  }
  __syncthreads();
  if (dimension >= kHeadDim) return;
  float numerator = 0.0F;
  for (std::uint32_t split = 0; split < splits; ++split) {
    const auto partial =
        static_cast<std::size_t>(split) * kQueryHeads + head;
    numerator += partial_outputs[partial * kHeadDim + dimension] *
        expf(partial_maxima[partial] - maximum);
  }
  local_outputs[static_cast<std::size_t>(head) * kHeadDim + dimension] =
      numerator;
  static_cast<void>(denominator);
}

__global__ void combine_devices(
    const float* maxima0, const float* sums0, const float* outputs0,
    const float* maxima1, const float* sums1, const float* outputs1,
    const float* maxima2, const float* sums2, const float* outputs2,
    const float* query_and_gate, float* output) {
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
  const float gate = query_and_gate[
      static_cast<std::size_t>(head) * 2U * kHeadDim + kHeadDim + dimension];
  output[index] = numerator / denominator / (1.0F + expf(-gate));
}

void enable_peers() {
  for (int device = 0; device < 3; ++device) {
    check(cudaSetDevice(device), "select attention peer device");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "query attention GPU");
    if (properties.major != 6 || properties.minor != 0) {
      throw std::runtime_error("devices 0-2 must be SM60 P100 GPUs");
    }
    for (int peer = 0; peer < 3; ++peer) {
      if (peer == device) continue;
      int available = 0;
      check(cudaDeviceCanAccessPeer(&available, device, peer),
            "query attention peer access");
      if (available == 0) {
        throw std::runtime_error("three-P100 peer matrix is incomplete");
      }
      const auto status = cudaDeviceEnablePeerAccess(peer, 0);
      if (status == cudaErrorPeerAccessAlreadyEnabled) {
        static_cast<void>(cudaGetLastError());
      } else {
        check(status, "enable attention peer access");
      }
    }
  }
}

class AttentionShard final {
 public:
  AttentionShard(int device, std::uint32_t tokens, std::uint32_t layers,
                 std::uint32_t split_tokens)
      : device_(device), tokens_(tokens), layers_(layers),
        split_tokens_(split_tokens),
        splits_((tokens + split_tokens - 1U) / split_tokens),
        layer_values_(static_cast<std::size_t>(tokens) * kKvHeads * kHeadDim),
        kv_(device, 2ULL * layers * layer_values_ * sizeof(__half)),
        query_(device, 2ULL * kQueryHeads * kHeadDim * sizeof(float)),
        partial_maxima_(device,
            static_cast<std::size_t>(splits_) * kQueryHeads * sizeof(float)),
        partial_sums_(device,
            static_cast<std::size_t>(splits_) * kQueryHeads * sizeof(float)),
        partial_outputs_(device, static_cast<std::size_t>(splits_) *
            kQueryHeads * kHeadDim * sizeof(float)),
        local_maxima_(device, kQueryHeads * sizeof(float)),
        local_sums_(device, kQueryHeads * sizeof(float)),
        local_outputs_(device, kQueryHeads * kHeadDim * sizeof(float)) {
    check(cudaSetDevice(device_), "select attention shard");
    check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
          "create attention stream");
    check(cudaMemsetAsync(kv_.as<void>(), 0,
                         2ULL * layers_ * layer_values_ * sizeof(__half),
                         stream_),
          "initialize exact F16 KV");
    check(cudaStreamSynchronize(stream_), "finish exact F16 KV initialization");
  }

  ~AttentionShard() {
    cudaSetDevice(device_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  AttentionShard(const AttentionShard&) = delete;
  AttentionShard& operator=(const AttentionShard&) = delete;

  void launch(std::uint32_t layer) {
    check(cudaSetDevice(device_), "select launching attention shard");
    const auto* keys = kv_.as<__half>() +
        static_cast<std::size_t>(layer) * 2U * layer_values_;
    const auto* values = keys + layer_values_;
    grouped_attention_split<<<dim3(kKvHeads, splits_), kThreads, 0, stream_>>>(
        query_.as<float>(), keys, values, partial_maxima_.as<float>(),
        partial_sums_.as<float>(), partial_outputs_.as<float>(), tokens_,
        split_tokens_);
    combine_local_splits<<<kQueryHeads, kThreads, 0, stream_>>>(
        partial_maxima_.as<float>(), partial_sums_.as<float>(),
        partial_outputs_.as<float>(), local_maxima_.as<float>(),
        local_sums_.as<float>(), local_outputs_.as<float>(), splits_);
    check(cudaPeekAtLastError(), "launch exact F16 attention shard");
  }

  void synchronize() {
    check(cudaSetDevice(device_), "select synchronizing attention shard");
    check(cudaStreamSynchronize(stream_), "synchronize attention shard");
  }

  void upload_kv(const std::vector<__half>& values) {
    if (values.size() != 2ULL * layers_ * layer_values_) {
      throw std::runtime_error("attention oracle KV size mismatch");
    }
    check(cudaSetDevice(device_), "select attention KV upload");
    check(cudaMemcpy(kv_.as<void>(), values.data(),
                     values.size() * sizeof(__half), cudaMemcpyHostToDevice),
          "upload attention oracle KV");
  }

  void upload_query(const std::vector<float>& query) {
    if (query.size() != 2ULL * kQueryHeads * kHeadDim) {
      throw std::runtime_error("attention query size mismatch");
    }
    check(cudaSetDevice(device_), "select attention query upload");
    check(cudaMemcpy(query_.as<void>(), query.data(),
                     query.size() * sizeof(float), cudaMemcpyHostToDevice),
          "upload attention query");
  }

  int device() const { return device_; }
  cudaStream_t stream() const { return stream_; }
  float* query() const { return query_.as<float>(); }
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
  std::uint32_t split_tokens_{};
  std::uint32_t splits_{};
  std::size_t layer_values_{};
  DeviceAllocation kv_;
  DeviceAllocation query_;
  DeviceAllocation partial_maxima_;
  DeviceAllocation partial_sums_;
  DeviceAllocation partial_outputs_;
  DeviceAllocation local_maxima_;
  DeviceAllocation local_sums_;
  DeviceAllocation local_outputs_;
  cudaStream_t stream_{};
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
    check(cudaSetDevice(0), "select attention combiner");
    check(cudaMemcpyPeerAsync(maxima_one_.as<void>(), 0, second.maxima(),
                              second.device(), scalar_bytes, first.stream()),
          "collect attention maxima from P100 1");
    check(cudaMemcpyPeerAsync(sums_one_.as<void>(), 0, second.sums(),
                              second.device(), scalar_bytes, first.stream()),
          "collect attention sums from P100 1");
    check(cudaMemcpyPeerAsync(outputs_one_.as<void>(), 0, second.outputs(),
                              second.device(), output_bytes, first.stream()),
          "collect attention outputs from P100 1");
    check(cudaMemcpyPeerAsync(maxima_two_.as<void>(), 0, third.maxima(),
                              third.device(), scalar_bytes, first.stream()),
          "collect attention maxima from P100 2");
    check(cudaMemcpyPeerAsync(sums_two_.as<void>(), 0, third.sums(),
                              third.device(), scalar_bytes, first.stream()),
          "collect attention sums from P100 2");
    check(cudaMemcpyPeerAsync(outputs_two_.as<void>(), 0, third.outputs(),
                              third.device(), output_bytes, first.stream()),
          "collect attention outputs from P100 2");
    combine_devices<<<kQueryHeads, kThreads, 0, first.stream()>>>(
        first.maxima(), first.sums(), first.outputs(), maxima_one_.as<float>(),
        sums_one_.as<float>(), outputs_one_.as<float>(), maxima_two_.as<float>(),
        sums_two_.as<float>(), outputs_two_.as<float>(), first.query(),
        output_.as<float>());
    check(cudaPeekAtLastError(), "launch cross-device attention combine");
    check(cudaStreamSynchronize(first.stream()),
          "complete cross-device attention combine");
  }

  std::vector<float> download() const {
    check(cudaSetDevice(0), "select attention result download");
    std::vector<float> result(kQueryHeads * kHeadDim);
    check(cudaMemcpy(result.data(), output_.as<void>(),
                     result.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "download attention result");
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

std::vector<float> make_query() {
  std::vector<float> query(2ULL * kQueryHeads * kHeadDim);
  for (std::size_t index = 0; index < query.size(); ++index) {
    query[index] = std::sin(static_cast<float>(index + 1U) * 0.0031F) * 0.08F;
  }
  return query;
}

void validate_oracle() {
  constexpr std::uint32_t tokens_per_device = 32U;
  constexpr std::uint32_t total_tokens = 3U * tokens_per_device;
  std::array<std::unique_ptr<AttentionShard>, 3> shards;
  const auto query = make_query();
  std::vector<__half> all_keys(
      static_cast<std::size_t>(total_tokens) * kKvHeads * kHeadDim);
  std::vector<__half> all_values(all_keys.size());
  for (std::uint32_t token = 0; token < total_tokens; ++token) {
    for (std::uint32_t head = 0; head < kKvHeads; ++head) {
      for (std::uint32_t dimension = 0; dimension < kHeadDim; ++dimension) {
        const auto index =
            (static_cast<std::size_t>(token) * kKvHeads + head) * kHeadDim +
            dimension;
        all_keys[index] = __float2half_rn(
            std::sin(static_cast<float>(index + 3U) * 0.0017F) * 0.1F);
        all_values[index] = __float2half_rn(
            std::cos(static_cast<float>(index + 7U) * 0.0013F) * 0.1F);
      }
    }
  }

  for (int device = 0; device < 3; ++device) {
    shards[device] = std::make_unique<AttentionShard>(
        device, tokens_per_device, 1U, tokens_per_device);
    std::vector<__half> local(2ULL * tokens_per_device * kKvHeads * kHeadDim);
    const auto source = static_cast<std::size_t>(device) * tokens_per_device *
                        kKvHeads * kHeadDim;
    const auto values_per_device =
        static_cast<std::size_t>(tokens_per_device) * kKvHeads * kHeadDim;
    std::copy_n(all_keys.begin() + source, values_per_device, local.begin());
    std::copy_n(all_values.begin() + source, values_per_device,
                local.begin() + values_per_device);
    shards[device]->upload_kv(local);
    shards[device]->upload_query(query);
    shards[device]->launch(0U);
  }
  for (auto& shard : shards) shard->synchronize();
  DeviceCombiner combiner;
  combiner.combine(*shards[0], *shards[1], *shards[2]);
  const auto actual = combiner.download();

  std::vector<float> expected(kQueryHeads * kHeadDim);
  for (std::uint32_t query_head = 0; query_head < kQueryHeads; ++query_head) {
    const auto kv_head = query_head / kGroupedHeads;
    const auto* query_head_values = query.data() +
        static_cast<std::size_t>(query_head) * 2U * kHeadDim;
    std::vector<double> scores(total_tokens);
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::uint32_t token = 0; token < total_tokens; ++token) {
      double score = 0.0;
      for (std::uint32_t dimension = 0; dimension < kHeadDim; ++dimension) {
        const auto index =
            (static_cast<std::size_t>(token) * kKvHeads + kv_head) * kHeadDim +
            dimension;
        score += static_cast<double>(query_head_values[dimension]) *
                 __half2float(all_keys[index]);
      }
      scores[token] = score * 0.0625;
      maximum = std::max(maximum, scores[token]);
    }
    double denominator = 0.0;
    for (const auto score : scores) denominator += std::exp(score - maximum);
    for (std::uint32_t dimension = 0; dimension < kHeadDim; ++dimension) {
      double numerator = 0.0;
      for (std::uint32_t token = 0; token < total_tokens; ++token) {
        const auto index =
            (static_cast<std::size_t>(token) * kKvHeads + kv_head) * kHeadDim +
            dimension;
        numerator += std::exp(scores[token] - maximum) *
                     __half2float(all_values[index]);
      }
      const double gate = query_head_values[kHeadDim + dimension];
      expected[static_cast<std::size_t>(query_head) * kHeadDim + dimension] =
          static_cast<float>(numerator / denominator /
                             (1.0 + std::exp(-gate)));
    }
  }

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
  if (maximum_error > 2.0e-5 || relative_l2 > 2.0e-4) {
    throw std::runtime_error("independent exact F16 attention oracle failed");
  }
}

class AttentionProgram final {
 public:
  AttentionProgram() : query_(make_query()) {
    for (int device = 0; device < 3; ++device) {
      shards_[device] = std::make_unique<AttentionShard>(
          device, kLocalTokens[static_cast<std::size_t>(device)],
          kAttentionLayers, kSplitTokens);
      shards_[device]->upload_query(query_);
    }
  }

  double run() {
    const auto begin = std::chrono::steady_clock::now();
    for (std::uint32_t layer = 0; layer < kAttentionLayers; ++layer) {
      constexpr auto query_bytes =
          2ULL * kQueryHeads * kHeadDim * sizeof(float);
      check(cudaSetDevice(1), "select P100 1 attention query broadcast");
      check(cudaMemcpyPeerAsync(shards_[1]->query(), 1, shards_[0]->query(), 0,
                                query_bytes, shards_[1]->stream()),
            "broadcast attention query to P100 1");
      check(cudaSetDevice(2), "select P100 2 attention query broadcast");
      check(cudaMemcpyPeerAsync(shards_[2]->query(), 2, shards_[0]->query(), 0,
                                query_bytes, shards_[2]->stream()),
            "broadcast attention query to P100 2");
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
  std::vector<float> query_;
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
  std::cerr << "fp16_attention_gate: " << error.what() << '\n';
  return EXIT_FAILURE;
}
