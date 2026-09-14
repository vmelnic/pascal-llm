#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class DeviceBuffer {
 public:
  DeviceBuffer(int device, std::size_t bytes) : device_(device) {
    cuda_check(cudaSetDevice(device_), "select device for allocation");
    cuda_check(cudaMalloc(&data_, bytes), "allocate device buffer");
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaSetDevice(device_);
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* get() const { return data_; }

 private:
  int device_{};
  void* data_{};
};

class Stream {
 public:
  explicit Stream(int device) : device_(device) {
    cuda_check(cudaSetDevice(device_), "select device for stream");
    cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "create stream");
  }

  ~Stream() {
    if (stream_ != nullptr) {
      cudaSetDevice(device_);
      cudaStreamDestroy(stream_);
    }
  }

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  cudaStream_t get() const { return stream_; }

 private:
  int device_{};
  cudaStream_t stream_{};
};

struct TimedResult {
  double microseconds{};
  double gib_per_second{};
};

int repetitions_for(std::size_t bytes) {
  if (bytes <= 16U * 1024U) return 4000;
  if (bytes <= 64U * 1024U) return 2000;
  if (bytes <= 1024U * 1024U) return 400;
  return 16;
}

TimedResult measure_peer_copy(int source, int destination, const void* source_data,
                              void* destination_data, std::size_t bytes) {
  const int repetitions = repetitions_for(bytes);
  cuda_check(cudaSetDevice(destination), "select peer-copy destination");
  Stream stream(destination);
  cudaEvent_t begin{};
  cudaEvent_t end{};
  cuda_check(cudaEventCreate(&begin), "create begin event");
  cuda_check(cudaEventCreate(&end), "create end event");

  cuda_check(cudaMemcpyPeerAsync(destination_data, destination, source_data,
                                 source, bytes, stream.get()),
             "warm peer copy");
  cuda_check(cudaStreamSynchronize(stream.get()), "synchronize warm peer copy");
  cuda_check(cudaEventRecord(begin, stream.get()), "record peer-copy begin");
  for (int iteration = 0; iteration < repetitions; ++iteration) {
    cuda_check(cudaMemcpyPeerAsync(destination_data, destination, source_data,
                                   source, bytes, stream.get()),
               "enqueue peer copy");
  }
  cuda_check(cudaEventRecord(end, stream.get()), "record peer-copy end");
  cuda_check(cudaEventSynchronize(end), "synchronize peer-copy end");

  float milliseconds = 0.0F;
  cuda_check(cudaEventElapsedTime(&milliseconds, begin, end),
             "measure peer-copy time");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);

  const double seconds_per_copy =
      static_cast<double>(milliseconds) / 1000.0 / repetitions;
  return {
      seconds_per_copy * 1.0e6,
      static_cast<double>(bytes) / static_cast<double>(1ULL << 30U) /
          seconds_per_copy,
  };
}

TimedResult measure_host_leg(int device, void* host, void* device_data,
                             std::size_t bytes, cudaMemcpyKind kind) {
  const int repetitions = repetitions_for(bytes);
  cuda_check(cudaSetDevice(device), "select host-copy device");
  Stream stream(device);
  cudaEvent_t begin{};
  cudaEvent_t end{};
  cuda_check(cudaEventCreate(&begin), "create host begin event");
  cuda_check(cudaEventCreate(&end), "create host end event");

  void* destination = kind == cudaMemcpyDeviceToHost ? host : device_data;
  const void* source = kind == cudaMemcpyDeviceToHost ? device_data : host;
  cuda_check(cudaMemcpyAsync(destination, source, bytes, kind, stream.get()),
             "warm host copy");
  cuda_check(cudaStreamSynchronize(stream.get()), "synchronize warm host copy");
  cuda_check(cudaEventRecord(begin, stream.get()), "record host-copy begin");
  for (int iteration = 0; iteration < repetitions; ++iteration) {
    cuda_check(cudaMemcpyAsync(destination, source, bytes, kind, stream.get()),
               "enqueue host copy");
  }
  cuda_check(cudaEventRecord(end, stream.get()), "record host-copy end");
  cuda_check(cudaEventSynchronize(end), "synchronize host-copy end");

  float milliseconds = 0.0F;
  cuda_check(cudaEventElapsedTime(&milliseconds, begin, end),
             "measure host-copy time");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);

  const double seconds_per_copy =
      static_cast<double>(milliseconds) / 1000.0 / repetitions;
  return {
      seconds_per_copy * 1.0e6,
      static_cast<double>(bytes) / static_cast<double>(1ULL << 30U) /
          seconds_per_copy,
  };
}

void enable_supported_peers(int device_count,
                            const std::vector<std::vector<int>>& access) {
  for (int device = 0; device < device_count; ++device) {
    cuda_check(cudaSetDevice(device), "select peer owner");
    for (int peer = 0; peer < device_count; ++peer) {
      if (device == peer || access[device][peer] == 0) continue;
      const auto status = cudaDeviceEnablePeerAccess(peer, 0);
      if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) {
        cuda_check(status, "enable peer access");
      }
      if (status == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
    }
  }
}

}  // namespace

int main() try {
  int device_count = 0;
  cuda_check(cudaGetDeviceCount(&device_count), "enumerate CUDA devices");
  if (device_count < 2) {
    throw std::runtime_error("at least two CUDA devices are required");
  }

  constexpr std::size_t maximum_bytes = 128ULL << 20U;
  std::vector<DeviceBuffer*> buffers;
  std::vector<std::vector<int>> access(
      device_count, std::vector<int>(device_count, 0));

  std::cout << "devices=" << device_count << '\n';
  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device),
               "read device properties");
    std::cout << "device=" << device << " name=\"" << properties.name
              << "\" sm=" << properties.major << properties.minor
              << " total_mib=" << properties.totalGlobalMem / (1ULL << 20U)
              << '\n';
    buffers.push_back(new DeviceBuffer(device, maximum_bytes));
    cuda_check(cudaMemset(buffers.back()->get(), 0x40 + device, maximum_bytes),
               "initialize device buffer");
  }

  for (int device = 0; device < device_count; ++device) {
    for (int peer = 0; peer < device_count; ++peer) {
      if (device != peer) {
        cuda_check(cudaDeviceCanAccessPeer(&access[device][peer], device, peer),
                   "query peer access");
      }
      std::cout << "peer_access from=" << device << " to=" << peer
                << " supported=" << access[device][peer] << '\n';
    }
  }
  enable_supported_peers(device_count, access);

  void* pinned_host = nullptr;
  cuda_check(cudaHostAlloc(&pinned_host, maximum_bytes, cudaHostAllocPortable),
             "allocate portable pinned host buffer");

  const std::vector<std::size_t> sizes = {
      4ULL << 10U,
      16ULL << 10U,
      64ULL << 10U,
      1ULL << 20U,
      maximum_bytes,
  };
  std::cout << std::fixed << std::setprecision(3);
  for (int source = 0; source < device_count; ++source) {
    for (int destination = 0; destination < device_count; ++destination) {
      if (source == destination) continue;
      for (const auto bytes : sizes) {
        if (access[destination][source] != 0) {
          const auto result = measure_peer_copy(
              source, destination, buffers[source]->get(),
              buffers[destination]->get(), bytes);
          std::cout << "copy path=peer source=" << source
                    << " destination=" << destination << " bytes=" << bytes
                    << " us=" << result.microseconds
                    << " gib_s=" << result.gib_per_second << '\n';
        } else {
          const auto device_to_host = measure_host_leg(
              source, pinned_host, buffers[source]->get(), bytes,
              cudaMemcpyDeviceToHost);
          const auto host_to_device = measure_host_leg(
              destination, pinned_host, buffers[destination]->get(), bytes,
              cudaMemcpyHostToDevice);
          std::cout << "copy path=host-staged source=" << source
                    << " destination=" << destination << " bytes=" << bytes
                    << " d2h_us=" << device_to_host.microseconds
                    << " d2h_gib_s=" << device_to_host.gib_per_second
                    << " h2d_us=" << host_to_device.microseconds
                    << " h2d_gib_s=" << host_to_device.gib_per_second
                    << " serial_us="
                    << device_to_host.microseconds + host_to_device.microseconds
                    << '\n';
        }
      }
    }
  }

  cudaFreeHost(pinned_host);
  for (auto* buffer : buffers) delete buffer;
  return EXIT_SUCCESS;
} catch (const std::exception& error) {
  std::cerr << "p2p_probe: " << error.what() << '\n';
  return EXIT_FAILURE;
}

