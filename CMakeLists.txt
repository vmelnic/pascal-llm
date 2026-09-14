cmake_minimum_required(VERSION 3.24)

project(pascal_llm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(p2p_probe runtime/tools/p2p_probe.cu)
set_target_properties(p2p_probe PROPERTIES
  CUDA_ARCHITECTURES "60;61"
  CUDA_SEPARABLE_COMPILATION OFF
)

add_executable(fp4_q8_gate runtime/tools/fp4_q8_gate.cu)
set_target_properties(fp4_q8_gate PROPERTIES
  CUDA_ARCHITECTURES "60;61"
  CUDA_SEPARABLE_COMPILATION OFF
)

add_executable(fp4_q8_sm61_gate runtime/tools/fp4_q8_sm61_gate.cu)
set_target_properties(fp4_q8_sm61_gate PROPERTIES
  CUDA_ARCHITECTURES "61"
  CUDA_SEPARABLE_COMPILATION OFF
)

add_executable(heterogeneous_mlp_gate runtime/tools/heterogeneous_mlp_gate.cu)
set_target_properties(heterogeneous_mlp_gate PROPERTIES
  CUDA_ARCHITECTURES "60;61"
  CUDA_SEPARABLE_COMPILATION OFF
)

add_executable(fp16_mlp_gate runtime/tools/fp16_mlp_gate.cu)
set_target_properties(fp16_mlp_gate PROPERTIES
  CUDA_ARCHITECTURES "60"
  CUDA_SEPARABLE_COMPILATION OFF
)
target_link_libraries(fp16_mlp_gate PRIVATE CUDA::cublas)

add_executable(fp16_mmv_gate runtime/tools/fp16_mmv_gate.cu)
set_target_properties(fp16_mmv_gate PROPERTIES
  CUDA_ARCHITECTURES "60"
  CUDA_SEPARABLE_COMPILATION OFF
)

add_executable(fp16_attention_gate runtime/tools/fp16_attention_gate.cu)
set_target_properties(fp16_attention_gate PROPERTIES
  CUDA_ARCHITECTURES "60"
  CUDA_SEPARABLE_COMPILATION OFF
)

add_executable(fp16_attention_blas_gate
  runtime/tools/fp16_attention_blas_gate.cu)
set_target_properties(fp16_attention_blas_gate PROPERTIES
  CUDA_ARCHITECTURES "60"
  CUDA_SEPARABLE_COMPILATION OFF
)
target_link_libraries(fp16_attention_blas_gate PRIVATE CUDA::cublas)

target_compile_options(p2p_probe PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>
)

target_compile_options(fp4_q8_gate PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>
)

target_compile_options(fp4_q8_sm61_gate PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>
)

target_compile_options(heterogeneous_mlp_gate PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>
)

target_compile_options(fp16_mlp_gate PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>
)

target_compile_options(fp16_mmv_gate PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>
)

target_compile_options(fp16_attention_gate PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>
)

target_compile_options(fp16_attention_blas_gate PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>
)
