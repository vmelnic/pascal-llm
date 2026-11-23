#!/usr/bin/env bash
set -euo pipefail

# Separate, unpatched upstream llama.cpp tree for architectures that the pinned
# P100 patchset source (ops/install-llama.sh) predates. It never touches
# work/llama.cpp or its build/ and build-sm61/ artifacts, so existing profiles
# keep their qualified binaries. Only profiles that explicitly select
# PASCAL_LLAMA_BUILD_DIR=${PASCAL_LLAMA_ROOT}-next/build use it.
#
# The P100 patchset (rebased onto v0.6.0) does not apply to this upstream
# revision: 30 of its 43 patches fail a clean dry run. This tree therefore uses
# upstream SM60 kernels only, plus the pinned local patches in
# patches/llama-next/.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# Upstream master including Inkling 2 Mini (#32694) and its loop-slot fix (#32704).
llama_commit="${PASCAL_LLAMA_NEXT_COMMIT:-212c547fe908b288f6659d48741e6ffac1ff409c}"
llama_root="${PASCAL_LLAMA_NEXT_ROOT:-${repo_root}/work/llama.cpp-next}"
build_dir="${llama_root}/build"
local_patch_root="${repo_root}/patches/llama-next"
build_jobs="${PASCAL_BUILD_JOBS:-$(nproc)}"
[[ "${build_jobs}" =~ ^[1-9][0-9]*$ ]] || {
  echo "PASCAL_BUILD_JOBS must be a positive integer" >&2
  exit 2
}

for command_name in cmake git ninja nvcc gcc-13 g++-13 cuobjdump sha256sum; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "missing required command: ${command_name}" >&2
    exit 2
  }
done
nvcc --version | grep -Eq 'release 12\.' || {
  echo "Pascal builds require CUDA 12.x" >&2
  exit 2
}

if [[ ! -d "${llama_root}/.git" ]]; then
  mkdir -p "$(dirname "${llama_root}")"
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "${llama_root}"
fi

git -C "${llama_root}" fetch --depth=1 origin "${llama_commit}"
git -C "${llama_root}" checkout --detach --force "${llama_commit}"
git -C "${llama_root}" clean -fd
[[ "$(git -C "${llama_root}" rev-parse HEAD)" == "${llama_commit}" ]] || {
  echo "llama.cpp-next revision mismatch" >&2
  exit 2
}

mapfile -t local_patch_files < <(find "${local_patch_root}" -maxdepth 1 -type f -name '*.patch' | sort)
for patch_file in "${local_patch_files[@]}"; do
  git -C "${llama_root}" apply --check "${patch_file}" || {
    echo "llama.cpp-next patch does not apply cleanly: ${patch_file}" >&2
    exit 2
  }
  git -C "${llama_root}" apply "${patch_file}"
done

# CUDA 12.4 requires the GCC 13 host toolchain on this host.
cmake -S "${llama_root}" -B "${build_dir}" -G Ninja \
  -DCMAKE_C_COMPILER="$(command -v gcc-13)" \
  -DCMAKE_CXX_COMPILER="$(command -v g++-13)" \
  -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++-13)" \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NCCL=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=60 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF

cmake --build "${build_dir}" \
  --target llama-server llama-bench llama-perplexity \
  -j "${build_jobs}"

# Capture first: `grep -q` closing the pipe early would fail under pipefail.
cuda_images="$(cuobjdump --list-elf "${build_dir}/bin/libggml-cuda.so")"
grep -q '\.sm_60\.cubin' <<< "${cuda_images}" || {
  echo "llama.cpp-next CUDA library has no SM60 image" >&2
  exit 2
}
sha256sum "${build_dir}/bin/llama-server" "${build_dir}/bin/libggml-cuda.so"
"${build_dir}/bin/llama-server" --version
