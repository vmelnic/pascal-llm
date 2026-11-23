#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
llama_root="${PASCAL_LLAMA_ROOT:-${repo_root}/work/llama.cpp}"
build_dir="${llama_root}/build-sm61"
expected_commit=5266f24da75dc449bd56cbed7addb9c8e4a6a73e
evidence_dir="${repo_root}/out/llama-p40-build"
build_jobs="${PASCAL_BUILD_JOBS:-$(nproc)}"
[[ "${build_jobs}" =~ ^[1-9][0-9]*$ ]] || {
  echo "PASCAL_BUILD_JOBS must be a positive integer" >&2
  exit 2
}

# Reuse the already prepared and patched source without resetting it or
# touching the existing SM60 build. Stop Strata before compiling with all
# available CPU threads; PASCAL_BUILD_JOBS can bound parallelism if needed.
[[ "$(git -C "${llama_root}" rev-parse HEAD)" == "${expected_commit}" ]] || {
  echo "P40 build requires the deployed pinned llama.cpp source" >&2
  exit 2
}
for executable in cmake ninja nvcc cuobjdump gcc-13 g++-13 sha256sum; do
  command -v "${executable}" >/dev/null || {
    echo "missing P40 build dependency: ${executable}" >&2
    exit 2
  }
done
[[ -x "${llama_root}/build/bin/llama-server" ]] || {
  echo "existing P100 build is missing; prepare it with install-llama.sh first" >&2
  exit 2
}
mkdir -p "${evidence_dir}"
sha256sum \
  "${llama_root}/build/bin/llama-server" \
  "${llama_root}/build/bin/libggml-cuda.so" \
  > "${evidence_dir}/p100-before.sha256"
git -C "${llama_root}" diff --binary > "${evidence_dir}/source.patch"
git -C "${llama_root}" ls-files --others --exclude-standard \
  > "${evidence_dir}/source-untracked.txt"

# Keep the embedded UI matched to this pinned source. The moving upstream
# download no longer satisfies its asset manifest. Reuse only static assets;
# the current source's embedder validates and compiles them into the new build.
ui_source="${llama_root}/build/tools/ui/dist"
[[ -f "${ui_source}/index.html" ]] || {
  echo "compatible static UI assets are missing from the existing build" >&2
  exit 2
}
mkdir -p "${build_dir}/tools/ui"
ui_staging="$(mktemp -d "${build_dir}/tools/ui/dist-stage.XXXXXX")"
cp -a "${ui_source}/." "${ui_staging}/"
if [[ -e "${build_dir}/tools/ui/dist" ]]; then
  mv "${build_dir}/tools/ui/dist" "${ui_staging}.previous"
fi
mv "${ui_staging}" "${build_dir}/tools/ui/dist"

cmake -S "${llama_root}" -B "${build_dir}" -G Ninja \
  -DCMAKE_C_COMPILER="$(command -v gcc-13)" \
  -DCMAKE_CXX_COMPILER="$(command -v g++-13)" \
  -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++-13)" \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NCCL=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build "${build_dir}" --target llama-server -j "${build_jobs}"

sha256sum --check "${evidence_dir}/p100-before.sha256"
sha256sum "${build_dir}/bin/llama-server" "${build_dir}/bin/libggml-cuda.so" \
  > "${evidence_dir}/p40.sha256"
cuobjdump --list-elf "${build_dir}/bin/libggml-cuda.so" \
  > "${evidence_dir}/p40-cuda-images.txt"
grep -q '\.sm_61\.cubin' "${evidence_dir}/p40-cuda-images.txt" || {
  echo "P40 CUDA binary has no SM61 image" >&2
  exit 2
}
CUDA_VISIBLE_DEVICES=3 "${build_dir}/bin/llama-server" --version
