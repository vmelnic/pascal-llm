#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
strata_repo="https://github.com/Niko1221/Strata.git"
strata_commit="${PASCAL_STRATA_COMMIT:-9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4}"
strata_root="${PASCAL_STRATA_ROOT:-${repo_root}/work/strata}"
build_root="${PASCAL_STRATA_BUILD_ROOT:-${strata_root}/build-sm60-gcc13}"
nvcc="${PASCAL_STRATA_NVCC:-$(command -v nvcc || true)}"
cc="${PASCAL_STRATA_CC:-$(command -v gcc-13 || true)}"
cxx="${PASCAL_STRATA_CXX:-$(command -v g++-13 || true)}"
build_jobs="${PASCAL_STRATA_BUILD_JOBS:-$(( ($(nproc) + 1) / 2 ))}"

for command_name in cmake git ninja nproc; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "missing required command: ${command_name}" >&2
    exit 2
  }
done

[[ -n "${nvcc}" && -x "${nvcc}" ]] || {
  echo "CUDA compiler is missing; set PASCAL_STRATA_NVCC to a CUDA 12 nvcc" >&2
  exit 2
}
[[ -n "${cc}" && -x "${cc}" && -n "${cxx}" && -x "${cxx}" ]] || {
  echo "GCC 13 is required to match the CUDA 12.4 host toolchain" >&2
  exit 2
}
"${nvcc}" --version | grep -Eq 'release 12\.' || {
  echo "Strata's Pascal build requires CUDA 12.x: ${nvcc}" >&2
  exit 2
}
[[ "${build_jobs}" =~ ^[1-9][0-9]*$ ]] || {
  echo "PASCAL_STRATA_BUILD_JOBS must be a positive integer" >&2
  exit 2
}

if [[ ! -d "${strata_root}/.git" ]]; then
  mkdir -p "$(dirname "${strata_root}")"
  git clone --filter=blob:none "${strata_repo}" "${strata_root}"
fi

git -C "${strata_root}" fetch --depth=1 origin "${strata_commit}"
git -C "${strata_root}" checkout --detach --force "${strata_commit}"
git -C "${strata_root}" clean -fd
[[ "$(git -C "${strata_root}" rev-parse HEAD)" == "${strata_commit}" ]] || {
  echo "Strata revision mismatch" >&2
  exit 2
}

cmake \
  -S "${strata_root}" \
  -B "${build_root}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${cc}" \
  -DCMAKE_CXX_COMPILER="${cxx}" \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES=60 \
  -DCMAKE_CUDA_COMPILER="${nvcc}"

cmake --build "${build_root}" \
  --target strata \
  -j "${build_jobs}"

[[ -x "${build_root}/strata" ]] || {
  echo "Strata binary is missing after the build: ${build_root}/strata" >&2
  exit 2
}

printf 'strata_commit=%s\n' "$(git -C "${strata_root}" rev-parse HEAD)"
printf 'strata_binary=%s\n' "${build_root}/strata"
"${build_root}/strata" --version
