#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
sdcpp_commit="${PASCAL_SDCPP_COMMIT:-42d6c0ab92fe6595776b28e3f7c8925db79b31f5}"
sdcpp_root="${PASCAL_SDCPP_ROOT:-${repo_root}/work/stable-diffusion.cpp}"

for command_name in cmake git ninja nvcc; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "missing required command: ${command_name}" >&2
    exit 2
  }
done

if [[ ! -d "${sdcpp_root}/.git" ]]; then
  mkdir -p "$(dirname "${sdcpp_root}")"
  git clone --filter=blob:none --recursive \
    https://github.com/leejet/stable-diffusion.cpp.git "${sdcpp_root}"
fi

git -C "${sdcpp_root}" fetch --depth=1 origin "${sdcpp_commit}"
git -C "${sdcpp_root}" checkout --detach "${sdcpp_commit}"
git -C "${sdcpp_root}" submodule update --init --recursive --depth=1

cmake \
  -S "${sdcpp_root}" \
  -B "${sdcpp_root}/build" \
  -G Ninja \
  -DSD_CUDA=ON \
  -DSD_BUILD_EXAMPLES=ON \
  -DCMAKE_CUDA_ARCHITECTURES='60;61' \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "${sdcpp_root}/build" \
  --target sd-server sd-cli \
  --clean-first \
  -j "$(nproc)"

"${sdcpp_root}/build/bin/sd-server" --version
