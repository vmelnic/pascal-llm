#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
llama_commit="${PASCAL_LLAMA_COMMIT:-4a89937354190cef5a97baf8eeb17336105eb72d}"
llama_root="${PASCAL_LLAMA_ROOT:-${repo_root}/work/llama.cpp}"

for command_name in cmake dpkg-query git ninja nvcc; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "missing required command: ${command_name}" >&2
    exit 2
  }
done

required_nccl_version="2.27.7-1+cuda12.9"
installed_nccl_version="$(dpkg-query -W -f='${Version}' libnccl-dev 2>/dev/null || true)"
[[ "${installed_nccl_version}" == "${required_nccl_version}" ]] || {
  echo "libnccl-dev ${required_nccl_version} is required; found '${installed_nccl_version:-missing}'" >&2
  exit 2
}

if [[ ! -d "${llama_root}/.git" ]]; then
  mkdir -p "$(dirname "${llama_root}")"
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "${llama_root}"
fi

git -C "${llama_root}" fetch --depth=1 origin "${llama_commit}"
git -C "${llama_root}" checkout --detach "${llama_commit}"

cmake \
  -S "${llama_root}" \
  -B "${llama_root}/build" \
  -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NCCL=ON \
  -DCMAKE_CUDA_ARCHITECTURES=60 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "${llama_root}/build" \
  --target llama-server llama-cli \
  --clean-first \
  -j "$(nproc)"

"${llama_root}/build/bin/llama-server" --version
