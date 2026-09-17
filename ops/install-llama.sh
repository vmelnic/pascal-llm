#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
llama_commit="${PASCAL_LLAMA_COMMIT:-5266f24da75dc449bd56cbed7addb9c8e4a6a73e}"
llama_root="${PASCAL_LLAMA_ROOT:-${repo_root}/work/llama.cpp}"
patchset_repo="https://github.com/shinbunbun/llama-cpp-p100-patches.git"
patchset_commit="52469952952ee6446207acc574c72055a0685ac4"
patchset_root="${PASCAL_LLAMA_PATCHSET_ROOT:-${repo_root}/work/llama-cpp-p100-patches}"
local_patch_root="${repo_root}/patches/llama"

for command_name in cmake dpkg-query git ninja nvcc patch; do
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

if [[ ! -d "${patchset_root}/.git" ]]; then
  mkdir -p "$(dirname "${patchset_root}")"
  git clone --filter=blob:none "${patchset_repo}" "${patchset_root}"
fi

git -C "${llama_root}" fetch --depth=1 origin "${llama_commit}"
git -C "${llama_root}" checkout --detach --force "${llama_commit}"
# The patchset creates CUDA sources that are untracked in upstream. Remove only
# untracked files from this installer-owned source clone before reapplying it;
# ignored build outputs and model data are not touched.
git -C "${llama_root}" clean -fd

git -C "${patchset_root}" fetch --depth=1 origin "${patchset_commit}"
git -C "${patchset_root}" checkout --detach --force "${patchset_commit}"
[[ "$(git -C "${patchset_root}" rev-parse HEAD)" == "${patchset_commit}" ]] || {
  echo "P100 patchset revision mismatch" >&2
  exit 2
}

mapfile -t patch_files < <(find "${patchset_root}/patches" -maxdepth 1 -type f -name '*.patch' | sort)
[[ "${#patch_files[@]}" -eq 29 ]] || {
  echo "expected 29 pinned P100 patches; found ${#patch_files[@]}" >&2
  exit 2
}

for patch_file in "${patch_files[@]}"; do
  patch_check="$({
    patch \
      --directory "${llama_root}" \
      --strip 1 \
      --fuzz=0 \
      --no-backup-if-mismatch \
      --dry-run \
      < "${patch_file}"
  } 2>&1)" || {
    printf '%s\n' "${patch_check}" >&2
    echo "P100 patch does not apply cleanly: ${patch_file}" >&2
    exit 2
  }
  if grep -Eq 'offset|fuzz' <<< "${patch_check}"; then
    printf '%s\n' "${patch_check}" >&2
    echo "P100 patch requires an offset or fuzz: ${patch_file}" >&2
    exit 2
  fi

  patch \
    --directory "${llama_root}" \
    --strip 1 \
    --fuzz=0 \
    --no-backup-if-mismatch \
    < "${patch_file}"
done

mapfile -t local_patch_files < <(find "${local_patch_root}" -maxdepth 1 -type f -name '*.patch' | sort)
for patch_file in "${local_patch_files[@]}"; do
  if ! git -C "${llama_root}" apply --check "${patch_file}"; then
    echo "local llama.cpp patch does not apply cleanly: ${patch_file}" >&2
    exit 2
  fi
  git -C "${llama_root}" apply "${patch_file}"
done

cmake \
  -S "${llama_root}" \
  -B "${llama_root}/build" \
  -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NCCL=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=60 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "${llama_root}/build" \
  --target llama-server llama-cli \
  --clean-first \
  -j "$(nproc)"

"${llama_root}/build/bin/llama-server" --version
