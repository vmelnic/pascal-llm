#!/usr/bin/env bash
set -euo pipefail

# Strata v0.2.20 in a separate source tree. The pinned 0.1.46.3 deployment in
# work/strata (ops/install-strata.sh) stays untouched until the new release is
# qualified for Swift as well. v0.2.20 adds the `strata-inkling` engine for
# Inkling 2 Mini and moved the serving front end to terminal/serve/.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
strata_repo="https://github.com/Niko1221/Strata.git"
# Release tag v0.2.20 (annotated tag object 4920a3a4) points at this commit.
strata_commit="${PASCAL_STRATA_NEXT_COMMIT:-a34979fc94f3d126b9bb8210f4ad311345688b52}"
strata_root="${PASCAL_STRATA_NEXT_ROOT:-${repo_root}/work/strata-next}"
build_root="${strata_root}/build-sm60"
nvcc="${PASCAL_STRATA_NVCC:-$(command -v nvcc || true)}"
cc="$(command -v gcc-13 || true)"
cxx="$(command -v g++-13 || true)"
build_jobs="${PASCAL_BUILD_JOBS:-$(nproc)}"
targets=(${PASCAL_STRATA_NEXT_TARGETS:-strata-inkling strata})

for command_name in cmake git ninja nproc cuobjdump sha256sum python3; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "missing required command: ${command_name}" >&2
    exit 2
  }
done
[[ -n "${nvcc}" && -x "${nvcc}" ]] || { echo "CUDA compiler is missing" >&2; exit 2; }
[[ -n "${cc}" && -n "${cxx}" ]] || { echo "GCC 13 is required by CUDA 12.4" >&2; exit 2; }
"${nvcc}" --version | grep -Eq 'release 12\.' || {
  echo "Pascal builds require CUDA 12.x: ${nvcc}" >&2
  exit 2
}
[[ "${build_jobs}" =~ ^[1-9][0-9]*$ ]] || { echo "PASCAL_BUILD_JOBS must be a positive integer" >&2; exit 2; }

if [[ ! -d "${strata_root}/.git" ]]; then
  mkdir -p "$(dirname "${strata_root}")"
  git clone --filter=blob:none "${strata_repo}" "${strata_root}"
fi
git -C "${strata_root}" fetch --depth=1 origin "${strata_commit}"
git -C "${strata_root}" checkout --detach --force "${strata_commit}"
# Keep the build tree (it holds the service binaries), the front-end venv and the
# repository-written run configurations and logs; drop everything else untracked.
git -C "${strata_root}" clean -fd -e build-sm60/ -e terminal/.venv/ -e 'strata-*.json' -e '*.log'
[[ "$(git -C "${strata_root}" rev-parse HEAD)" == "${strata_commit}" ]] || {
  echo "Strata revision mismatch" >&2
  exit 2
}

# pascal-llm patches (tensor-parallel Inkling decode), applied in order.
mapfile -t local_patches < <(find "${repo_root}/patches/strata-next" -maxdepth 1 -type f -name '*.patch' 2>/dev/null | sort)
for patch_file in "${local_patches[@]}"; do
  git -C "${strata_root}" apply --check "${patch_file}" || {
    echo "Strata patch does not apply cleanly: ${patch_file}" >&2
    exit 2
  }
  git -C "${strata_root}" apply "${patch_file}"
done

# CUDA 12.4's EDG front end does not know GCC 13's AMX builtins. Host
# translation units that include <immintrin.h> through a .cu file fail inside
# amx*intrin.h although no CUDA source uses AMX. Pre-defining only those header
# guards for CUDA compilation skips the AMX declarations; any real AMX use
# would still fail to compile instead of silently changing code generation.
amx_guard_flags="-D_AMXTILEINTRIN_H_INCLUDED -D_AMXINT8INTRIN_H_INCLUDED -D_AMXBF16INTRIN_H_INCLUDED -D_AMXFP16INTRIN_H_INCLUDED -D_AMXCOMPLEXINTRIN_H_INCLUDED"

cmake -S "${strata_root}/engine" -B "${build_root}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${cc}" \
  -DCMAKE_CXX_COMPILER="${cxx}" \
  -DCMAKE_CUDA_HOST_COMPILER="${cxx}" \
  -DCMAKE_CUDA_COMPILER="${nvcc}" \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES=60 \
  -DSTRATA_INK_CUDA_ARCHS=60 \
  -DCMAKE_CUDA_FLAGS="${amx_guard_flags}"

cmake --build "${build_root}" --target "${targets[@]}" -j "${build_jobs}"

for target in "${targets[@]}"; do
  binary="$(find "${build_root}" -maxdepth 3 -type f -name "${target}" -perm -u+x | head -n 1)"
  [[ -n "${binary}" ]] || { echo "built binary is missing: ${target}" >&2; exit 2; }
  images="$(cuobjdump --list-elf "${binary}" 2>/dev/null || true)"
  grep -q 'sm_60' <<< "${images}" || { echo "${target} has no SM60 CUDA image" >&2; exit 2; }
  sha256sum "${binary}"
  "${binary}" --version || true
done

# The serving front end uses its own virtual environment inside terminal/.
# The host has no python3-venv/ensurepip, so the repository's pinned uv
# (work/bin/uv) creates it when present.
venv="${strata_root}/terminal/.venv"
uv_bin="${repo_root}/work/bin/uv"
if [[ -x "${uv_bin}" ]]; then
  [[ -x "${venv}/bin/python" ]] || "${uv_bin}" venv --quiet --python 3.12 "${venv}"
  "${uv_bin}" pip install --quiet --python "${venv}/bin/python" -r "${strata_root}/terminal/requirements.txt"
else
  [[ -x "${venv}/bin/python" ]] || python3 -m venv "${venv}"
  "${venv}/bin/python" -m pip install --quiet -r "${strata_root}/terminal/requirements.txt"
fi
printf 'strata_commit=%s\n' "$(git -C "${strata_root}" rev-parse HEAD)"
