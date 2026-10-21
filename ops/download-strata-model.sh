#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
source_dir="${PASCAL_STRATA_SOURCE_DIR:-${repo_root}/work/strata}"
data_dir="${PASCAL_STRATA_DATA_DIR:-${repo_root}/work/strata-data}"
config_path="${source_dir}/strata-swift-iq3_s.json"
expected_commit="9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4"

[[ -d "${source_dir}/.git" ]] || {
  echo "Strata source is missing: ${source_dir}; run ops/install-strata.sh first" >&2
  exit 2
}
actual_commit="$(git -C "${source_dir}" rev-parse HEAD)"
[[ "${actual_commit}" == "${expected_commit}" ]] || {
  echo "unexpected Strata revision: ${actual_commit}" >&2
  exit 2
}

for tool in python3 cmake ninja /usr/bin/nvcc /usr/bin/gcc-13 /usr/bin/g++-13; do
  if [[ "${tool}" == /* ]]; then
    [[ -x "${tool}" ]] || { echo "required installed tool is missing: ${tool}" >&2; exit 2; }
  else
    command -v "${tool}" >/dev/null 2>&1 || { echo "required installed tool is missing: ${tool}" >&2; exit 2; }
  fi
done
uv_bin="${repo_root}/work/bin/uv"
if [[ -x "${uv_bin}" ]]; then
  export PATH="$(dirname "${uv_bin}"):${PATH}"
  export STRATA_UV=1
else
  python3 -c 'import ensurepip, venv; assert __import__("sys").version_info >= (3, 10)'
fi

mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits)
for index in 0 1 2; do
  [[ "${gpu_rows[${index}]:-}" == "${index}, Tesla P100-PCIE-16GB" ]] || {
    echo "CUDA index ${index} is not the expected Tesla P100" >&2
    exit 2
  }
done
[[ "${gpu_rows[3]:-}" == "3, Tesla P40" ]] || {
  echo "CUDA index 3 is not the expected excluded Tesla P40" >&2
  exit 2
}

mkdir -p "${data_dir}"
cd "${source_dir}"
if [[ ! -x .venv/bin/python && ! -x "${uv_bin}" ]]; then
  python3 -m venv .venv
fi

export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDAHOSTCXX=/usr/bin/g++-13
export STRATA_NVCC=/usr/bin/nvcc
export STRATA_EXPERIMENTAL_SM60=1

# Upstream setup requires a packed-KV selector when generating a long-context
# config. The repository normalizer below replaces it with the required FP16
# runtime KV and retains only the upstream-generated streaming support.
./setup.sh \
  --setup \
  --family swift \
  --model IQ3_S \
  --context 262144 \
  --kv int8 \
  --kv-streaming on \
  --vision no \
  --gpus 0,1,2 \
  --cuda 12 \
  --build \
  --low-ram resident \
  --parallel 2 \
  --host 0.0.0.0 \
  --api-key local \
  --port 8080 \
  --no-browser \
  --data-dir "${data_dir}" \
  --set-default-data-dir \
  --no-start \
  --yes

"${script_dir}/configure-strata.py" --config "${config_path}"

models_dir="${data_dir}/models/swift-IQ3_S"
shard_one="${models_dir}/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
shard_two="${models_dir}/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"
printf '%s  %s\n' \
  '46cb3996cd1a17fe8bc5ae1f9bb16869480de5a03f17f868d6047a20a46edd3b' "${shard_one}" \
  '70ce6fb48a94fe34bd7e9a53a526be68066c64b5a5077cdc82fb31350b66b608' "${shard_two}" |
  sha256sum --check --strict

"${script_dir}/configure-strata.py" --config "${config_path}" --check
echo "Strata Swift 1.5 IQ3_S is downloaded and prepared at ${data_dir}."
