#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
config_file="${PASCAL_COMFYUI_CONFIG:-${HOME}/.config/pascal-llm/comfyui.env}"
[[ -r "${config_file}" ]] || {
  echo "ComfyUI config is missing: ${config_file}" >&2
  exit 2
}

set -a
# shellcheck disable=SC1090
source "${config_file}"
set +a

[[ "${PASCAL_COMFYUI_CUDA_DEVICES:-}" =~ ^[0-9]+$ ]] || {
  echo "PASCAL_COMFYUI_CUDA_DEVICES must select exactly one physical CUDA device" >&2
  exit 2
}

comfyui_commit="${PASCAL_COMFYUI_COMMIT:-828b1b9953175b6df79459f417d1032869d0b46a}"
comfyui_root="${PASCAL_COMFYUI_ROOT:-${repo_root}/work/ComfyUI}"
python_root="${PASCAL_PYTHON_ROOT:-${repo_root}/work/python}"
venv_root="${PASCAL_COMFYUI_VENV:-${repo_root}/work/venv/comfyui}"
data_root="${PASCAL_COMFYUI_DATA_ROOT:-${repo_root}/work/comfyui-data}"
uv_version="${PASCAL_UV_VERSION:-0.8.17}"
uv_sha256="920cbcaad514cc185634f6f0dcd71df5e8f4ee4456d440a22e0f8c0f142a8203"
uv_binary="${repo_root}/work/bin/uv"
python_version="3.12.11"

for command_name in curl git sha256sum tar; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "missing required command: ${command_name}" >&2
    exit 2
  }
done

if [[ ! -x "${uv_binary}" ]]; then
  temporary_dir="$(mktemp -d)"
  trap 'rm -rf "${temporary_dir}"' EXIT
  archive="${temporary_dir}/uv-x86_64-unknown-linux-gnu.tar.gz"
  curl --fail --location --proto '=https' --tlsv1.2 \
    "https://github.com/astral-sh/uv/releases/download/${uv_version}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    --output "${archive}"
  printf '%s  %s\n' "${uv_sha256}" "${archive}" | sha256sum --check --status
  tar -xzf "${archive}" -C "${temporary_dir}"
  mkdir -p "$(dirname "${uv_binary}")"
  install -m 0755 \
    "${temporary_dir}/uv-x86_64-unknown-linux-gnu/uv" \
    "${uv_binary}"
fi

if [[ ! -d "${comfyui_root}/.git" ]]; then
  mkdir -p "$(dirname "${comfyui_root}")"
  git clone --filter=blob:none https://github.com/Comfy-Org/ComfyUI.git "${comfyui_root}"
fi

git -C "${comfyui_root}" fetch --depth=1 origin "${comfyui_commit}"
git -C "${comfyui_root}" checkout --detach "${comfyui_commit}"
[[ "$(git -C "${comfyui_root}" rev-parse HEAD)" == "${comfyui_commit}" ]] || {
  echo "ComfyUI checkout does not match the pinned commit" >&2
  exit 2
}

export UV_PYTHON_INSTALL_DIR="${python_root}"
"${uv_binary}" python install "${python_version}"

if [[ ! -x "${venv_root}/bin/python" ]]; then
  mkdir -p "$(dirname "${venv_root}")"
  "${uv_binary}" venv --python "${python_version}" "${venv_root}"
fi

[[ "$("${venv_root}/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')" == "${python_version}" ]] || {
  echo "ComfyUI venv does not use Python ${python_version}" >&2
  exit 2
}

"${uv_binary}" pip install --python "${venv_root}/bin/python" \
  --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
"${uv_binary}" pip install --python "${venv_root}/bin/python" \
  --requirement "${comfyui_root}/requirements.txt"
"${uv_binary}" pip install --python "${venv_root}/bin/python" requests==2.34.2

mkdir -p "${data_root}/input" "${data_root}/output" "${data_root}/temp" "${data_root}/user"

(
cd "${comfyui_root}"
CUDA_VISIBLE_DEVICES="${PASCAL_COMFYUI_CUDA_DEVICES}" "${venv_root}/bin/python" - <<'PY'
import requests
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
architectures = torch.cuda.get_arch_list()
if "P40" not in name or capability != (6, 1):
    raise SystemExit(f"expected Tesla P40 SM61, got {name} SM{capability[0]}{capability[1]}")
if "sm_60" not in architectures:
    raise SystemExit(f"PyTorch wheel does not contain a Pascal cubin: {architectures}")

left = torch.randn((512, 512), device="cuda", dtype=torch.float32)
right = torch.randn((512, 512), device="cuda", dtype=torch.float32)
product = left @ right
convolution = torch.nn.Conv2d(4, 16, 3, padding=1, device="cuda", dtype=torch.float32)
pixels = convolution(torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float32))
torch.cuda.synchronize()
if not torch.isfinite(product).all().item() or not torch.isfinite(pixels).all().item():
    raise SystemExit("P40 CUDA execution produced non-finite validation output")
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": name,
    "capability": capability,
    "architectures": architectures,
    "cuda_execution": "passed",
    "requests": requests.__version__,
})

PY

CUDA_VISIBLE_DEVICES="${PASCAL_COMFYUI_CUDA_DEVICES}" "${venv_root}/bin/python" "${comfyui_root}/main.py" \
  --quick-test-for-ci \
  --base-directory "${data_root}" \
  --user-directory "${data_root}/user" \
  --database-url "sqlite:///${data_root}/user/comfyui.db" \
  --disable-auto-launch \
  --disable-all-custom-nodes \
  --force-fp32 \
  --highvram
)
