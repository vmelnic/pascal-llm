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

comfyui_root="${PASCAL_COMFYUI_ROOT:-${repo_root}/work/ComfyUI}"
venv_root="${PASCAL_COMFYUI_VENV:-${repo_root}/work/venv/comfyui}"
data_root="${PASCAL_COMFYUI_DATA_ROOT:-${repo_root}/work/comfyui-data}"
PASCAL_MODEL_ROOT="${PASCAL_MODEL_ROOT:-${repo_root}/work/models}"
profile_dir="${PASCAL_IMAGE_PROFILE_DIR:-${repo_root}/config/images}"

required_variables=(
  PASCAL_COMFYUI_CUDA_DEVICES
  PASCAL_COMFYUI_HOST
  PASCAL_COMFYUI_PORT
  PASCAL_COMFYUI_FORCE_FP32
  PASCAL_COMFYUI_VRAM_MODE
  PASCAL_COMFYUI_RESERVE_VRAM_GIB
  PASCAL_COMFYUI_DISABLE_CUSTOM_NODES
)
for variable_name in "${required_variables[@]}"; do
  [[ -n "${!variable_name:-}" ]] || {
    echo "required ComfyUI variable is empty: ${variable_name}" >&2
    exit 2
  }
done

[[ "${PASCAL_COMFYUI_CUDA_DEVICES}" =~ ^[0-9]+$ ]] || {
  echo "PASCAL_COMFYUI_CUDA_DEVICES must select exactly one physical CUDA device" >&2
  exit 2
}
[[ "${PASCAL_COMFYUI_PORT}" =~ ^[1-9][0-9]*$ ]] && (( PASCAL_COMFYUI_PORT <= 65535 )) || {
  echo "PASCAL_COMFYUI_PORT must be a valid TCP port" >&2
  exit 2
}
[[ "${PASCAL_COMFYUI_RESERVE_VRAM_GIB}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "PASCAL_COMFYUI_RESERVE_VRAM_GIB must be a non-negative decimal number" >&2
  exit 2
}
case "${PASCAL_COMFYUI_FORCE_FP32}" in 0|1) ;; *)
  echo "PASCAL_COMFYUI_FORCE_FP32 must be 0 or 1" >&2
  exit 2
esac
case "${PASCAL_COMFYUI_DISABLE_CUSTOM_NODES}" in 0|1) ;; *)
  echo "PASCAL_COMFYUI_DISABLE_CUSTOM_NODES must be 0 or 1" >&2
  exit 2
esac
case "${PASCAL_COMFYUI_VRAM_MODE}" in normal|high) ;; *)
  echo "PASCAL_COMFYUI_VRAM_MODE must be normal or high" >&2
  exit 2
esac

[[ -f "${comfyui_root}/main.py" && -x "${venv_root}/bin/python" ]] || {
  echo "ComfyUI is not installed; run ops/comfyui.sh install" >&2
  exit 2
}

checkpoint_dir="${data_root}/models/checkpoints/pascal-llm"
mkdir -p \
  "${checkpoint_dir}" \
  "${data_root}/input" \
  "${data_root}/output" \
  "${data_root}/temp" \
  "${data_root}/user"

linked_models=0
for profile_file in "${profile_dir}"/*.env; do
  [[ -f "${profile_file}" ]] || continue
  unset PASCAL_IMAGE_MODEL_PATH PASCAL_IMAGE_FILE_SIZE PASCAL_IMAGE_FILE_SHA256
  # shellcheck disable=SC1090
  source "${profile_file}"
  [[ -n "${PASCAL_IMAGE_MODEL_PATH:-}" ]] || {
    echo "image profile has no model path: ${profile_file}" >&2
    exit 2
  }
  [[ -f "${PASCAL_IMAGE_MODEL_PATH}" ]] || continue
  [[ -n "${PASCAL_IMAGE_FILE_SIZE:-}" && -n "${PASCAL_IMAGE_FILE_SHA256:-}" ]] || {
    echo "image profile has no artifact integrity declaration: ${profile_file}" >&2
    exit 2
  }
  completion_file="$(dirname "${PASCAL_IMAGE_MODEL_PATH}")/.complete"
  [[ -f "${completion_file}" ]] || {
    echo "image artifact has no completion marker: ${PASCAL_IMAGE_MODEL_PATH}" >&2
    exit 2
  }
  [[ "$(stat -c %s "${PASCAL_IMAGE_MODEL_PATH}")" == "${PASCAL_IMAGE_FILE_SIZE}" ]] || {
    echo "image artifact size does not match profile: ${PASCAL_IMAGE_MODEL_PATH}" >&2
    exit 2
  }
  grep -Fqx "sha256=${PASCAL_IMAGE_FILE_SHA256}" "${completion_file}" || {
    echo "image artifact completion marker does not match profile: ${PASCAL_IMAGE_MODEL_PATH}" >&2
    exit 2
  }
  profile_name="$(basename "${profile_file}" .env)"
  ln -sfn "${PASCAL_IMAGE_MODEL_PATH}" "${checkpoint_dir}/${profile_name}.safetensors"
  ((linked_models += 1))
done
(( linked_models > 0 )) || {
  echo "no complete image checkpoints were found under ${PASCAL_MODEL_ROOT}" >&2
  exit 2
}

declare -a arguments=(
  --listen "${PASCAL_COMFYUI_HOST}"
  --port "${PASCAL_COMFYUI_PORT}"
  --base-directory "${data_root}"
  --user-directory "${data_root}/user"
  --database-url "sqlite:///${data_root}/user/comfyui.db"
  --disable-auto-launch
  --reserve-vram "${PASCAL_COMFYUI_RESERVE_VRAM_GIB}"
)
[[ "${PASCAL_COMFYUI_FORCE_FP32}" == "1" ]] && arguments+=(--force-fp32)
[[ "${PASCAL_COMFYUI_VRAM_MODE}" == "high" ]] && arguments+=(--highvram)
[[ "${PASCAL_COMFYUI_VRAM_MODE}" == "normal" ]] && arguments+=(--normalvram)
[[ "${PASCAL_COMFYUI_DISABLE_CUSTOM_NODES}" == "1" ]] && arguments+=(--disable-all-custom-nodes)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PASCAL_COMFYUI_CUDA_DEVICES}"
exec "${venv_root}/bin/python" "${comfyui_root}/main.py" "${arguments[@]}"
