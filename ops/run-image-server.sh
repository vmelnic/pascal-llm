#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
config_file="${PASCAL_IMAGE_SERVER_CONFIG:-${HOME}/.config/pascal-llm/image-server.env}"
[[ -r "${config_file}" ]] || {
  echo "image-server config is missing: ${config_file}" >&2
  exit 2
}

set -a
# shellcheck disable=SC1090
source "${config_file}"
set +a

PASCAL_MODEL_ROOT="${PASCAL_MODEL_ROOT:-${repo_root}/work/models}"
PASCAL_SDCPP_ROOT="${PASCAL_SDCPP_ROOT:-${repo_root}/work/stable-diffusion.cpp}"
active_profile_file="${PASCAL_ACTIVE_IMAGE_FILE:-${HOME}/.config/pascal-llm/active-image-model}"
image_profile="${PASCAL_IMAGE_PROFILE:-}"
if [[ -z "${image_profile}" && -r "${active_profile_file}" ]]; then
  IFS= read -r image_profile < "${active_profile_file}"
fi
[[ "${image_profile}" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || {
  echo "active image profile is missing or invalid: '${image_profile}'" >&2
  exit 2
}

profile_dir="${PASCAL_IMAGE_PROFILE_DIR:-${repo_root}/config/images}"
profile_file="${profile_dir}/${image_profile}.env"
[[ -r "${profile_file}" ]] || {
  echo "image profile is missing: ${profile_file}" >&2
  exit 2
}

set -a
# shellcheck disable=SC1090
source "${profile_file}"
set +a

required_variables=(
  PASCAL_IMAGE_MODEL_PATH
  PASCAL_IMAGE_MODEL_ID
  PASCAL_IMAGE_FILE_SIZE
  PASCAL_IMAGE_FILE_SHA256
  PASCAL_IMAGE_CUDA_DEVICES
  PASCAL_IMAGE_BACKEND
  PASCAL_IMAGE_WEIGHT_TYPE
  PASCAL_IMAGE_WIDTH
  PASCAL_IMAGE_HEIGHT
  PASCAL_IMAGE_STEPS
  PASCAL_IMAGE_CFG_SCALE
  PASCAL_IMAGE_SAMPLING_METHOD
  PASCAL_IMAGE_SCHEDULER
  PASCAL_IMAGE_FLASH_ATTENTION
  PASCAL_IMAGE_VAE_TILING
  PASCAL_IMAGE_HOST
  PASCAL_IMAGE_PORT
  PASCAL_IMAGE_THREADS
)
for variable_name in "${required_variables[@]}"; do
  [[ -n "${!variable_name:-}" ]] || {
    echo "required image-server variable is empty: ${variable_name}" >&2
    exit 2
  }
done

PASCAL_IMAGE_NEGATIVE_PROMPT="${PASCAL_IMAGE_NEGATIVE_PROMPT:-}"

server_binary="${PASCAL_SDCPP_ROOT}/build/bin/sd-server"
completion_file="$(dirname "${PASCAL_IMAGE_MODEL_PATH}")/.complete"
[[ -x "${server_binary}" ]] || {
  echo "sd-server is missing: ${server_binary}" >&2
  exit 2
}
[[ -f "${PASCAL_IMAGE_MODEL_PATH}" && -f "${completion_file}" ]] || {
  echo "image artifact is incomplete: ${PASCAL_IMAGE_MODEL_PATH}" >&2
  exit 2
}
[[ "$(stat -c %s "${PASCAL_IMAGE_MODEL_PATH}")" == "${PASCAL_IMAGE_FILE_SIZE}" ]] || {
  echo "image artifact size does not match its profile" >&2
  exit 2
}
grep -Fqx "sha256=${PASCAL_IMAGE_FILE_SHA256}" "${completion_file}" || {
  echo "image artifact completion marker does not match its profile" >&2
  exit 2
}

case "${PASCAL_IMAGE_FLASH_ATTENTION}" in
  0|1) ;;
  *)
    echo "PASCAL_IMAGE_FLASH_ATTENTION must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${PASCAL_IMAGE_VAE_TILING}" in
  0|1) ;;
  *)
    echo "PASCAL_IMAGE_VAE_TILING must be 0 or 1" >&2
    exit 2
    ;;
esac

[[ "${PASCAL_IMAGE_CUDA_DEVICES}" =~ ^[0-9]+$ ]] || {
  echo "PASCAL_IMAGE_CUDA_DEVICES must select exactly one CUDA device" >&2
  exit 2
}
[[ "${PASCAL_IMAGE_BACKEND}" =~ ^cuda[0-9]+$ ]] || {
  echo "PASCAL_IMAGE_BACKEND must select one CUDA backend, for example cuda0" >&2
  exit 2
}
case "${PASCAL_IMAGE_WEIGHT_TYPE}" in
  f32|f16) ;;
  *)
    echo "PASCAL_IMAGE_WEIGHT_TYPE must be f32 or f16" >&2
    exit 2
    ;;
esac
for numeric_variable in PASCAL_IMAGE_WIDTH PASCAL_IMAGE_HEIGHT PASCAL_IMAGE_STEPS \
  PASCAL_IMAGE_PORT PASCAL_IMAGE_THREADS; do
  [[ "${!numeric_variable}" =~ ^[1-9][0-9]*$ ]] || {
    echo "${numeric_variable} must be a positive integer" >&2
    exit 2
  }
done
[[ "${PASCAL_IMAGE_CFG_SCALE}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "PASCAL_IMAGE_CFG_SCALE must be a non-negative decimal number" >&2
  exit 2
}
(( PASCAL_IMAGE_PORT <= 65535 )) || {
  echo "PASCAL_IMAGE_PORT must be at most 65535" >&2
  exit 2
}

declare -a optional_arguments=()
optional_arguments+=(--type "${PASCAL_IMAGE_WEIGHT_TYPE}")
if [[ "${PASCAL_IMAGE_FLASH_ATTENTION}" == "1" ]]; then
  optional_arguments+=(--diffusion-fa)
fi
if [[ "${PASCAL_IMAGE_VAE_TILING}" == "1" ]]; then
  optional_arguments+=(--vae-tiling)
fi

export CUDA_VISIBLE_DEVICES="${PASCAL_IMAGE_CUDA_DEVICES}"

exec "${server_binary}" \
  --model "${PASCAL_IMAGE_MODEL_PATH}" \
  --backend "${PASCAL_IMAGE_BACKEND}" \
  --auto-fit off \
  --listen-ip "${PASCAL_IMAGE_HOST}" \
  --listen-port "${PASCAL_IMAGE_PORT}" \
  --threads "${PASCAL_IMAGE_THREADS}" \
  --width "${PASCAL_IMAGE_WIDTH}" \
  --height "${PASCAL_IMAGE_HEIGHT}" \
  --steps "${PASCAL_IMAGE_STEPS}" \
  --cfg-scale "${PASCAL_IMAGE_CFG_SCALE}" \
  --negative-prompt "${PASCAL_IMAGE_NEGATIVE_PROMPT}" \
  --sampling-method "${PASCAL_IMAGE_SAMPLING_METHOD}" \
  --scheduler "${PASCAL_IMAGE_SCHEDULER}" \
  --rng cuda \
  "${optional_arguments[@]}"
