#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=ops/lib/cuda-device-sets.sh
source "${script_dir}/lib/cuda-device-sets.sh"
env_file="${PASCAL_ENV_FILE:-${repo_root}/.env}"
[[ -r "${env_file}" ]] || {
  echo "deployment config is missing: ${env_file}; copy .env.example to .env" >&2
  exit 2
}

set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a

for variable_name in PASCAL_REMOTE_HOST PASCAL_REMOTE_ROOT PASCAL_SERVER_URL; do
  [[ -n "${!variable_name:-}" ]] || {
    echo "required deployment variable is empty: ${variable_name}" >&2
    exit 2
  }
done

[[ "${PASCAL_REMOTE_HOST}" =~ ^[A-Za-z_][A-Za-z0-9._-]*@[A-Za-z0-9._:-]+$ ]] || {
  echo "PASCAL_REMOTE_HOST must use user@host syntax" >&2
  exit 2
}
[[ "${PASCAL_REMOTE_ROOT}" =~ ^/[A-Za-z0-9._/-]+$ && \
   "${PASCAL_REMOTE_ROOT}" != "/" ]] || {
  echo "PASCAL_REMOTE_ROOT must be a safe absolute path other than /" >&2
  exit 2
}
[[ "${PASCAL_SERVER_URL}" =~ ^https?://[A-Za-z0-9._:-]+$ ]] || {
  echo "PASCAL_SERVER_URL must contain only the scheme, host and optional port" >&2
  exit 2
}

remote_host="${PASCAL_REMOTE_HOST}"
remote_root="${PASCAL_REMOTE_ROOT}"
server_url="${PASCAL_SERVER_URL%/}"
service_name="pascal-llm.service"
profiles_dir="${repo_root}/config/models"

validate_profile() {
  local profile_name="$1"
  [[ "${profile_name}" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || {
    echo "invalid model profile: ${profile_name}" >&2
    exit 2
  }
  [[ -f "${profiles_dir}/${profile_name}.env" ]] || {
    echo "unknown model profile: ${profile_name}" >&2
    exit 2
  }
}

select_profile() {
  local profile_name="$1"
  validate_profile "${profile_name}"
  ssh "${remote_host}" \
    "mkdir -p \"\$HOME/.config/pascal-llm\"; \
     printf '%s\\n' '${profile_name}' > \"\$HOME/.config/pascal-llm/active-model.tmp\"; \
     mv \"\$HOME/.config/pascal-llm/active-model.tmp\" \"\$HOME/.config/pascal-llm/active-model\""
}

require_active_profile() {
  ssh "${remote_host}" \
    "test -s \"\$HOME/.config/pascal-llm/active-model\"" || {
    echo "no active model profile; use '$0 start <profile>'" >&2
    exit 2
  }
}

sync_remote() {
  ssh "${remote_host}" "mkdir -p '${remote_root}'"
  rsync -az \
    --exclude .git/ \
    --exclude .env \
    --exclude build/ \
    --exclude out/ \
    --exclude work/ \
    --exclude 'config/*.env' \
    "${repo_root}/" "${remote_host}:${remote_root}/"
}

install_service() {
  ssh "${remote_host}" \
    "mkdir -p \"\$HOME/.config/pascal-llm\" \"\$HOME/.config/systemd/user\"; \
     if [ ! -f \"\$HOME/.config/pascal-llm/server.env\" ]; then \
       cp '${remote_root}/config/server.env.example' \"\$HOME/.config/pascal-llm/server.env\"; \
       chmod 600 \"\$HOME/.config/pascal-llm/server.env\"; \
     fi; \
     sed 's|@PASCAL_REMOTE_ROOT@|${remote_root}|g' \
       '${remote_root}/ops/pascal-llm.service' \
       > \"\$HOME/.config/systemd/user/${service_name}.tmp\"; \
     mv \"\$HOME/.config/systemd/user/${service_name}.tmp\" \
       \"\$HOME/.config/systemd/user/${service_name}\"; \
     systemctl --user daemon-reload"
}

active_text_profile() {
  if [[ -n "${1:-}" ]]; then
    printf '%s\n' "$1"
  else
    ssh "${remote_host}" "cat \"\$HOME/.config/pascal-llm/active-model\""
  fi
}

require_image_service_compatible() {
  local text_profile="$1"
  local image_profile
  local text_devices
  local image_devices

  ssh "${remote_host}" "systemctl --user is-active --quiet pascal-image.service" || return 0
  image_profile="$(ssh "${remote_host}" "cat \"\$HOME/.config/pascal-llm/active-image-model\" 2>/dev/null" || true)"
  validate_profile "${text_profile}"
  [[ "${image_profile}" =~ ^[a-z0-9][a-z0-9._-]*$ && \
     -f "${repo_root}/config/images/${image_profile}.env" ]] || {
    echo "active image profile is missing or invalid; refusing concurrent GPU placement" >&2
    exit 2
  }
  text_devices="$(pascal_profile_value "${profiles_dir}/${text_profile}.env" PASCAL_CUDA_DEVICES)"
  image_devices="$(pascal_profile_value "${repo_root}/config/images/${image_profile}.env" PASCAL_IMAGE_CUDA_DEVICES)"
  if pascal_cuda_device_sets_overlap "${text_devices}" "${image_devices}"; then
    echo "text profile ${text_profile} overlaps active image profile ${image_profile} on CUDA devices ${text_devices}/${image_devices}" >&2
    exit 2
  else
    overlap_status=$?
    (( overlap_status == 1 )) || {
      echo "invalid CUDA device declaration in text or image profile" >&2
      exit 2
    }
  fi
}

require_comfyui_service_compatible() {
  local text_profile="$1"
  local text_devices
  local comfyui_devices

  ssh "${remote_host}" "systemctl --user is-active --quiet pascal-comfyui.service" || return 0
  text_devices="$(pascal_profile_value "${profiles_dir}/${text_profile}.env" PASCAL_CUDA_DEVICES)"
  comfyui_devices="$(ssh "${remote_host}" \
    "source \"\$HOME/.config/pascal-llm/comfyui.env\" 2>/dev/null && \
     printf '%s' \"\${PASCAL_COMFYUI_CUDA_DEVICES:-}\"")"
  [[ -n "${comfyui_devices}" ]] || {
    echo "active ComfyUI service has no valid CUDA-device declaration" >&2
    exit 2
  }
  if pascal_cuda_device_sets_overlap "${text_devices}" "${comfyui_devices}"; then
    echo "text profile ${text_profile} overlaps active ComfyUI on CUDA devices ${text_devices}/${comfyui_devices}" >&2
    exit 2
  else
    overlap_status=$?
    (( overlap_status == 1 )) || {
      echo "invalid CUDA device declaration in text or ComfyUI config" >&2
      exit 2
    }
  fi
}

command_name="${1:-status}"
case "${command_name}" in
  sync)
    sync_remote
    ;;
  start)
    sync_remote
    install_service
    if [[ -z "${2:-}" ]]; then
      require_active_profile
    fi
    target_profile="$(active_text_profile "${2:-}")"
    validate_profile "${target_profile}"
    require_image_service_compatible "${target_profile}"
    require_comfyui_service_compatible "${target_profile}"
    if [[ -n "${2:-}" ]]; then
      select_profile "$2"
      ssh "${remote_host}" "systemctl --user restart '${service_name}'"
    else
      require_active_profile
      ssh "${remote_host}" "systemctl --user start '${service_name}'"
    fi
    echo "Service start requested; use '$0 status' until it reports ready."
    ;;
  download)
    profile_name="${2:-}"
    validate_profile "${profile_name}"
    sync_remote
    ssh "${remote_host}" "'${remote_root}/ops/download-model.sh' '${profile_name}'"
    ;;
  stop)
    ssh "${remote_host}" "systemctl --user stop '${service_name}'"
    ;;
  restart)
    sync_remote
    install_service
    if [[ -z "${2:-}" ]]; then
      require_active_profile
    fi
    target_profile="$(active_text_profile "${2:-}")"
    validate_profile "${target_profile}"
    require_image_service_compatible "${target_profile}"
    require_comfyui_service_compatible "${target_profile}"
    if [[ -n "${2:-}" ]]; then
      select_profile "$2"
    else
      require_active_profile
    fi
    ssh "${remote_host}" "systemctl --user restart '${service_name}'"
    echo "Service restart requested; use '$0 status' until it reports ready."
    ;;
  status)
    ssh "${remote_host}" \
      "printf 'active-profile='; cat \"\$HOME/.config/pascal-llm/active-model\" 2>/dev/null || printf 'unset\\n'; \
       systemctl --user --no-pager --full status '${service_name}' || true"
    curl --silent --show-error --max-time 3 \
      -H 'Authorization: Bearer local' \
      "${server_url}/health" || true
    echo
    ;;
  logs)
    ssh "${remote_host}" "journalctl --user -u '${service_name}' -n 200 --no-pager"
    ;;
  profiles)
    find "${profiles_dir}" -maxdepth 1 -type f -name '*.env' \
      -exec basename {} .env \; | sort
    ;;
  *)
    echo "usage: $0 {sync|download <profile>|start [profile]|stop|restart [profile]|status|logs|profiles}" >&2
    exit 2
    ;;
esac
