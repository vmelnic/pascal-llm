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

for variable_name in PASCAL_REMOTE_HOST PASCAL_REMOTE_ROOT PASCAL_IMAGE_SERVER_URL; do
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
[[ "${PASCAL_IMAGE_SERVER_URL}" =~ ^https?://[A-Za-z0-9._:-]+$ ]] || {
  echo "PASCAL_IMAGE_SERVER_URL must contain only the scheme, host and optional port" >&2
  exit 2
}

remote_host="${PASCAL_REMOTE_HOST}"
remote_root="${PASCAL_REMOTE_ROOT}"
image_server_url="${PASCAL_IMAGE_SERVER_URL%/}"
service_name="pascal-image.service"
profiles_dir="${repo_root}/config/images"

validate_profile() {
  local profile_name="$1"
  [[ "${profile_name}" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || {
    echo "invalid image profile: ${profile_name}" >&2
    exit 2
  }
  [[ -f "${profiles_dir}/${profile_name}.env" ]] || {
    echo "unknown image profile: ${profile_name}" >&2
    exit 2
  }
}

sync_remote() {
  "${script_dir}/model.sh" sync
}

install_service() {
  ssh "${remote_host}" \
    "mkdir -p \"\$HOME/.config/pascal-llm\" \"\$HOME/.config/systemd/user\"; \
     if [ ! -f \"\$HOME/.config/pascal-llm/image-server.env\" ]; then \
       cp '${remote_root}/config/image-server.env.example' \"\$HOME/.config/pascal-llm/image-server.env\"; \
       chmod 600 \"\$HOME/.config/pascal-llm/image-server.env\"; \
     fi; \
     sed 's|@PASCAL_REMOTE_ROOT@|${remote_root}|g' \
       '${remote_root}/ops/pascal-image.service' \
       > \"\$HOME/.config/systemd/user/${service_name}.tmp\"; \
     mv \"\$HOME/.config/systemd/user/${service_name}.tmp\" \
       \"\$HOME/.config/systemd/user/${service_name}\"; \
     systemctl --user daemon-reload"
}

select_profile() {
  local profile_name="$1"
  validate_profile "${profile_name}"
  ssh "${remote_host}" \
    "mkdir -p \"\$HOME/.config/pascal-llm\"; \
     printf '%s\\n' '${profile_name}' > \"\$HOME/.config/pascal-llm/active-image-model.tmp\"; \
     mv \"\$HOME/.config/pascal-llm/active-image-model.tmp\" \
       \"\$HOME/.config/pascal-llm/active-image-model\""
}

require_text_service_compatible() {
  local image_profile="$1"
  local text_profile
  local image_devices
  local text_devices

  ssh "${remote_host}" "systemctl --user is-active --quiet pascal-llm.service" || return 0
  text_profile="$(ssh "${remote_host}" "cat \"\$HOME/.config/pascal-llm/active-model\" 2>/dev/null" || true)"
  validate_profile "${image_profile}"
  [[ "${text_profile}" =~ ^[a-z0-9][a-z0-9._-]*$ && \
     -f "${repo_root}/config/models/${text_profile}.env" ]] || {
    echo "active text profile is missing or invalid; refusing concurrent GPU placement" >&2
    exit 2
  }
  image_devices="$(pascal_profile_value "${profiles_dir}/${image_profile}.env" PASCAL_IMAGE_CUDA_DEVICES)"
  text_devices="$(pascal_profile_value "${repo_root}/config/models/${text_profile}.env" PASCAL_CUDA_DEVICES)"
  if pascal_cuda_device_sets_overlap "${image_devices}" "${text_devices}"; then
    echo "image profile ${image_profile} overlaps active text profile ${text_profile} on CUDA devices ${image_devices}/${text_devices}" >&2
    exit 2
  else
    overlap_status=$?
    (( overlap_status == 1 )) || {
      echo "invalid CUDA device declaration in image or text profile" >&2
      exit 2
    }
  fi
}

require_comfyui_inactive() {
  if ssh "${remote_host}" "systemctl --user is-active --quiet pascal-comfyui.service"; then
    echo "pascal-comfyui.service already owns the image GPU; stop it before starting pascal-image" >&2
    exit 2
  fi
}

command_name="${1:-status}"
case "${command_name}" in
  install)
    sync_remote
    ssh "${remote_host}" "'${remote_root}/ops/install-stable-diffusion.sh'"
    ;;
  download)
    profile_name="${2:-}"
    validate_profile "${profile_name}"
    sync_remote
    ssh "${remote_host}" "'${remote_root}/ops/download-image-model.sh' '${profile_name}'"
    ;;
  start)
    profile_name="${2:-}"
    validate_profile "${profile_name}"
    sync_remote
    install_service
    require_comfyui_inactive
    require_text_service_compatible "${profile_name}"
    select_profile "${profile_name}"
    ssh "${remote_host}" "systemctl --user restart '${service_name}'"
    echo "Image service start requested; use '$0 status' until /v1/models responds."
    ;;
  stop)
    ssh "${remote_host}" "systemctl --user stop '${service_name}'"
    ;;
  status)
    ssh "${remote_host}" \
      "printf 'active-image-profile='; cat \"\$HOME/.config/pascal-llm/active-image-model\" 2>/dev/null || printf 'unset\\n'; \
       systemctl --user --no-pager --full status '${service_name}' || true"
    curl --silent --show-error --max-time 3 "${image_server_url}/v1/models" || true
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
    echo "usage: $0 {install|download <profile>|start <profile>|stop|status|logs|profiles}" >&2
    exit 2
    ;;
esac
