#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
env_file="${PASCAL_ENV_FILE:-${repo_root}/.env}"
[[ -r "${env_file}" ]] || {
  echo "deployment config is missing: ${env_file}; copy .env.example to .env" >&2
  exit 2
}

set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a

for variable_name in PASCAL_REMOTE_HOST PASCAL_REMOTE_ROOT PASCAL_COMFYUI_URL; do
  [[ -n "${!variable_name:-}" ]] || {
    echo "required deployment variable is empty: ${variable_name}" >&2
    exit 2
  }
done

[[ "${PASCAL_REMOTE_HOST}" =~ ^[A-Za-z_][A-Za-z0-9._-]*@[A-Za-z0-9._:-]+$ ]] || {
  echo "PASCAL_REMOTE_HOST must use user@host syntax" >&2
  exit 2
}
[[ "${PASCAL_REMOTE_ROOT}" =~ ^/[A-Za-z0-9._/-]+$ && "${PASCAL_REMOTE_ROOT}" != "/" ]] || {
  echo "PASCAL_REMOTE_ROOT must be a safe absolute path other than /" >&2
  exit 2
}
[[ "${PASCAL_COMFYUI_URL}" =~ ^https?://[A-Za-z0-9._:-]+$ ]] || {
  echo "PASCAL_COMFYUI_URL must contain only the scheme, host and optional port" >&2
  exit 2
}

remote_host="${PASCAL_REMOTE_HOST}"
remote_root="${PASCAL_REMOTE_ROOT}"
comfyui_url="${PASCAL_COMFYUI_URL%/}"
service_name="pascal-comfyui.service"

sync_remote() {
  "${script_dir}/model.sh" sync
}

install_service() {
  ssh "${remote_host}" \
    "mkdir -p \"\$HOME/.config/pascal-llm\" \"\$HOME/.config/systemd/user\"; \
     if [ ! -f \"\$HOME/.config/pascal-llm/comfyui.env\" ]; then \
       cp '${remote_root}/config/comfyui.env.example' \"\$HOME/.config/pascal-llm/comfyui.env\"; \
       chmod 600 \"\$HOME/.config/pascal-llm/comfyui.env\"; \
     fi; \
     sed 's|@PASCAL_REMOTE_ROOT@|${remote_root}|g' \
       '${remote_root}/ops/pascal-comfyui.service' \
       > \"\$HOME/.config/systemd/user/${service_name}.tmp\"; \
     mv \"\$HOME/.config/systemd/user/${service_name}.tmp\" \
       \"\$HOME/.config/systemd/user/${service_name}\"; \
     systemctl --user daemon-reload"
}

require_image_service_inactive() {
  if ssh "${remote_host}" "systemctl --user is-active --quiet pascal-image.service"; then
    echo "pascal-image.service already owns the P40; stop it before starting ComfyUI" >&2
    exit 2
  fi
}

require_comfyui_inactive() {
  if ssh "${remote_host}" "systemctl --user is-active --quiet '${service_name}'"; then
    echo "stop ${service_name} before changing its native Python environment" >&2
    exit 2
  fi
}

wait_until_ready() {
  local port
  local attempt
  port="$(ssh "${remote_host}" \
    "source \"\$HOME/.config/pascal-llm/comfyui.env\" && printf '%s' \"\${PASCAL_COMFYUI_PORT}\"")"
  [[ "${port}" =~ ^[1-9][0-9]*$ ]] || {
    echo "remote ComfyUI port is invalid" >&2
    return 1
  }

  for ((attempt = 0; attempt < 45; attempt += 1)); do
    if ssh "${remote_host}" \
      "curl --fail --silent --max-time 2 'http://127.0.0.1:${port}/system_stats' >/dev/null"; then
      echo "ComfyUI is ready at ${comfyui_url}."
      return 0
    fi
    if ! ssh "${remote_host}" "systemctl --user is-active --quiet '${service_name}'"; then
      ssh "${remote_host}" "journalctl --user -u '${service_name}' -n 80 --no-pager" >&2
      return 1
    fi
    sleep 1
  done

  echo "ComfyUI did not become ready; stopping the unhealthy service" >&2
  ssh "${remote_host}" "systemctl --user stop '${service_name}'; journalctl --user -u '${service_name}' -n 80 --no-pager" >&2
  return 1
}

command_name="${1:-status}"
case "${command_name}" in
  install)
    sync_remote
    install_service
    require_image_service_inactive
    require_comfyui_inactive
    ssh "${remote_host}" "'${remote_root}/ops/install-comfyui.sh'"
    ;;
  start)
    sync_remote
    install_service
    require_image_service_inactive
    ssh "${remote_host}" "systemctl --user restart '${service_name}'"
    wait_until_ready
    ;;
  stop)
    ssh "${remote_host}" "systemctl --user stop '${service_name}'"
    ;;
  restart)
    sync_remote
    install_service
    require_image_service_inactive
    ssh "${remote_host}" "systemctl --user restart '${service_name}'"
    wait_until_ready
    ;;
  status)
    ssh "${remote_host}" "systemctl --user --no-pager --full status '${service_name}' || true"
    curl --silent --show-error --max-time 3 "${comfyui_url}/system_stats" || true
    echo
    ;;
  logs)
    ssh "${remote_host}" "journalctl --user -u '${service_name}' -n 200 --no-pager"
    ;;
  models)
    curl --fail --silent --show-error --max-time 10 \
      "${comfyui_url}/object_info/CheckpointLoaderSimple"
    echo
    ;;
  *)
    echo "usage: $0 {install|start|stop|restart|status|logs|models}" >&2
    exit 2
    ;;
esac
