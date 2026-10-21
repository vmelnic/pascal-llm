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
[[ "${PASCAL_REMOTE_ROOT}" =~ ^/[A-Za-z0-9._/-]+$ && "${PASCAL_REMOTE_ROOT}" != "/" ]] || {
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
service_name="pascal-strata.service"

sync_remote() {
  "${script_dir}/model.sh" sync
}

install_service() {
  ssh "${remote_host}" \
    "mkdir -p \"\$HOME/.config/systemd/user\"; \
     sed 's|@PASCAL_REMOTE_ROOT@|${remote_root}|g' \
       '${remote_root}/ops/pascal-strata.service' \
       > \"\$HOME/.config/systemd/user/${service_name}.tmp\"; \
     mv \"\$HOME/.config/systemd/user/${service_name}.tmp\" \
       \"\$HOME/.config/systemd/user/${service_name}\"; \
     systemctl --user daemon-reload"
}

require_llama_inactive() {
  if ssh "${remote_host}" "systemctl --user is-active --quiet pascal-llm.service"; then
    echo "pascal-llm.service owns P100 devices 0,1,2; stop it with ./ops/model.sh stop" >&2
    exit 2
  fi
}

wait_until_ready() {
  local attempt
  local health
  for ((attempt = 0; attempt < 600; attempt += 1)); do
    health="$(curl --fail --silent --max-time 2 \
      -H 'Authorization: Bearer local' "${server_url}/health" 2>/dev/null || true)"
    if [[ "${health}" == *'"status": "ok"'* && "${health}" == *'"service": "strata"'* ]]; then
      echo "Strata is ready at ${server_url}."
      return 0
    fi
    if ! ssh "${remote_host}" "systemctl --user is-active --quiet '${service_name}'"; then
      ssh "${remote_host}" "journalctl --user -u '${service_name}' -n 120 --no-pager" >&2
      return 1
    fi
    sleep 1
  done
  echo "Strata did not become ready; stopping the unhealthy service" >&2
  ssh "${remote_host}" \
    "systemctl --user stop '${service_name}'; journalctl --user -u '${service_name}' -n 120 --no-pager" >&2
  return 1
}

command_name="${1:-status}"
case "${command_name}" in
  download)
    sync_remote
    require_llama_inactive
    ssh "${remote_host}" "'${remote_root}/ops/download-strata-model.sh'"
    ;;
  start)
    sync_remote
    install_service
    require_llama_inactive
    ssh "${remote_host}" "systemctl --user restart '${service_name}'"
    wait_until_ready
    ;;
  stop)
    ssh "${remote_host}" "systemctl --user stop '${service_name}'"
    ;;
  restart)
    sync_remote
    install_service
    require_llama_inactive
    ssh "${remote_host}" "systemctl --user restart '${service_name}'"
    wait_until_ready
    ;;
  status)
    ssh "${remote_host}" "systemctl --user --no-pager --full status '${service_name}' || true"
    curl --silent --show-error --max-time 3 \
      -H 'Authorization: Bearer local' "${server_url}/health" || true
    echo
    ;;
  logs)
    ssh "${remote_host}" "journalctl --user -u '${service_name}' -n 240 --no-pager"
    ;;
  models)
    curl --fail --silent --show-error --max-time 10 \
      -H 'Authorization: Bearer local' "${server_url}/v1/models"
    echo
    ;;
  pi-config)
    "${script_dir}/configure-pi-strata.py"
    ;;
  pi)
    shift
    exec "${script_dir}/pi.sh" "$@"
    ;;
  *)
    echo "usage: $0 {download|start|stop|restart|status|logs|models|pi-config|pi [Pi args...]}" >&2
    exit 2
    ;;
esac
