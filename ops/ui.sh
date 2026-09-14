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

for variable_name in PASCAL_REMOTE_HOST PASCAL_REMOTE_ROOT; do
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

remote_host="${PASCAL_REMOTE_HOST}"
remote_root="${PASCAL_REMOTE_ROOT}"
compose_file="${remote_root}/compose.yaml"

sync_remote() {
  "${script_dir}/model.sh" sync
}

provision_secrets() {
  ssh "${remote_host}" "REMOTE_ROOT='${remote_root}' bash -s" <<'REMOTE'
set -euo pipefail

config_dir="${REMOTE_ROOT}/config"
mkdir -p "${config_dir}"

create_config() {
  local target="$1"
  local template="$2"
  shift 2

  if [[ -e "${target}" ]]; then
    return
  fi

  cp "${template}" "${target}"
  while (( $# > 0 )); do
    local placeholder="$1"
    local secret
    secret="$(openssl rand -hex 32)"
    sed -i "0,/${placeholder}/s//${secret}/" "${target}"
    shift
  done
  chmod 600 "${target}"
}

create_config \
  "${config_dir}/open-webui.env" \
  "${config_dir}/open-webui.env.example" \
  replace-with-openssl-rand-hex-32
REMOTE

  ssh "${remote_host}" \
    "python3 '${remote_root}/ops/configure-open-webui.py' \
      '${remote_root}' '${remote_root}/config/open-webui.env'"
}

compose_remote() {
  ssh "${remote_host}" \
    "docker compose --project-directory '${remote_root}' --file '${compose_file}' $*"
}

command_name="${1:-status}"
case "${command_name}" in
  start)
    sync_remote
    provision_secrets
    compose_remote "pull"
    compose_remote "up -d"
    ;;
  stop)
    compose_remote "stop"
    ;;
  restart)
    sync_remote
    provision_secrets
    compose_remote "up -d --force-recreate"
    ;;
  status)
    compose_remote "ps"
    ;;
  logs)
    compose_remote "logs --tail 200"
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status|logs}" >&2
    exit 2
    ;;
esac
