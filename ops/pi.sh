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

[[ -n "${PASCAL_SERVER_URL:-}" ]] || {
  echo "required deployment variable is empty: PASCAL_SERVER_URL" >&2
  exit 2
}
[[ "${PASCAL_SERVER_URL}" =~ ^https?://[A-Za-z0-9._:-]+$ ]] || {
  echo "PASCAL_SERVER_URL must contain only the scheme, host and optional port" >&2
  exit 2
}

profiles_dir="${repo_root}/config/models"
model_id="${PASCAL_MODEL_ID:-}"

if [[ -n "${1:-}" && -f "${profiles_dir}/${1}.env" ]]; then
  profile_name="$1"
  shift
  model_id="$(sed -n 's/^PASCAL_MODEL_ID=//p' "${profiles_dir}/${profile_name}.env")"
fi

if [[ -z "${model_id}" ]]; then
  model_id="$(curl -fsS --max-time 5 \
    -H 'Authorization: Bearer local' \
    "${PASCAL_SERVER_URL%/}/v1/models" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
fi

command -v pi >/dev/null 2>&1 || {
  echo "pi CLI is not installed or not on PATH" >&2
  exit 2
}

exec pi \
  --approve \
  --provider pascal-llm \
  --model "${model_id}" \
  --thinking xhigh \
  "$@"
