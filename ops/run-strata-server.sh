#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
source_dir="${PASCAL_STRATA_SOURCE_DIR:-${repo_root}/work/strata}"
config_path="${PASCAL_STRATA_CONFIG:-${source_dir}/strata-swift-iq3_s.json}"

[[ -x "${source_dir}/.venv/bin/python" ]] || {
  echo "Strata Python environment is missing: ${source_dir}/.venv" >&2
  exit 2
}
[[ -f "${source_dir}/serve/server.py" ]] || {
  echo "Strata server source is missing: ${source_dir}/serve/server.py" >&2
  exit 2
}

"${script_dir}/configure-strata.py" --config "${config_path}" --check

if systemctl --user is-active --quiet pascal-llm.service; then
  echo "pascal-llm.service already owns P100 devices 0,1,2" >&2
  exit 2
fi

export STRATA_EXPERIMENTAL_SM60=1
exec "${source_dir}/.venv/bin/python" \
  "${source_dir}/serve/server.py" \
  --engine strata \
  --config "${config_path}" \
  --port 8080
