#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# Strata v0.2.20 (ops/install-strata-next.sh); its front end lives in terminal/.
# The pinned 0.1.46.3 tree in work/strata remains only as a rollback runtime.
source_dir="${PASCAL_STRATA_SOURCE_DIR:-${repo_root}/work/strata-next}"
active_file="${PASCAL_STRATA_ACTIVE_FILE:-${HOME}/.config/pascal-llm/active-strata-model}"
port=8081

# One P100 service, one model at a time: `swift` (default) or `inkling`.
model="swift"
if [[ -r "${active_file}" ]]; then
  IFS= read -r model < "${active_file}"
fi

[[ -x "${source_dir}/terminal/.venv/bin/python" ]] || {
  echo "Strata Python environment is missing: ${source_dir}/terminal/.venv" >&2
  exit 2
}
[[ -f "${source_dir}/terminal/serve/server.py" ]] || {
  echo "Strata server source is missing: ${source_dir}/terminal/serve/server.py" >&2
  exit 2
}

case "${model}" in
  swift)
    config_path="${PASCAL_STRATA_CONFIG:-${source_dir}/strata-swift-iq3_s.json}"
    "${script_dir}/configure-strata.py" --config "${config_path}" --check
    ;;
  inkling)
    # Inkling 2 Mini SQ-20 with the pascal-llm tensor-parallel decode
    # (patches/strata-next/, `--tp 3` over P100 0,1,2).
    model_path="${PASCAL_INKLING_MODEL:-${repo_root}/work/models/inkling-2-mini-strata-sq20/Inkling-2-Mini-SQ-20.strata}"
    config_path="${source_dir}/strata-inkling-mini-sq20.json"
    log_dir="${repo_root}/out/inkling-mini-strata"
    [[ -x "${source_dir}/build-sm60/src/inkling2/strata-inkling" ]] || {
      echo "strata-inkling is missing; run ops/install-strata-next.sh" >&2
      exit 2
    }
    [[ -f "${model_path}" ]] || {
      echo "Inkling SQ-20 pack is missing; run PASCAL_MODEL_PROFILE_DIR=config/strata ops/download-model.sh inkling-mini-sq20" >&2
      exit 2
    }
    mkdir -p "${log_dir}"
    python3 - "${config_path}" "${source_dir}" "${model_path}" "${log_dir}" "${port}" <<'EOF'
import json, sys
path, src, model, logs, port = sys.argv[1:6]
json.dump({
    "exe": f"{src}/build-sm60/src/inkling2/strata-inkling",
    "arch": "inkling2_loop",
    "args": ["--native", model, "--max-context", "262144", "--kv", "fp16", "--draft", "off", "--tp", "3"],
    "cwd": src,
    "tokenizer": model.rsplit("/", 1)[0],
    "model_name": "inkling-2-mini-sq20",
    "log": f"{logs}/engine.log",
    "port": int(port), "cuda": 12, "gpu": [0, 1, 2], "gpus_asked": True, "layer_split": "auto",
    "host": "0.0.0.0", "api_key": "local", "open_browser": False, "parallel": 1,
}, open(path, "w"), indent=1)
EOF
    ;;
  *)
    echo "unknown active Strata model in ${active_file}: '${model}' (expected swift or inkling)" >&2
    exit 2
    ;;
esac

python3 "${script_dir}/check-service-placement.py" \
  --service pascal-strata --devices 0,1,2 --port "${port}"

export STRATA_EXPERIMENTAL_SM60=1
cd "${source_dir}/terminal"
exec "${source_dir}/terminal/.venv/bin/python" \
  "${source_dir}/terminal/serve/server.py" \
  --engine strata \
  --config "${config_path}" \
  --port "${port}"
