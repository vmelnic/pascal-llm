#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
profile_name="${1:-}"

[[ "${profile_name}" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || {
  echo "usage: $0 <profile>" >&2
  exit 2
}

config_file="${PASCAL_SERVER_CONFIG:-${HOME}/.config/pascal-llm/server.env}"
[[ -r "${config_file}" ]] || {
  echo "server config is missing: ${config_file}" >&2
  exit 2
}

set -a
# shellcheck disable=SC1090
source "${config_file}"
set +a

PASCAL_MODEL_ROOT="${PASCAL_MODEL_ROOT:-${repo_root}/work/models}"
PASCAL_LLAMA_ROOT="${PASCAL_LLAMA_ROOT:-${repo_root}/work/llama.cpp}"
profile_dir="${PASCAL_MODEL_PROFILE_DIR:-${repo_root}/config/models}"
profile_file="${profile_dir}/${profile_name}.env"
[[ -r "${profile_file}" ]] || {
  echo "model profile is missing: ${profile_file}" >&2
  exit 2
}

set -a
# shellcheck disable=SC1090
source "${profile_file}"
set +a

required_variables=(
  PASCAL_MODEL_PATH
  PASCAL_MODEL_HF_REPO
  PASCAL_MODEL_HF_REVISION
  PASCAL_MODEL_HF_FILE
  PASCAL_MODEL_FILE_SIZE
  PASCAL_MODEL_FILE_SHA256
)
for variable_name in "${required_variables[@]}"; do
  [[ -n "${!variable_name:-}" ]] || {
    echo "model profile does not declare downloadable artifact variable: ${variable_name}" >&2
    exit 2
  }
done

[[ "${PASCAL_MODEL_FILE_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "PASCAL_MODEL_FILE_SIZE must be a positive integer" >&2
  exit 2
}
[[ "${PASCAL_MODEL_FILE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "PASCAL_MODEL_FILE_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 2
}
[[ "$(basename "${PASCAL_MODEL_PATH}")" == "${PASCAL_MODEL_HF_FILE}" ]] || {
  echo "PASCAL_MODEL_PATH basename must match PASCAL_MODEL_HF_FILE" >&2
  exit 2
}

hf_binary="${PASCAL_HF_BINARY:-${HOME}/.local/bin/hf}"
[[ -x "${hf_binary}" ]] || {
  echo "Hugging Face CLI is missing: ${hf_binary}" >&2
  exit 2
}

# Optional repository path when the published name differs from the local
# artifact name, for example a subfolder or an intentionally renamed extension.
source_file="${PASCAL_MODEL_HF_SOURCE_FILE:-${PASCAL_MODEL_HF_FILE}}"
[[ "${source_file}" =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ && "${source_file}" != *..* ]] || {
  echo "PASCAL_MODEL_HF_SOURCE_FILE must be a safe relative repository path" >&2
  exit 2
}

# Optional small companion files (tokenizer, config) published beside the
# artifact. Only the main artifact carries the pinned size and SHA-256; the
# companions are pinned by the repository revision.
read -r -a extra_files <<< "${PASCAL_MODEL_HF_EXTRA_FILES:-}"
for extra_file in "${extra_files[@]}"; do
  [[ "${extra_file}" =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ && "${extra_file}" != *..* ]] || {
    echo "PASCAL_MODEL_HF_EXTRA_FILES entries must be safe relative repository paths" >&2
    exit 2
  }
done

target_dir="$(dirname "${PASCAL_MODEL_PATH}")"
candidate_dir="${target_dir}.partial"
candidate_file="${candidate_dir}/${PASCAL_MODEL_HF_FILE}"
completion_file="${target_dir}/.complete"

if [[ -f "${PASCAL_MODEL_PATH}" && -f "${completion_file}" ]] &&
   grep -Fqx "sha256=${PASCAL_MODEL_FILE_SHA256}" "${completion_file}" &&
   [[ "$(stat -c %s "${PASCAL_MODEL_PATH}")" == "${PASCAL_MODEL_FILE_SIZE}" ]]; then
  echo "model artifact is already complete: ${PASCAL_MODEL_PATH}"
  exit 0
fi

[[ ! -e "${target_dir}" ]] || {
  echo "incomplete target exists and was not overwritten: ${target_dir}" >&2
  exit 2
}

mkdir -p "${candidate_dir}"
HF_XET_HIGH_PERFORMANCE=1 "${hf_binary}" download \
  "${PASCAL_MODEL_HF_REPO}" \
  "${source_file}" \
  "${extra_files[@]}" \
  README.md \
  --revision "${PASCAL_MODEL_HF_REVISION}" \
  --local-dir "${candidate_dir}" \
  --max-workers 4

if [[ "${source_file}" != "${PASCAL_MODEL_HF_FILE}" ]]; then
  [[ -f "${candidate_dir}/${source_file}" ]] || {
    echo "download did not produce ${candidate_dir}/${source_file}" >&2
    exit 1
  }
  mv "${candidate_dir}/${source_file}" "${candidate_file}"
fi
[[ -f "${candidate_file}" ]] || {
  echo "download did not produce ${candidate_file}" >&2
  exit 1
}
actual_size="$(stat -c %s "${candidate_file}")"
[[ "${actual_size}" == "${PASCAL_MODEL_FILE_SIZE}" ]] || {
  echo "model artifact size mismatch: expected ${PASCAL_MODEL_FILE_SIZE}, got ${actual_size}" >&2
  exit 1
}
actual_sha256="$(sha256sum "${candidate_file}" | awk '{ print $1 }')"
[[ "${actual_sha256}" == "${PASCAL_MODEL_FILE_SHA256}" ]] || {
  echo "model artifact SHA-256 mismatch" >&2
  exit 1
}

printf 'repo=%s\nrevision=%s\nsource=%s\nfile=%s\nsize=%s\nsha256=%s\n' \
  "${PASCAL_MODEL_HF_REPO}" \
  "${PASCAL_MODEL_HF_REVISION}" \
  "${source_file}" \
  "${PASCAL_MODEL_HF_FILE}" \
  "${PASCAL_MODEL_FILE_SIZE}" \
  "${PASCAL_MODEL_FILE_SHA256}" \
  > "${candidate_dir}/.complete"
mv "${candidate_dir}" "${target_dir}"

echo "model artifact published: ${PASCAL_MODEL_PATH}"
