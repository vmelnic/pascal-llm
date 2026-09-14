#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
profile_name="${1:-}"

[[ "${profile_name}" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || {
  echo "usage: $0 <profile>" >&2
  exit 2
}

profile_dir="${PASCAL_IMAGE_PROFILE_DIR:-${repo_root}/config/images}"
profile_file="${profile_dir}/${profile_name}.env"
[[ -r "${profile_file}" ]] || {
  echo "image profile is missing: ${profile_file}" >&2
  exit 2
}

PASCAL_MODEL_ROOT="${PASCAL_MODEL_ROOT:-${repo_root}/work/models}"
set -a
# shellcheck disable=SC1090
source "${profile_file}"
set +a

required_variables=(
  PASCAL_IMAGE_MODEL_PATH
  PASCAL_IMAGE_HF_REPO
  PASCAL_IMAGE_HF_REVISION
  PASCAL_IMAGE_HF_FILE
  PASCAL_IMAGE_FILE_SIZE
  PASCAL_IMAGE_FILE_SHA256
)
for variable_name in "${required_variables[@]}"; do
  [[ -n "${!variable_name:-}" ]] || {
    echo "required image-profile variable is empty: ${variable_name}" >&2
    exit 2
  }
done

[[ "${PASCAL_IMAGE_FILE_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "PASCAL_IMAGE_FILE_SIZE must be a positive integer" >&2
  exit 2
}
[[ "${PASCAL_IMAGE_FILE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "PASCAL_IMAGE_FILE_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 2
}

hf_binary="${PASCAL_HF_BINARY:-${HOME}/.local/bin/hf}"
[[ -x "${hf_binary}" ]] || {
  echo "Hugging Face CLI is missing: ${hf_binary}" >&2
  exit 2
}

target_dir="$(dirname "${PASCAL_IMAGE_MODEL_PATH}")"
candidate_dir="${target_dir}.partial"
candidate_file="${candidate_dir}/${PASCAL_IMAGE_HF_FILE}"
completion_file="${target_dir}/.complete"

if [[ -f "${PASCAL_IMAGE_MODEL_PATH}" && -f "${completion_file}" ]] &&
   grep -Fqx "sha256=${PASCAL_IMAGE_FILE_SHA256}" "${completion_file}" &&
   [[ "$(stat -c %s "${PASCAL_IMAGE_MODEL_PATH}")" == "${PASCAL_IMAGE_FILE_SIZE}" ]]; then
  echo "image artifact is already complete: ${PASCAL_IMAGE_MODEL_PATH}"
  exit 0
fi

[[ ! -e "${target_dir}" ]] || {
  echo "incomplete target exists and was not overwritten: ${target_dir}" >&2
  exit 2
}

mkdir -p "${candidate_dir}"
HF_XET_HIGH_PERFORMANCE=1 "${hf_binary}" download \
  "${PASCAL_IMAGE_HF_REPO}" \
  "${PASCAL_IMAGE_HF_FILE}" \
  README.md \
  --revision "${PASCAL_IMAGE_HF_REVISION}" \
  --local-dir "${candidate_dir}" \
  --max-workers 1

[[ -f "${candidate_file}" ]] || {
  echo "download did not produce ${candidate_file}" >&2
  exit 1
}
actual_size="$(stat -c %s "${candidate_file}")"
[[ "${actual_size}" == "${PASCAL_IMAGE_FILE_SIZE}" ]] || {
  echo "image artifact size mismatch: expected ${PASCAL_IMAGE_FILE_SIZE}, got ${actual_size}" >&2
  exit 1
}
actual_sha256="$(sha256sum "${candidate_file}" | awk '{ print $1 }')"
[[ "${actual_sha256}" == "${PASCAL_IMAGE_FILE_SHA256}" ]] || {
  echo "image artifact SHA-256 mismatch" >&2
  exit 1
}

printf 'repo=%s\nrevision=%s\nfile=%s\nsize=%s\nsha256=%s\n' \
  "${PASCAL_IMAGE_HF_REPO}" \
  "${PASCAL_IMAGE_HF_REVISION}" \
  "${PASCAL_IMAGE_HF_FILE}" \
  "${PASCAL_IMAGE_FILE_SIZE}" \
  "${PASCAL_IMAGE_FILE_SHA256}" \
  > "${candidate_dir}/.complete"
mv "${candidate_dir}" "${target_dir}"

echo "image artifact published: ${PASCAL_IMAGE_MODEL_PATH}"
