#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
config_file="${PASCAL_SERVER_CONFIG:-${HOME}/.config/pascal-llm/server.env}"
[[ -r "${config_file}" ]] || {
  echo "server config is missing: ${config_file}" >&2
  exit 2
}

set -a
# shellcheck disable=SC1090
source "${config_file}"
set +a

PASCAL_DECODE_GRAPH_SLOTS="${PASCAL_DECODE_GRAPH_SLOTS:-0}"
PASCAL_MODEL_ROOT="${PASCAL_MODEL_ROOT:-${repo_root}/work/models}"
PASCAL_LLAMA_ROOT="${PASCAL_LLAMA_ROOT:-${repo_root}/work/llama.cpp}"
active_profile_file="${PASCAL_ACTIVE_MODEL_FILE:-${HOME}/.config/pascal-llm/active-model}"
model_profile="${PASCAL_MODEL_PROFILE:-}"
if [[ -z "${model_profile}" && -r "${active_profile_file}" ]]; then
  IFS= read -r model_profile < "${active_profile_file}"
fi
[[ "${model_profile}" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || {
  echo "active model profile is missing or invalid: '${model_profile}'" >&2
  exit 2
}

profile_dir="${PASCAL_MODEL_PROFILE_DIR:-${repo_root}/config/models}"
profile_file="${profile_dir}/${model_profile}.env"
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
  PASCAL_MODEL_ID
  PASCAL_LLAMA_ROOT
  PASCAL_CUDA_DEVICES
  PASCAL_TENSOR_SPLIT
  PASCAL_SPLIT_MODE
  PASCAL_CUDA_P2P
  PASCAL_CONTEXT_SIZE
  PASCAL_PARALLEL_SLOTS
  PASCAL_CACHE_TYPE_K
  PASCAL_CACHE_TYPE_V
  PASCAL_PROMPT_CACHE_MIB
  PASCAL_HOST
  PASCAL_PORT
  PASCAL_API_KEY
  PASCAL_THREADS
  PASCAL_BATCH_THREADS
  PASCAL_DECODE_GRAPH_SLOTS
)

for variable_name in "${required_variables[@]}"; do
  [[ -n "${!variable_name:-}" ]] || {
    echo "required variable is empty: ${variable_name}" >&2
    exit 2
  }
done

[[ "${PASCAL_DECODE_GRAPH_SLOTS}" =~ ^[0-9]+$ ]] || {
  echo "PASCAL_DECODE_GRAPH_SLOTS must be a non-negative integer" >&2
  exit 2
}

case "${PASCAL_SPLIT_MODE}" in
  none|layer|row|tensor)
    ;;
  *)
    echo "unsupported PASCAL_SPLIT_MODE: ${PASCAL_SPLIT_MODE}" >&2
    exit 2
    ;;
esac

case "${PASCAL_CUDA_P2P}" in
  0|1)
    ;;
  *)
    echo "PASCAL_CUDA_P2P must be 0 or 1" >&2
    exit 2
    ;;
esac

server_binary="${PASCAL_LLAMA_ROOT}/build/bin/llama-server"
[[ -x "${server_binary}" ]] || {
  echo "llama-server is missing: ${server_binary}" >&2
  exit 2
}
[[ -f "${PASCAL_MODEL_PATH}" ]] || {
  echo "model artifact is missing: ${PASCAL_MODEL_PATH}" >&2
  exit 2
}

declare -a model_arguments=()
mtp_draft_vocab_path=""
if [[ -n "${PASCAL_MMPROJ_PATH:-}" ]]; then
  [[ -f "${PASCAL_MMPROJ_PATH}" ]] || {
    echo "multimodal projector is missing: ${PASCAL_MMPROJ_PATH}" >&2
    exit 2
  }
  model_arguments+=(--mmproj "${PASCAL_MMPROJ_PATH}")
fi

case "${PASCAL_SPEC_TYPE:-none}" in
  none)
    ;;
  draft-mtp)
    for variable_name in \
      PASCAL_SPEC_DRAFT_N_MAX \
      PASCAL_SPEC_DRAFT_N_GPU_LAYERS \
      PASCAL_SPEC_DRAFT_CACHE_TYPE_K \
      PASCAL_SPEC_DRAFT_CACHE_TYPE_V; do
      [[ -n "${!variable_name:-}" ]] || {
        echo "required MTP variable is empty: ${variable_name}" >&2
        exit 2
      }
    done
    [[ "${PASCAL_SPEC_DRAFT_N_MAX}" =~ ^[1-9][0-9]*$ ]] || {
      echo "PASCAL_SPEC_DRAFT_N_MAX must be a positive integer" >&2
      exit 2
    }
    [[ "${PASCAL_SPEC_DRAFT_CACHE_TYPE_K}" == "f16" && \
       "${PASCAL_SPEC_DRAFT_CACHE_TYPE_V}" == "f16" ]] || {
      echo "the active fidelity contract requires F16 MTP K/V caches" >&2
      exit 2
    }
    if [[ -n "${PASCAL_SPEC_DRAFT_MODEL_PATH:-}" ]]; then
      [[ -f "${PASCAL_SPEC_DRAFT_MODEL_PATH}" ]] || {
        echo "MTP draft artifact is missing: ${PASCAL_SPEC_DRAFT_MODEL_PATH}" >&2
        exit 2
      }
      model_arguments+=(--spec-draft-model "${PASCAL_SPEC_DRAFT_MODEL_PATH}")
    fi
    if [[ -n "${PASCAL_MTP_DRAFT_VOCAB_PATH:-}" ]]; then
      [[ -r "${PASCAL_MTP_DRAFT_VOCAB_PATH}" ]] || {
        echo "MTP draft vocabulary is missing: ${PASCAL_MTP_DRAFT_VOCAB_PATH}" >&2
        exit 2
      }
      [[ "${PASCAL_MTP_DRAFT_VOCAB_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "PASCAL_MTP_DRAFT_VOCAB_SHA256 must be a lowercase SHA-256 digest" >&2
        exit 2
      }
      actual_draft_vocab_sha256="$(sha256sum "${PASCAL_MTP_DRAFT_VOCAB_PATH}" | awk '{print $1}')"
      [[ "${actual_draft_vocab_sha256}" == "${PASCAL_MTP_DRAFT_VOCAB_SHA256}" ]] || {
        echo "MTP draft vocabulary hash mismatch" >&2
        exit 2
      }
      awk '
        BEGIN { previous = -1 }
        !/^[0-9]+$/ || $1 <= previous { exit 1 }
        { previous = $1 }
        END { if (NR == 0) exit 1 }
      ' "${PASCAL_MTP_DRAFT_VOCAB_PATH}" || {
        echo "MTP draft vocabulary must contain strictly increasing token IDs" >&2
        exit 2
      }
      mtp_draft_vocab_path="${PASCAL_MTP_DRAFT_VOCAB_PATH}"
    fi
    model_arguments+=(
      --spec-type draft-mtp
      --spec-draft-n-max "${PASCAL_SPEC_DRAFT_N_MAX}"
      --spec-draft-ngl "${PASCAL_SPEC_DRAFT_N_GPU_LAYERS}"
      --spec-draft-type-k "${PASCAL_SPEC_DRAFT_CACHE_TYPE_K}"
      --spec-draft-type-v "${PASCAL_SPEC_DRAFT_CACHE_TYPE_V}"
    )
    ;;
  *)
    echo "unsupported speculative profile capability: ${PASCAL_SPEC_TYPE}" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="${PASCAL_CUDA_DEVICES}"
export GGML_CUDA_P2P="${PASCAL_CUDA_P2P}"
export LLAMA_DEC_SLOTS="${PASCAL_DECODE_GRAPH_SLOTS}"

# These patchset experiments are intentionally disabled: the patch author
# reports non-deterministic MoE output when the graph fusions are enabled.
export GGML_CUDA_FUSE_PRE_ADD=0
export GGML_CUDA_FUSE_ADD_UNARY_MUL=0
if [[ -n "${mtp_draft_vocab_path}" ]]; then
  export LLAMA_MTP_DRAFT_VOCAB="${mtp_draft_vocab_path}"
else
  unset LLAMA_MTP_DRAFT_VOCAB
fi

exec "${server_binary}" \
  --model "${PASCAL_MODEL_PATH}" \
  --alias "${PASCAL_MODEL_ID}" \
  --host "${PASCAL_HOST}" \
  --port "${PASCAL_PORT}" \
  --api-key "${PASCAL_API_KEY}" \
  --n-gpu-layers all \
  --split-mode "${PASCAL_SPLIT_MODE}" \
  --tensor-split "${PASCAL_TENSOR_SPLIT}" \
  --fit off \
  --flash-attn on \
  --cache-type-k "${PASCAL_CACHE_TYPE_K}" \
  --cache-type-v "${PASCAL_CACHE_TYPE_V}" \
  --ctx-size "${PASCAL_CONTEXT_SIZE}" \
  --parallel "${PASCAL_PARALLEL_SLOTS}" \
  --kv-unified \
  --cont-batching \
  --cache-prompt \
  --cache-ram "${PASCAL_PROMPT_CACHE_MIB}" \
  "${model_arguments[@]}" \
  --threads "${PASCAL_THREADS}" \
  --threads-batch "${PASCAL_BATCH_THREADS}" \
  --jinja \
  --metrics \
  --slots
