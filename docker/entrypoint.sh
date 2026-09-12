#!/usr/bin/env bash
set -eo pipefail

source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /opt/asv_vla/install/setup.bash
set -u

if [[ ! -r "${ASV_VLA_MODEL_DIR}/actor_semantic16_deploysafe.pt" ]]; then
    echo "Missing actor weights under ${ASV_VLA_MODEL_DIR}" >&2
    exit 64
fi
if [[ ! -r "${ASV_VLA_MODEL_DIR}/qwen_task_embed.npz" ]]; then
    echo "Missing task embeddings under ${ASV_VLA_MODEL_DIR}" >&2
    exit 64
fi
if [[ ! -d "${ASV_VLA_MODEL_DIR}/hf" ]]; then
    echo "Missing Hugging Face cache under ${ASV_VLA_MODEL_DIR}" >&2
    exit 64
fi

exec "$@"
