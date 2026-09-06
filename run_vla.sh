#!/usr/bin/env bash
set -eo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
source "${workspace}/install/setup.bash"
set -u
export FASTRTPS_DEFAULT_PROFILES_FILE="${workspace}/fastdds.xml"
export ASV_VLA_MODEL_DIR="${ASV_VLA_MODEL_DIR:-${workspace}/models}"
exec ros2 launch bringup vla.launch.py "$@"
