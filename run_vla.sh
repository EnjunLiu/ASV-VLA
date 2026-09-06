#!/usr/bin/env bash
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
source "${workspace}/install/setup.bash"
export FASTRTPS_DEFAULT_PROFILES_FILE="${workspace}/fastdds.xml"
exec ros2 launch bringup vla.launch.py "$@"
