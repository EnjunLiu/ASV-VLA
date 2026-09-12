ARG BASE_IMAGE=nvcr.io/nvidia/l4t-cuda:12.6.11-runtime
ARG ROS_DISTRO=humble

FROM ${BASE_IMAGE} AS runtime-base

ARG ROS_DISTRO
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=${ROS_DISTRO} \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    ASV_VLA_MODEL_DIR=/opt/asv_vla/models \
    FASTRTPS_DEFAULT_PROFILES_FILE=/opt/asv_vla/fastdds.xml \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TRANSFORMERS_NO_TF=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN printf 'Acquire::Retries "20";\nAcquire::http::Pipeline-Depth "0";\nAcquire::https::Pipeline-Depth "0";\n' \
        > /etc/apt/apt.conf.d/80-retries \
    && apt_install() { for attempt in $(seq 1 20); do apt-get install -y --no-install-recommends "$@" && return 0; sleep 2; done; return 1; } \
    && apt-get update && apt_install \
        ca-certificates \
        curl \
        gnupg \
        locales \
    && locale-gen en_US.UTF-8 \
    && curl --retry 10 --retry-all-errors -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && curl --retry 10 --retry-all-errors -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb \
        -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt_install \
        libcudnn9-cuda-12=9.20.0.48-1 \
        libopenblas0 \
        python3-pip \
        ros-${ROS_DISTRO}-ament-index-python \
        ros-${ROS_DISTRO}-launch \
        ros-${ROS_DISTRO}-launch-ros \
        ros-${ROS_DISTRO}-nav-msgs \
        ros-${ROS_DISTRO}-rclcpp \
        ros-${ROS_DISTRO}-rclpy \
        ros-${ROS_DISTRO}-rmw-fastrtps-cpp \
        ros-${ROS_DISTRO}-ros2launch \
        ros-${ROS_DISTRO}-rosgraph-msgs \
        ros-${ROS_DISTRO}-sensor-msgs \
        ros-${ROS_DISTRO}-std-msgs \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_WHEEL=https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
COPY requirements-container.txt /tmp/requirements-container.txt
RUN python3 -m pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu keras || true \
    && python3 -m pip install --timeout 1000 --retries 20 "${TORCH_WHEEL}"
RUN apt_install() { for attempt in $(seq 1 20); do apt-get install -y --no-install-recommends "$@" && return 0; sleep 2; done; return 1; } \
    && apt-get update \
    && apt_install \
        cuda-cupti-12-6=12.6.80-1 \
        libcusparselt0=0.7.1.0-1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --timeout 1000 --retries 20 --only-binary=:all: regex==2026.9.3
RUN python3 -m pip install --timeout 1000 --retries 20 -r /tmp/requirements-container.txt
RUN python3 - <<'PY'
import cv2
import torch
import transformers

print("container torch", torch.__version__, "cuda", torch.version.cuda)
print("container transformers", transformers.__version__)
print("container opencv", cv2.__version__)
PY

FROM runtime-base AS builder

RUN apt_install() { for attempt in $(seq 1 20); do apt-get install -y --no-install-recommends "$@" && return 0; sleep 2; done; return 1; } \
    && apt-get update && apt_install \
        cmake \
        g++ \
        make \
        nlohmann-json3-dev \
        python3-colcon-common-extensions \
        ros-${ROS_DISTRO}-ament-cmake \
        ros-${ROS_DISTRO}-ament-cmake-python \
        ros-${ROS_DISTRO}-rosidl-default-generators \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/asv_vla
COPY src ./src
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release

FROM runtime-base AS runtime

WORKDIR /opt/asv_vla
COPY --from=builder /opt/asv_vla/install ./install
COPY fastdds.xml ./fastdds.xml
COPY docker/entrypoint.sh /usr/local/bin/asv-vla-entrypoint
RUN chmod +x /usr/local/bin/asv-vla-entrypoint

ENTRYPOINT ["/usr/local/bin/asv-vla-entrypoint"]
CMD ["ros2", "launch", "bringup", "vla.launch.py", "backend:=isaac", "color:=red", "standoff:=4", "device:=cuda"]
