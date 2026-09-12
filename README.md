# ASV-VLA

ROS 2 deployment workspace for task-conditioned visual target following on
Jetson Orin Nano. The runtime turns a camera stream and vessel state into a
body-frame desired displacement while keeping perception, tracking and policy
inference outside the low-level controller.

## Runtime pipeline

1. OWL-ViT detects task-relevant vessels from the live RGB stream.
2. A Kalman tracker maintains identity and body-frame position/velocity through
   intermittent detections.
3. The task-conditioned semantic set actor combines per-entity OWL appearance features with a
   precomputed Qwen3 task embedding, selects the target and predicts a
   two-dimensional body-frame displacement. The task embedding conditions both
   target selection and two task-conditioned multiplicative gains that modulate
   the attention value projection and output weights.
4. Range-aware safety gating, action limiting and target-loss handling validate
   the command before it is published on `/espapp/input`.

The actor is rotation-equivariant in the horizontal plane and accepts an
unordered set of tracked entities. Target identity is retained explicitly so a
brief occlusion cannot silently transfer control to a similar distractor.

## Backends

- **Isaac Sim:** subscribes to `/asv/camera/image_raw`, `/asv/imu` and
  `/asv/state_estimate`.
- **Unreal Engine:** the included C++ TCP bridge converts Unreal camera and
  vessel-state packets to `/ue/camera_frame` and `/ue/asv_state`. It also sends
  the resulting command back to Unreal when that backend is used, avoiding a
  native ROS dependency inside UE.

Both paths publish the same existing `interfaces/msg/Input` message on
`/espapp/input`: simulation timestamp, desired body-frame `x/y`, measured
`u/v/r` and a validity flag. The message contract is shared with ASV-Control and
is not redefined by either backend.

## Workspace

- `src/asv_vla/`: perception adapters, Kalman tracking and semantic actor runtime.
- `src/bridge/`: bidirectional Unreal TCP/ROS 2 bridge.
- `src/interfaces/`: ROS 2 messages used at the deployment boundary.
- `src/bringup/`: one launch entry point for both backends.
- `fastdds.xml`: DDS configuration for the distributed Jetson/Isaac deployment.

Model assets are stored locally under the workspace-level ignored `models/`
directory, so the Jetson folder is a self-contained runtime while weights remain
outside GitHub. It contains the deployed actor, `qwen_task_embed.npz` and the
OWL-ViT Hugging Face cache under `models/hf/`. Qwen does not run on Jetson; the
six supported task embeddings are loaded from the small NPZ file. Set
`ASV_VLA_MODEL_DIR` only to override that project-local default.

## Build and run

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
./run_vla.sh backend:=isaac color:=red standoff:=4
```

For Unreal Engine, set `backend:=ue` and provide `execution_address` when the
Unreal host is not resolved by the bridge configuration.

## Docker deployment on Jetson

The deployment image targets Jetson Linux R36.4 on ARM64. It layers NVIDIA's
Jetson CUDA 12.6 runtime, cuDNN and the matching NVIDIA PyTorch wheel instead of
shipping unrelated training tools. Build it directly on the Jetson:

```bash
docker compose build
docker compose run --rm asv-vla python3 -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__)'
docker compose up asv-vla
```

`compose.yaml` uses the NVIDIA runtime and host networking so ROS 2 DDS and the
UE TCP ports retain their existing contracts. The ignored workspace `models/`
directory is mounted read-only at `/opt/asv_vla/models`; weights are neither
copied into the image nor sent in the Docker build context.

Select a backend and task without rebuilding:

```bash
ASV_BACKEND=isaac ASV_COLOR=blue ASV_STANDOFF=3 docker compose up asv-vla
docker compose run --rm asv-vla ros2 launch bringup vla.launch.py \
  backend:=ue color:=red standoff:=4 device:=cuda \
  execution_address:=192.168.137.1 execution_port:=8081
```

The native `colcon` workflow remains supported. Docker is the reproducible
deployment path; it does not change ROS messages, topic semantics or policy
behavior.

The repository contains one deployment policy path only. Generated ROS
workspaces, model artifacts, datasets, experiment logs and offline learning
pipelines are excluded.
