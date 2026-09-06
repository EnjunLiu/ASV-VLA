import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    workspace = Path(get_package_prefix("bringup")).resolve().parents[1]
    model_dir = os.environ.get(
        "ASV_VLA_MODEL_DIR",
        str(workspace / "models"),
    )
    bridge_config = os.path.join(
        get_package_share_directory("bridge"), "config", "ue_bridge.yaml"
    )
    backend = LaunchConfiguration("backend")
    models = LaunchConfiguration("model_dir")
    return LaunchDescription([
        DeclareLaunchArgument("backend", default_value="isaac"),
        DeclareLaunchArgument("color", default_value="red"),
        DeclareLaunchArgument("standoff", default_value="4"),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("model_dir", default_value=model_dir),
        DeclareLaunchArgument("execution_address", default_value=""),
        DeclareLaunchArgument("execution_port", default_value="8081"),
        Node(
            package="bridge",
            executable="bridge_node",
            name="ue_bridge",
            output="screen",
            parameters=[bridge_config, {
                "execution_address": LaunchConfiguration("execution_address"),
                "execution_port": LaunchConfiguration("execution_port"),
            }],
            condition=IfCondition(PythonExpression(["'", backend, "' == 'ue'"])),
        ),
        Node(
            package="asv_vla",
            executable="vla_node",
            name="asv_vla",
            output="screen",
            arguments=[
                "--backend", backend,
                "--color", LaunchConfiguration("color"),
                "--standoff", LaunchConfiguration("standoff"),
                "--device", LaunchConfiguration("device"),
                "--weights", [models, "/actor_ppo_semantic16_v14_isaaclab_deploysafe.pt"],
                "--qwen-embed", [models, "/qwen_task_embed.npz"],
                "--hf-home", [models, "/hf"],
            ],
        ),
    ])
