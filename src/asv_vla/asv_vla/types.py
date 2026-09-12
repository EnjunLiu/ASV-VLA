from __future__ import annotations

from dataclasses import dataclass

# 无人船通过传感器（IMU、GPS等）获取的自身状态数据
@dataclass
class SensorState:
    t: float
    yaw_rate: float = 0.0 # 无人船的偏航角速度，可通过 IMU 计算得到
    surge_velocity: float = 0.0 # 无人船的纵向速度，可通过 GNSS + IMU 计算得到
    sway_velocity: float = 0.0 # 无人船的横向速度，可通过 GNSS + IMU 计算得到
    ego_pos: tuple[float, float, float] = (0.0, 0.0, 0.0) # 世界坐标，只用于仿真相关
    ego_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0) # 世界坐标系下的四元数，只用于仿真相关
