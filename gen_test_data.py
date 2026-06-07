"""
根据 lance_def.txt 的数据结构定义，伪造 20 帧 0.2s 帧率的自动驾驶数据，
写入 Lance 文件用于测试。
"""

import math
import time
import pickle
from typing import Dict, Any, List

from lance_utils import write_lance_frames


def _make_sensor_cfg() -> Dict[str, Any]:
    """伪造传感器标定参数（简化 mock）"""
    return {
        "cam_calib": {
            f"cam_{pos}": {
                "intrinsic": [1200.0, 0.0, 960.0, 0.0, 1200.0, 640.0, 0.0, 0.0, 1.0],
                "extrinsic": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
            for pos in ["front", "front_30fov", "front_right", "front_left",
                        "rear", "rear_left", "rear_right"]
        },
        "ldr_calib": {
            "lidar_at128_front_left": {"rotation": [1, 0, 0, 0], "translation": [1.5, 0.3, 1.2]},
            "right_lidarlidar_at128_front_right": {"rotation": [1, 0, 0, 0], "translation": [1.5, -0.3, 1.2]},
        },
        "rad_calib": {},
    }


def make_frame(
    frame_index: int,
    base_utc: float,
    dt: float,
    ego_x: float,
    ego_y: float,
    ego_vx: float,
    ego_vy: float,
    heading: float,
) -> Dict[str, Any]:
    """
    生成单帧数据。

    Args:
        frame_index: 帧号
        base_utc: 起始 UTC 时间戳
        dt: 帧间隔（秒）
        ego_x, ego_y: 自车 ENU 坐标
        ego_vx, ego_vy: 自车速度分量
        heading: 航向角（度）
    """
    utc = base_utc + frame_index * dt

    # 自车状态历史 41 帧 + 未来 80 帧 = 121 帧
    # 用当前真实值代替，简化模拟
    ego_state_row = [
        ego_x, ego_y, 0.0,            # x, y, z
        math.radians(heading),         # heading (rad)
        ego_vx, ego_vy,                # vx, vy
        0.0, 0.0,                      # ax, ay
        1, 1.8, 4.9, 1.8,             # type, width, length, height
        0.0, 0.0,                      # steer, steer_fix
        utc,                           # timestep
    ]
    ego_states = [ego_state_row] * 121

    # 一个目标车辆
    agent_state_row = [
        ego_x + 20.0, ego_y - 2.0, 0.0,  # x, y, z
        math.radians(heading),            # heading (rad)
        ego_vx - 1.0, ego_vy,             # vx, vy
        0.0, 0.0,                         # ax, ay
        12, 1.8, 4.5, 1.6,                # type, width, length, height
        1.0,                              # confidence
        utc,                              # timestamps
    ]

    return {
        "dat_name": "test_dat_001",
        "timestamp": utc,
        "frameindex": frame_index,
        "sensor_cfg": _make_sensor_cfg(),
        "ego": {
            "vel": math.sqrt(ego_vx**2 + ego_vy**2),
            "acc": (0.0, 0.0),
            "yaw_rate": 0.0,
            "heading_angle_global": heading,
            "pos_utm": [ego_x, ego_y],
            "pos_enu": [ego_x, ego_y],
            "turn_signal": 0,
            "steering_angle": 0.0,
            "steering_angle_offset": 0.0,
            "decision": "cruise",
            "ego_states": ego_states,
            "ego_states_from_odom": [],
            "ego_pose_odo_future": [],
            "longitude": 108.95 + frame_index * 0.00001,
            "latitude": 34.26 + frame_index * 0.00001,
            "first_frame_longitude": 108.95,
            "first_frame_latitude": 34.26,
            "first_frame_pos_enu": [0.0, 0.0],
            "rainfall_level": 0.0,
        },
        "agents": [
            {
                "track_id": f"agent_{1001 + frame_index}",
                "type": 12,
                "pos_utm": [ego_x + 20.0, ego_y - 2.0],
                "pos_vcs": [20.0, -2.0],
                "pos_enu": [ego_x + 20.0, ego_y - 2.0],
                "heading_angle_global": heading,
                "heading_angle_local": 0.0,
                "yaw_rate": 0.0,
                "vel": (ego_vx - 1.0, ego_vy),
                "acc": (0.0, 0.0),
                "width": 1.8,
                "lenght": 4.5,
                "height": 1.6,
                "confidence": 100.0,
                "agent_state": [agent_state_row],
                "agent_state_from_odom": [],
                "agent_state_from_odom_refined": [],
                "lane_info": {
                    "direction": 1,
                    "distance": 0,
                    "orientation": 0,
                },
            },
        ],
        "sensor": {
            "camera": {
                "cam_front": {
                    "timestamp_tsn": frame_index * 100000,
                    "data_path": f"/data/cam/front/{frame_index:04d}.jpg",
                },
            },
            "lidar": {
                "lidar_at128_front_left": {
                    "timestamp_tsn": frame_index * 100000,
                    "data_path": f"/data/lidar/{frame_index:04d}.pcd",
                },
            },
        },
        "map": {
            "lanes": {
                "lane_1": {
                    "points": [
                        [0.0 * (frame_index + 1), 1.75, 0.0],
                        [20.0 * (frame_index + 1), 1.75, 0.0],
                        [40.0 * (frame_index + 1), 1.75, 0.0],
                    ],
                    "type": 1,
                    "width": 3.5,
                    "length": 50.0,
                    "speed_max": 120,
                    "speed_min": 60,
                },
            },
            "lines": {},
            "connected_affinity_matrix": [],
            "connected_lanes_order_list": [],
            "map_lanes_vectors": [],
            "crosswalk_features": [],
            "stopline_features": [],
            "boundary_features": [],
            "vp_trafficlights_vectors": [],
            "pnc_ego_trafficlight_vectors": [],
            "lite_map_lanes": {},
            "lite_map_lines": {},
            "lite_map_ground_markings": {},
            "intersection_roads": {},
            "target_link_path": [],
        },
        "algo_env": {
            "sync_utc": utc,
            "header": {},
            "position": {
                "pose": {
                    "position": {"x": ego_x, "y": ego_y, "z": 0.0},
                },
            },
            "lanes": [],
            "lines": [],
            "boundaries": [],
            "ground_markings": [],
            "traffic_lights": [],
            "traffic_sign": [],
            "parking_slots": [],
            "static_objs": [],
            "barrier_gates": [],
            "junctions": [],
            "seq_header": {},
            "road_info": {},
            "raw_lines": [],
        },
        "sd_map": {
            "path_info": {},
        },
        "route": {
            "route_path": [],
            "refline_path": [],
            "traj_path": [],
        },
        "targets_states": [],
        "targets_ids": [],
        "meta_decision": [],
        "last_meta_decision": [],
    }


def generate_test_data(
    num_frames: int = 20,
    dt: float = 0.2,
    output_path: str = "/tmp/test_lance_data",
    column: str = "src_label_stag_1",
    speed_mps: float = 10.0,
) -> str:
    """
    生成测试用自动驾驶数据，写入 LanceDB。

    Args:
        num_frames: 帧数
        dt: 帧间隔（秒）
        output_path: LanceDB 目录路径
        column: 列名
        speed_mps: 自车速度（m/s）

    Returns:
        LanceDB 路径
    """
    base_utc = time.time()
    heading = 45.0  # 东北方向

    frames: List[Dict[str, Any]] = []
    for i in range(num_frames):
        ego_x = speed_mps * dt * i * math.cos(math.radians(heading))
        ego_y = speed_mps * dt * i * math.sin(math.radians(heading))
        vx = speed_mps * math.cos(math.radians(heading))
        vy = speed_mps * math.sin(math.radians(heading))

        frame = make_frame(
            frame_index=i,
            base_utc=base_utc,
            dt=dt,
            ego_x=round(ego_x, 3),
            ego_y=round(ego_y, 3),
            ego_vx=round(vx, 3),
            ego_vy=round(vy, 3),
            heading=heading,
        )
        frames.append(frame)

    write_lance_frames(output_path, frames, column=column, mode="overwrite")
    return output_path


if __name__ == "__main__":
    path = generate_test_data(
        num_frames=20,
        dt=0.2,
        output_path="/tmp/test_lance_data",
    )
    print(f"✅ 测试数据已写入: {path}")
    print(f"   帧数: 20, 帧率: 0.2s, 速度: 10m/s (~36km/h)")
