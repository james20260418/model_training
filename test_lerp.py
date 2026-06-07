"""
测试 lance_utils 中的 lerp_snapshots 函数。
"""
import copy
import math
from lance_utils import lerp_snapshots, _lerp_angle


def make_dummy_frame(
    frameindex: int,
    timestamp: float,
    ego_enu_x: float,
    ego_enu_y: float,
    ego_heading: float,
    ego_vel: float,
    agents: list = None,
) -> dict:
    return {
        "dat_name": "test_dat",
        "timestamp": timestamp,
        "frameindex": frameindex,
        "ego": {
            "vel": ego_vel,
            "acc": (0.5, 0.1),
            "yaw_rate": 0.02,
            "heading_angle_global": ego_heading,
            "pos_enu": [ego_enu_x, ego_enu_y],
            "pos_utm": [ego_enu_x, ego_enu_y],
            "turn_signal": 0,
            "steering_angle": 0.0,
            "steering_angle_offset": 0.0,
            "decision": "cruise",
            "longitude": 108.95,
            "latitude": 34.26,
            "first_frame_longitude": 108.95,
            "first_frame_latitude": 34.26,
            "first_frame_pos_enu": [0.0, 0.0],
            "ego_states": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            1, 1.8, 4.9, 1.8, 0.0, 0.0, timestamp]] * 121,
            "ego_states_from_odom": [],
            "ego_pose_odo_future": [],
            "rainfall_level": 0.0,
        },
        "agents": agents or [],
        "sensor_cfg": {"cam_calib": {}, "ldr_calib": {}, "rad_calib": {}},
        "sensor": {"camera": {}, "lidar": {}},
        "map": {"lanes": {}, "lines": {}},
        "algo_env": {"sync_utc": timestamp, "position": {}, "lanes": []},
        "sd_map": {"path_info": {}},
        "route": {},
        "targets_states": [],
        "targets_ids": [],
        "meta_decision": [],
        "last_meta_decision": [],
    }


def test_lerp_basic():
    """基础内差验证"""
    s0 = make_dummy_frame(0, 100.0, ego_enu_x=0.0, ego_enu_y=0.0,
                          ego_heading=45.0, ego_vel=10.0)
    s1 = make_dummy_frame(1, 100.2, ego_enu_x=2.0, ego_enu_y=2.0,
                          ego_heading=47.0, ego_vel=12.0)

    # t=0 → 完全等于 s0
    r0 = lerp_snapshots(s0, s1, 0.0)
    assert r0["frameindex"] == 0
    assert abs(r0["ego"]["pos_enu"][0] - 0.0) < 1e-9
    assert abs(r0["ego"]["vel"] - 10.0) < 1e-9

    # t=1 → 完全等于 s1
    r1 = lerp_snapshots(s0, s1, 1.0)
    assert abs(r1["ego"]["vel"] - 12.0) < 1e-9

    # t=0.5 → 中间值
    r05 = lerp_snapshots(s0, s1, 0.5)
    assert abs(r05["ego"]["pos_enu"][0] - 1.0) < 1e-6, f"got {r05['ego']['pos_enu'][0]}"
    assert abs(r05["ego"]["pos_enu"][1] - 1.0) < 1e-6
    assert abs(r05["ego"]["vel"] - 11.0) < 1e-6
    assert abs(r05["ego"]["acc"][0] - 0.5) < 1e-6  # s0=s1, 不变

    print("✅ test_lerp_basic passed")


def test_lerp_angle():
    """角度跨 360° 边界"""
    s0 = make_dummy_frame(0, 100.0, 0, 0, ego_heading=350.0, ego_vel=10.0)
    s1 = make_dummy_frame(1, 100.2, 0, 0, ego_heading=10.0, ego_vel=10.0)
    r = lerp_snapshots(s0, s1, 0.5)
    # 350° → 10° 最短弧只走 20°，t=0.5 应该是 0°（即 360°）
    assert abs(r["ego"]["heading_angle_global"] - 0.0) < 0.1, \
        f"got {r['ego']['heading_angle_global']}"

    # 另一种方向：10° → 350°，最短弧走 -20°
    s0_ = make_dummy_frame(0, 100.0, 0, 0, ego_heading=10.0, ego_vel=10.0)
    s1_ = make_dummy_frame(1, 100.2, 0, 0, ego_heading=350.0, ego_vel=10.0)
    r2 = lerp_snapshots(s0_, s1_, 0.5)
    assert abs(r2["ego"]["heading_angle_global"] - 0.0) < 0.1, \
        f"got {r2['ego']['heading_angle_global']}"

    print("✅ test_lerp_angle passed")


def test_agents_by_track_id():
    """agent 按 track_id 配对，落单的移除"""
    agent_a = {"track_id": "a1", "type": 12, "pos_enu": [10.0, 0.0],
               "vel": (1.0, 0.0), "heading_angle_global": 90.0,
               "confidence": 100.0, "width": 1.8, "lenght": 4.5,
               "height": 1.6, "agent_state": [], "yaw_rate": 0.0,
               "acc": (0.0, 0.0), "heading_angle_local": 0.0,
               "pos_utm": (10.0, 0.0), "pos_vcs": (10.0, 0.0),
               "lane_info": {}, "traj_valid": {}, "traj_quality": {}}
    agent_b = {"track_id": "a1", "type": 12, "pos_enu": [12.0, 0.0],
               "vel": (1.2, 0.0), "heading_angle_global": 92.0,
               "confidence": 100.0, "width": 1.8, "lenght": 4.5,
               "height": 1.6, "agent_state": [], "yaw_rate": 0.0,
               "acc": (0.1, 0.0), "heading_angle_local": 0.0,
               "pos_utm": (12.0, 0.0), "pos_vcs": (6.0, 0.0),
               "lane_info": {}, "traj_valid": {}, "traj_quality": {}}
    agent_only_s1 = {"track_id": "b2", "type": 12, "pos_enu": [0.0, 20.0],
                     "vel": (0.0, -1.0), "heading_angle_global": 180.0,
                     "confidence": 80.0, "width": 1.8, "lenght": 4.5,
                     "height": 1.6, "agent_state": [], "yaw_rate": 0.0,
                     "acc": (0.0, 0.0), "heading_angle_local": 0.0,
                     "pos_utm": (0.0, 20.0), "pos_vcs": (0.0, 10.0),
                     "lane_info": {}, "traj_valid": {}, "traj_quality": {}}

    s0 = make_dummy_frame(0, 100.0, 0, 0, 45.0, 10.0, agents=[agent_a])
    s1 = make_dummy_frame(1, 100.2, 0, 0, 45.0, 10.0,
                          agents=[agent_b, agent_only_s1])

    r = lerp_snapshots(s0, s1, 0.5)

    # a1 两帧都有，应该被内差，pos_enu.x = 11.0
    a1 = [a for a in r["agents"] if a["track_id"] == "a1"]
    assert len(a1) == 1, f"expected 1 a1, got {len(a1)}"
    assert abs(a1[0]["pos_enu"][0] - 11.0) < 1e-6, \
        f"a1 pos_enu.x = {a1[0]['pos_enu'][0]}"

    # b2 只在 s1 出现→移除
    b2 = [a for a in r["agents"] if a["track_id"] == "b2"]
    assert len(b2) == 0, f"b2 should be removed, got {len(b2)}"

    # a1 的 angle 正确内差 90→92，t=0.5 → 91
    assert abs(a1[0]["heading_angle_global"] - 91.0) < 1e-6

    print("✅ test_agents_by_track_id passed")


def test_trajectory_nearest():
    """轨迹类字段取最近帧"""
    s0 = make_dummy_frame(0, 100.0, 0, 0, 45.0, 10.0)
    s1 = make_dummy_frame(1, 100.2, 0, 0, 45.0, 10.0)
    s0["ego"]["ego_states"][0][0] = 1.23  # 改个值区分

    # t=0.9 → 近 s1，ego_states 应该取 s1 的值
    r = lerp_snapshots(s0, s1, 0.9)
    assert abs(r["ego"]["ego_states"][0][0] - 0.0) < 1e-9, \
        f"ego_states[0][0] = {r['ego']['ego_states'][0][0]} (expect 0.0 from s1)"

    # t=0.1 → 近 s0，应该取 s0
    r2 = lerp_snapshots(s0, s1, 0.1)
    assert abs(r2["ego"]["ego_states"][0][0] - 1.23) < 1e-9, \
        f"ego_states[0][0] = {r2['ego']['ego_states'][0][0]} (expect 1.23 from s0)"

    print("✅ test_trajectory_nearest passed")


def test_non_lerp_fields_preserved():
    """不内差字段保留 s0（就近原则）"""
    s0 = make_dummy_frame(0, 100.0, 0, 0, 45.0, 10.0)
    s1 = make_dummy_frame(1, 100.2, 2, 2, 47.0, 12.0)

    r = lerp_snapshots(s0, s1, 0.3)

    # turn_signal、decision 不应该变
    assert r["ego"]["turn_signal"] == s0["ego"]["turn_signal"]
    assert r["ego"]["decision"] == s0["ego"]["decision"]

    # map 不应内差
    assert r["map"] == s0["map"]
    assert r["algo_env"]["sync_utc"] == s0["algo_env"]["sync_utc"]

    # sensor_cfg 不应内差
    assert r["sensor_cfg"] == s0["sensor_cfg"]

    print("✅ test_non_lerp_fields_preserved passed")


if __name__ == "__main__":
    test_lerp_basic()
    test_lerp_angle()
    test_agents_by_track_id()
    test_trajectory_nearest()
    test_non_lerp_fields_preserved()
    print("\n🎉 All tests passed!")
