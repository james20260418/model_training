"""
测试 time_scale.py
"""
import math
import copy
import sys
sys.path.insert(0, "/james_pm/model_training")
from time_scale import (
    stretch_timestamps,
    scale_physics,
    time_scale_frames,
)


def make_frame(fi: int, ts: float, **ego_kw) -> dict:
    ego_default = {
        "vel": 10.0 + fi,
        "acc": (1.0, 0.1),
        "yaw_rate": 0.02,
        "heading_angle_global": 45.0 + fi,
        "pos_enu": [fi * 2.0, fi * 1.5],
        "steering_angle": 0.5,
        "turn_signal": 0,
        "decision": "cruise",
        "rainfall_level": 0.0,
        "ego_states": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 1, 1.8, 4.9, 1.8, 0.0, 0.0, ts],
            [0.1, 0.1, 0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 1, 1.8, 4.9, 1.8, 0.0, 0.0, ts + 0.1],
        ],
    }
    ego_default.update(ego_kw)
    return {
        "dat_name": "test",
        "timestamp": ts,
        "frameindex": fi,
        "ego": ego_default,
        "agents": [
            {"track_id": "a_1", "vel": (5.0 + fi, 0.0),
             "acc": (0.5 + fi * 0.1, 0.0),
             "yaw_rate": 0.01, "pos_enu": (15.0 + fi, 3.0),
             "heading_angle_global": 90.0, "width": 1.8, "lenght": 4.5,
             "type": 12, "confidence": 98.0, "height": 1.6,
             "agent_state": [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 12, 1.8, 4.5, 1.6, 98.0, ts],
             ]},
        ],
        "sensor": {
            "camera": {"cam_front": {"timestamp_tsn": 100.0 + fi * 10.0,
                                      "data_path": "/img.jpg"}},
            "lidar": {"lidar_main": {"timestamp_tsn": 200.0 + fi * 10.0}},
        },
        "algo_env": {"sync_utc": 1710000000.0 + fi * 0.1, "lanes": []},
        "targets_states": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 12, 1.8, 4.5, 1.6, ts],
        ],
        "meta_decision": [["label", 0, 10]],
    }


def make_sequence(n, dt=0.1, start_ts=1000.0) -> list:
    return [make_frame(i, start_ts + i * dt) for i in range(n)]


# ===================================================================
# scale_physics tests
# ===================================================================

def test_scale_physics_alpha1():
    src = make_sequence(3)
    out = scale_physics(src, alpha=1.0)
    assert len(out) == 3
    for i in range(3):
        assert out[i]["ego"]["vel"] == src[i]["ego"]["vel"]
        assert out[i]["ego"]["acc"] == src[i]["ego"]["acc"]
    print("✅ test_scale_physics_alpha1 passed")


def test_scale_physics_2x():
    """2x 快放 (alpha=0.5) → vel × 2, acc × 4"""
    src = make_sequence(3)
    out = scale_physics(src, alpha=0.5)
    for i in range(3):
        assert abs(out[i]["ego"]["vel"] - src[i]["ego"]["vel"] * 2.0) < 1e-12
        assert abs(out[i]["ego"]["acc"][0] - src[i]["ego"]["acc"][0] * 4.0) < 1e-12
        assert abs(out[i]["ego"]["yaw_rate"] - src[i]["ego"]["yaw_rate"] * 2.0) < 1e-12
        # agent
        assert abs(out[i]["agents"][0]["vel"][0] - src[i]["agents"][0]["vel"][0] * 2.0) < 1e-12
        assert abs(out[i]["agents"][0]["acc"][0] - src[i]["agents"][0]["acc"][0] * 4.0) < 1e-12
        # 位置不变
        assert out[i]["ego"]["pos_enu"] == src[i]["ego"]["pos_enu"]
        # 离散量不变
        assert out[i]["ego"]["turn_signal"] == src[i]["ego"]["turn_signal"]
    print("✅ test_scale_physics_2x passed")


def test_scale_physics_single():
    """单帧也缩放物理量"""
    src = [make_frame(0, 1000.0)]
    out = scale_physics(src, alpha=0.5)
    assert abs(out[0]["ego"]["vel"] - src[0]["ego"]["vel"] * 2.0) < 1e-12
    print("✅ test_scale_physics_single passed")


def test_scale_physics_slow():
    """0.5x 慢放 (alpha=2.0) → vel × 0.5, acc × 0.25"""
    src = make_sequence(3)
    out = scale_physics(src, alpha=2.0)
    for i in range(3):
        assert abs(out[i]["ego"]["vel"] - src[i]["ego"]["vel"] * 0.5) < 1e-12
        assert abs(out[i]["ego"]["acc"][0] - src[i]["ego"]["acc"][0] * 0.25) < 1e-12
    print("✅ test_scale_physics_slow passed")


# ===================================================================
# stretch_timestamps tests
# ===================================================================

def test_stretch_timestamps_alpha1():
    src = make_sequence(3)
    out = stretch_timestamps(src, alpha=1.0)
    assert len(out) == 3
    for i in range(3):
        assert out[i]["timestamp"] == src[i]["timestamp"]
    print("✅ test_stretch_timestamps_alpha1 passed")


def test_stretch_timestamps_2x():
    """2x 快放 (alpha=0.5) → 全局偏移量缩半"""
    src = make_sequence(4, dt=0.2, start_ts=1000.0)
    out = stretch_timestamps(src, alpha=0.5)

    # 第一帧不变
    assert out[0]["timestamp"] == 1000.0

    # 后续帧 offset 缩半
    for i in range(1, 4):
        orig_off = src[i]["timestamp"] - 1000.0
        expected = 1000.0 + orig_off * 0.5
        assert abs(out[i]["timestamp"] - expected) < 1e-12

    print("✅ test_stretch_timestamps_2x passed")


def test_stretch_timestamps_single():
    src = [make_frame(0, 1000.0)]
    out = stretch_timestamps(src, alpha=0.3)
    assert out[0]["timestamp"] == 1000.0
    print("✅ test_stretch_timestamps_single passed")


def test_stretch_timestamps_three_dims():
    """
    三个维度各自独立缩放，零点不同：
      timestamp=1000, utc=1710000000, tsn_cam=100, tsn_lid=200
    """
    src = make_sequence(3, dt=0.1, start_ts=1000.0)
    out = stretch_timestamps(src, alpha=0.5)

    # 维度 1: 全局 timestamp
    assert out[0]["timestamp"] == 1000.0
    assert abs(out[1]["timestamp"] - 1000.05) < 1e-12

    # 维度 2: sync_utc
    # 初帧 utc=1710000000, 第1帧 utc=1710000000.1
    # 缩放后: 1710000000 + 0.1*0.5 = 1710000000.05
    assert out[0]["algo_env"]["sync_utc"] == src[0]["algo_env"]["sync_utc"]
    utc_expected = 1710000000.0 + 0.1 * 0.5
    assert abs(out[1]["algo_env"]["sync_utc"] - utc_expected) < 1e-8, \
        f"utc[1] = {out[1]['algo_env']['sync_utc']}, expected {utc_expected}"

    # 维度 3a: camera timestamp_tsn
    # 初帧 tsn=100, 第1帧 tsn=110
    # 缩放后: 100 + 10*0.5 = 105
    assert out[0]["sensor"]["camera"]["cam_front"]["timestamp_tsn"] == 100.0
    assert abs(out[1]["sensor"]["camera"]["cam_front"]["timestamp_tsn"] - 105.0) < 1e-12, \
        f"cam_tsn[1] = {out[1]['sensor']['camera']['cam_front']['timestamp_tsn']}"

    # 维度 3b: lidar timestamp_tsn
    # 初帧 tsn=200, 第1帧 tsn=210, 缩放: 200 + 10*0.5 = 205
    assert out[0]["sensor"]["lidar"]["lidar_main"]["timestamp_tsn"] == 200.0
    assert abs(out[1]["sensor"]["lidar"]["lidar_main"]["timestamp_tsn"] - 205.0) < 1e-12

    print("✅ test_stretch_timestamps_three_dims passed")


def test_stretch_timestamps_trajectory_unchanged():
    """轨迹数组内时间戳不被修改"""
    src = make_sequence(3, dt=0.1, start_ts=1000.0)
    out = stretch_timestamps(src, alpha=0.5)

    # ego_states 内的 timestep 不变
    for i in range(3):
        for r in range(len(out[i]["ego"]["ego_states"])):
            assert out[i]["ego"]["ego_states"][r] == src[i]["ego"]["ego_states"][r], \
                f"ego_states[{i}][{r}] changed"

    # agent_state 不变
    for i in range(3):
        assert out[i]["agents"][0]["agent_state"] == src[i]["agents"][0]["agent_state"]

    # targets_states 不变
    for i in range(3):
        assert out[i]["targets_states"] == src[i]["targets_states"]

    print("✅ test_stretch_timestamps_trajectory_unchanged passed")


# ===================================================================
# time_scale_frames integration tests
# ===================================================================

def test_combined_2x():
    """2x 快放完整流程"""
    src = make_sequence(3, dt=0.1, start_ts=1000.0)
    out = time_scale_frames(src, alpha=0.5)

    # 时间戳
    assert out[0]["timestamp"] == 1000.0
    assert abs(out[1]["timestamp"] - 1000.05) < 1e-12

    # 物理量
    assert abs(out[1]["ego"]["vel"] - src[1]["ego"]["vel"] * 2.0) < 1e-12
    assert abs(out[1]["ego"]["acc"][0] - src[1]["ego"]["acc"][0] * 4.0) < 1e-12

    # 三维时间各自独立
    utc_expected = 1710000000.0 + 0.1 * 0.5
    assert abs(out[1]["algo_env"]["sync_utc"] - utc_expected) < 1e-8
    assert abs(out[1]["sensor"]["camera"]["cam_front"]["timestamp_tsn"] - 105.0) < 1e-12

    print("✅ test_combined_2x passed")


if __name__ == "__main__":
    test_scale_physics_alpha1()
    test_scale_physics_2x()
    test_scale_physics_single()
    test_scale_physics_slow()
    test_stretch_timestamps_alpha1()
    test_stretch_timestamps_2x()
    test_stretch_timestamps_single()
    test_stretch_timestamps_three_dims()
    test_stretch_timestamps_trajectory_unchanged()
    test_combined_2x()
    print("\n🎉 All tests passed!")
