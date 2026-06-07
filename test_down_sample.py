"""
测试 down_sample.py
"""
import copy
import math
from down_sample import downsample_frames, fractional_downsample


def make_frame(fi: int, ts: float) -> dict:
    return {
        "dat_name": "test",
        "timestamp": ts,
        "frameindex": fi,
        "ego": {"vel": 10.0 + fi * 0.5, "acc": (0.0, 0.0), "pos_enu": [fi * 2.0, 0.0]},
        "agents": [],
        "meta_decision": [[f"label_{fi}", 0, 10]],
        "last_meta_decision": [[f"label_{fi}", 5, 20]],
    }


# ===================================================================
# 整数版
# ===================================================================

def test_alpha_1():
    src = [make_frame(i, 100.0 + i * 0.1) for i in range(3)]
    out = downsample_frames(src, alpha=1)
    assert len(out) == 3
    assert out[0]["frameindex"] == 0
    assert "meta_decision" in out[0]
    assert out[0] is not src[0]
    print("✅ test_alpha_1 passed")


def test_alpha_3():
    src = [make_frame(i, 100.0 + i * 0.1) for i in range(10)]
    out = downsample_frames(src, alpha=3)
    assert len(out) == 4
    for i, f in enumerate(out):
        assert f["frameindex"] == i
        assert "meta_decision" not in f
    assert out[1]["timestamp"] == 100.3
    print("✅ test_alpha_3 passed")


def test_alpha_big():
    src = [make_frame(i, 100.0 + i * 0.1) for i in range(5)]
    out = downsample_frames(src, alpha=99)
    assert len(out) == 1
    assert out[0]["timestamp"] == 100.0
    print("✅ test_alpha_big passed")


def test_non_meta_frames():
    src = [{"frameindex": i, "ego": {}, "agents": [], "dat_name": "t"} for i in range(5)]
    out = downsample_frames(src, alpha=2)
    assert len(out) == 3
    for i, f in enumerate(out):
        assert f["frameindex"] == i
    print("✅ test_non_meta_frames passed")


def test_float_alpha_accepted():
    """float 整数 alpha 被自动 int 化，不报错"""
    src = [make_frame(i, 100.0 + i * 0.1) for i in range(10)]
    out = downsample_frames(src, alpha=3.0)
    assert len(out) == 4
    for i in range(4):
        assert out[i]["frameindex"] == i
    assert out[1]["timestamp"] == 100.3
    print("✅ test_float_alpha_accepted passed")


# ===================================================================
# 小数版
# ===================================================================

def make_uniform_sequence(n, dt=0.1, start_ts=1000.0):
    """生成均匀帧序列"""
    return [make_frame(i, start_ts + i * dt) for i in range(n)]


def test_fractional_alpha_1():
    """alpha=1 → 不变"""
    src = make_uniform_sequence(5)
    out = fractional_downsample(src, alpha=1.0)
    assert len(out) == 5
    for i in range(5):
        assert out[i]["timestamp"] == src[i]["timestamp"]
        assert out[i]["frameindex"] == i
    print("✅ test_fractional_alpha_1 passed")


def test_fractional_alpha_2():
    """alpha=2 → 等价于整数版 alpha=2"""
    src = make_uniform_sequence(10, dt=0.1)
    out = fractional_downsample(src, alpha=2.0)
    # 帧间隔从 0.1 → 0.2
    # 目标时间点: 1000, 1000.2, 1000.4, ...
    assert len(out) == 5  # (0.9 / 0.2) + 1 = 5
    for i, f in enumerate(out):
        expected_ts = 1000.0 + i * 0.2
        assert abs(f["timestamp"] - expected_ts) < 1e-9, \
            f"frame[{i}] ts = {f['timestamp']}, expected {expected_ts}"
        assert f["frameindex"] == i
    print("✅ test_fractional_alpha_2 passed")


def test_fractional_alpha_25():
    """alpha=2.5 → 10fps→4fps"""
    src = make_uniform_sequence(10, dt=0.1)
    out = fractional_downsample(src, alpha=2.5)
    # new_dt = 0.1 * 2.5 = 0.25
    # 时间点: 1000, 1000.25, 1000.5, 1000.75 → 4 帧
    assert len(out) == 4, f"expected 4, got {len(out)}"
    for i, f in enumerate(out):
        expected_ts = 1000.0 + i * 0.25
        assert abs(f["timestamp"] - expected_ts) < 1e-9
        assert f["frameindex"] == i
    print("✅ test_fractional_alpha_25 passed")


def test_fractional_lerp_values():
    """验证 lerp 内差了物理量"""
    n = 5
    dt = 0.1
    src = make_uniform_sequence(n, dt=dt)

    # alpha=2.5, new_dt=0.25
    # 第 1 帧 ts=1000.25：在原始第 2 帧(1000.2) 和 第 3 帧(1000.3) 之间
    # t_ratio = (1000.25 - 1000.2) / 0.1 = 0.5
    out = fractional_downsample(src, alpha=2.5)
    assert len(out) >= 2, f"need at least 2 frames, got {len(out)}"

    # 第 1 帧：vel 应该 lerp 了 src[2].vel=11.0 和 src[3].vel=11.5 → 11.25
    f1 = out[1]
    expected_vel = 11.0 + (11.5 - 11.0) * 0.5
    assert abs(f1["ego"]["vel"] - expected_vel) < 1e-9, \
        f"f1 vel = {f1['ego']['vel']}, expected {expected_vel}"

    # pos_enu.x 应该 lerp 了 src[2].pos_enu[0]=4.0 和 src[3].pos_enu[0]=6.0 → 5.0
    expected_pos = 4.0 + (6.0 - 4.0) * 0.5
    assert abs(f1["ego"]["pos_enu"][0] - expected_pos) < 1e-9, \
        f"f1 pos.x = {f1['ego']['pos_enu'][0]}, expected {expected_pos}"

    print("✅ test_fractional_lerp_values passed")


def test_fractional_single_frame():
    """单帧 → 直接返回"""
    src = [make_frame(0, 1000.0)]
    out = fractional_downsample(src, alpha=2.0)
    assert len(out) == 1
    assert out[0]["timestamp"] == 1000.0
    print("✅ test_fractional_single_frame passed")


def test_fractional_aligns_with_int():
    """alpha=2 时小数版和整数版帧数一致"""
    for n_frames in [3, 5, 10, 21]:
        src = make_uniform_sequence(n_frames, dt=0.1)
        int_out = downsample_frames(src, alpha=2)
        f_out = fractional_downsample(src, alpha=2.0)
        assert len(int_out) == len(f_out), \
            f"{n_frames} frames: int={len(int_out)}, frac={len(f_out)}"
        for i in range(len(int_out)):
            assert int_out[i]["frameindex"] == f_out[i]["frameindex"]
    print("✅ test_fractional_aligns_with_int passed")


def test_fractional_large_alpha():
    """alpha > len(frames) → 只返回第一帧"""
    src = make_uniform_sequence(5)
    out = fractional_downsample(src, alpha=100.0)
    assert len(out) == 1
    print("✅ test_fractional_large_alpha passed")


def test_empty_frames():
    """空帧列表 → 空列表"""
    assert fractional_downsample([], alpha=2.0) == []
    print("✅ test_empty_frames passed")


def test_fractional_alpha_too_small():
    """alpha < 1 应报错"""
    try:
        fractional_downsample(make_uniform_sequence(5), alpha=0.5)
        assert False
    except ValueError:
        pass
    print("✅ test_fractional_alpha_too_small passed")


def test_fractional_meta_removed():
    """meta_decision 被移除"""
    src = make_uniform_sequence(5)
    out = fractional_downsample(src, alpha=2.0)
    for f in out:
        assert "meta_decision" not in f
        assert "last_meta_decision" not in f
    print("✅ test_fractional_meta_removed passed")


if __name__ == "__main__":
    # 整数版
    test_alpha_1()
    test_alpha_3()
    test_alpha_big()
    test_non_meta_frames()
    test_float_alpha_accepted()

    # 小数版
    test_fractional_alpha_1()
    test_fractional_alpha_2()
    test_fractional_alpha_25()
    test_fractional_lerp_values()
    test_fractional_single_frame()
    test_fractional_aligns_with_int()
    test_fractional_large_alpha()
    test_empty_frames()
    test_fractional_alpha_too_small()
    test_fractional_meta_removed()
    print("\n🎉 All tests passed!")
