"""
测试 down_sample.py
"""
import copy
from down_sample import downsample_frames


def make_frame(fi: int, ts: float) -> dict:
    return {
        "dat_name": "test",
        "timestamp": ts,
        "frameindex": fi,
        "ego": {"vel": 10.0 + fi},
        "agents": [],
        "meta_decision": [[f"label_{fi}", 0, 10]],
        "last_meta_decision": [[f"label_{fi}", 5, 20]],
    }


def test_alpha_1():
    """alpha=1 → deepcopy 但数据不变"""
    src = [make_frame(i, 100.0 + i * 0.1) for i in range(3)]
    out = downsample_frames(src, alpha=1)

    assert len(out) == 3
    assert out[0]["frameindex"] == 0
    assert out[0]["timestamp"] == 100.0
    assert "meta_decision" in out[0]
    assert out[0] is not src[0]

    print("✅ test_alpha_1 passed")


def test_alpha_3():
    """alpha=3 每 3 帧取 1"""
    src = [make_frame(i, 100.0 + i * 0.1) for i in range(10)]
    out = downsample_frames(src, alpha=3)

    # 10 帧 → 帧号 0, 3, 6, 9 → 4 帧
    assert len(out) == 4, f"expected 4, got {len(out)}"

    # frameindex 重新编号 0,1,2,3
    for i, f in enumerate(out):
        assert f["frameindex"] == i, f"frame[{i}] index = {f['frameindex']}"

    # meta_decision / last_meta_decision 全部移除
    for f in out:
        assert "meta_decision" not in f, f"meta_decision not removed from frame {f['frameindex']}"
        assert "last_meta_decision" not in f, f"last_meta_decision not removed"

    # timestamp 不变
    assert out[0]["timestamp"] == 100.0
    assert out[1]["timestamp"] == 100.3
    assert out[2]["timestamp"] == 100.6

    # 原帧不受影响
    assert "meta_decision" in src[0]

    print("✅ test_alpha_3 passed")


def test_alpha_big():
    """alpha > len(frames) → 只取第 0 帧"""
    src = [make_frame(i, 100.0 + i * 0.1) for i in range(5)]
    out = downsample_frames(src, alpha=99)
    assert len(out) == 1
    assert out[0]["frameindex"] == 0
    assert out[0]["timestamp"] == 100.0
    print("✅ test_alpha_big passed")


def test_non_meta_frames():
    """没有 meta_decision 的帧也能正常处理"""
    src = [{"frameindex": i, "ego": {}, "agents": [], "dat_name": "t"} for i in range(5)]
    out = downsample_frames(src, alpha=2)
    assert len(out) == 3  # 0, 2, 4
    for i, f in enumerate(out):
        assert f["frameindex"] == i
    print("✅ test_non_meta_frames passed")


def test_alpha_must_be_int():
    """非整数 alpha 应报错"""
    src = [make_frame(0, 100.0)]
    try:
        downsample_frames(src, alpha=2.5)
        assert False, "should have raised"
    except ValueError:
        pass
    print("✅ test_alpha_must_be_int passed")


if __name__ == "__main__":
    test_alpha_1()
    test_alpha_3()
    test_alpha_big()
    test_non_meta_frames()
    test_alpha_must_be_int()
    print("\n🎉 All tests passed!")
