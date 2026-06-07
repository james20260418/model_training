"""
均匀时间缩放（Uniform Time Scaling）

将帧序列的时间轴压缩/拉伸 alpha 倍。

- alpha < 1 → 快放（时间收缩，物理量增速）
- alpha > 1 → 慢放（时间扩张，物理量减速）
- alpha = 1 → 不变（返回原序列 deepcopy）

两个独立子功能（均可单独调用）：
  stretch_timestamps(frames, alpha) — 全局时间戳缩放
    - 初始帧 timestamp / sync_utc / camera-tsn 保持不变
    - 后续帧的 timestamp / sync_utc / camera-tsn 各自以初始帧为基准缩放
    - 三套时间各自独立零点，互不影响
    - 帧内轨迹数组（ego_states, agent_state, targets_states 等）不做处理
    - 单帧返回 deepcopy 不变

  scale_physics(frames, alpha) — 每帧物理量缩放
    - 每帧独立缩放，与总帧数无关
    - vel / yaw_rate → 1/alpha 倍
    - acc → 1/alpha² 倍
    - 位置、角度、形状、离散量 → 不变
    - 单帧也正常缩放

物理量变换规则（对每个帧的 ego 和 agent）：
  不变（无时间维）：
    - 位置（pos_enu, pos_utm, pos_vcs）
    - 角度（heading_angle_global, heading_angle_local）
    - 形状尺寸（width, lenght, height, type）
    - 置信度（confidence）
    - 离散标量（turn_signal, decision）
    - 方向盘角度（steering_angle, steering_angle_offset）
  1/alpha 倍（速度类）：
    - vel, yaw_rate
  1/alpha² 倍（加速度类）：
    - acc

用法：
    from time_scale import stretch_timestamps, scale_physics, time_scale_frames
    scaled = time_scale_frames(frames, alpha=2.0)  # 2x 快放
"""

import copy
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# 子函数 A：时间戳缩放
# ---------------------------------------------------------------------------

def stretch_timestamps(
    frames: List[Dict[str, Any]],
    alpha: float,
) -> List[Dict[str, Any]]:
    """
    将帧序列的时间戳以初始帧为基准整体缩放。

    三个维度独立缩放，各自以初始帧对应值为零点：
      1. 全局 timestamp — 初始帧 timestamp 不动，后续帧偏移缩放
      2. algo_env.sync_utc — 初始帧值不动，后续帧偏移缩放
      3. sensor.*.timestamp_tsn — 初始帧值不动，后续帧偏移缩放

    帧内轨迹数组（ego_states, agent_state, targets_states,
    ego_pose_odo_future 等）不做处理。

    Args:
        frames: 原始帧列表（按时间升序）
        alpha:  时间缩放因子

    Returns:
        时间戳缩放后的帧列表（deepcopy）
    """
    if alpha == 1.0:
        return copy.deepcopy(frames)

    out = copy.deepcopy(frames)
    if len(out) <= 1:
        return out

    # 读取初始帧的三个维度的零点
    t0 = out[0]["timestamp"]
    utc0 = _get_utc0(out[0])
    tsn0_map = _get_tsn0(out[0])

    for i in range(1, len(out)):
        f = out[i]
        fi = frames[i]  # 原始帧（deepcopy 之前的值）

        # 1. 全局 timestamp
        orig_offset = fi["timestamp"] - t0
        f["timestamp"] = t0 + orig_offset * alpha

        # 2. algo_env.sync_utc
        _stretch_utc(f, fi, utc0, alpha)

        # 3. sensor.*.timestamp_tsn
        _stretch_tsn(f, fi, tsn0_map, alpha)

    return out


def _get_utc0(frame: Dict[str, Any]) -> float | None:
    """读取初始帧的 sync_utc 零点"""
    ae = frame.get("algo_env", {})
    if isinstance(ae, dict):
        val = ae.get("sync_utc")
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _get_tsn0(frame: Dict[str, Any]) -> Dict[str, float]:
    """读取初始帧各传感器（camera / lidar）的 timestamp_tsn 零点"""
    tsn0: Dict[str, float] = {}
    sensor = frame.get("sensor", {})
    for cam_or_lidar in ("camera", "lidar"):
        devs = sensor.get(cam_or_lidar, {})
        if not isinstance(devs, dict):
            continue
        for name, dev in devs.items():
            if isinstance(dev, dict):
                val = dev.get("timestamp_tsn")
                if isinstance(val, (int, float)):
                    tsn0[f"{cam_or_lidar}/{name}"] = float(val)
    return tsn0


def _stretch_utc(
    out_frame: Dict[str, Any],
    orig_frame: Dict[str, Any],
    utc0: float | None,
    alpha: float,
) -> None:
    """缩放 algo_env.sync_utc"""
    if utc0 is None:
        return
    ae = out_frame.get("algo_env", {})
    if not isinstance(ae, dict):
        return
    if "sync_utc" not in ae:
        return
    val = orig_frame.get("algo_env", {}).get("sync_utc")
    if isinstance(val, (int, float)):
        ae["sync_utc"] = utc0 + (val - utc0) * alpha


def _stretch_tsn(
    out_frame: Dict[str, Any],
    orig_frame: Dict[str, Any],
    tsn0_map: Dict[str, float],
    alpha: float,
) -> None:
    """缩放 sensor.*.timestamp_tsn"""
    if not tsn0_map:
        return

    def _stretch_device(name: str, dev: Dict[str, Any], orig_dev: Dict[str, Any]) -> None:
        t0 = tsn0_map.get(name)
        if t0 is None:
            return
        if not isinstance(dev, dict) or "timestamp_tsn" not in dev:
            return
        orig_val = orig_dev.get("timestamp_tsn") if orig_dev else None
        if isinstance(orig_val, (int, float)):
            dev["timestamp_tsn"] = t0 + (orig_val - t0) * alpha

    out_sensor = out_frame.get("sensor", {})
    orig_sensor = orig_frame.get("sensor", {})
    for cam_or_lidar in ("camera", "lidar"):
        out_devs = out_sensor.get(cam_or_lidar, {})
        orig_devs = orig_sensor.get(cam_or_lidar, {})
        if not isinstance(out_devs, dict) or not isinstance(orig_devs, dict):
            continue
        for name, out_dev in out_devs.items():
            _stretch_device(f"{cam_or_lidar}/{name}",
                            out_dev,
                            orig_devs.get(name))


# ---------------------------------------------------------------------------
# 子函数 B：物理量缩放
# ---------------------------------------------------------------------------

def scale_physics(
    frames: List[Dict[str, Any]],
    alpha: float,
) -> List[Dict[str, Any]]:
    """
    对帧序列中每帧的物理量做时间缩放。每帧独立，与总帧数无关。

    Args:
        frames: 原始帧列表
        alpha:  时间缩放因子

    Returns:
        物理量缩放后的帧列表（deepcopy）
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    if alpha == 1.0:
        return copy.deepcopy(frames)

    sv = 1.0 / alpha          # 速度缩放因子
    sa = 1.0 / (alpha * alpha)  # 加速度缩放因子

    return [_scale_frame_single(f, sv, sa) for f in frames]


def _scale_frame_single(frame: Dict[str, Any],
                        sv: float,
                        sa: float) -> Dict[str, Any]:
    """
    对单个 frame 的物理量做缩放，返回 deepcopy。

    sv = 1/alpha (速度缩放因子)
    sa = 1/alpha² (加速度缩放因子)
    """
    out = copy.deepcopy(frame)

    # ---- ego ----
    e = out.get("ego")
    if e:
        for k in ("vel", "yaw_rate"):
            if k in e:
                e[k] = e[k] * sv
        if "acc" in e:
            e["acc"] = (e["acc"][0] * sa, e["acc"][1] * sa)

    # ---- agents ----
    for a in out.get("agents", []):
        if "vel" in a:
            a["vel"] = (a["vel"][0] * sv, a["vel"][1] * sv)
        if "yaw_rate" in a:
            a["yaw_rate"] = a["yaw_rate"] * sv
        if "acc" in a:
            a["acc"] = (a["acc"][0] * sa, a["acc"][1] * sa)

    return out


# ---------------------------------------------------------------------------
# 组合入口
# ---------------------------------------------------------------------------

def time_scale_frames(
    frames: List[Dict[str, Any]],
    alpha: float,
) -> List[Dict[str, Any]]:
    """
    对帧序列做均匀时间缩放（等价于 scale_physics + stretch_timestamps）。

    Args:
        frames: 原始帧列表（按时间升序）
        alpha:  时间缩放因子

    Returns:
        缩放后的帧列表（deepcopy，不修改入参）
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    if alpha == 1.0:
        return copy.deepcopy(frames)

    # scale_physics 不碰任何 timestamp 字段，所以传给 stretch_timestamps
    # 时其帧的时间戳和原始帧一致，计算正确。
    out = scale_physics(frames, alpha)
    return stretch_timestamps(out, alpha)


__all__ = ["time_scale_frames", "stretch_timestamps", "scale_physics"]
