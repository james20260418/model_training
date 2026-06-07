"""
Lance 帧序列降采样 + 帧间线性内差工具

整数版 + 小数版两种降采样，共用 alpha = 倍速因子语义。

整数版 downsample_frames(frames, alpha):
  - alpha 必须为正整数
  - 每 alpha 帧取 1，不做内差

小数版 fractional_downsample(frames, alpha):
  - alpha 为正实数（含整数）
  - 通过 lerp 在目标时间点内差出新帧
  - 图片等无法内差的字段就近取近帧

使用方法：
    from down_sample import downsample_frames, fractional_downsample

    sampled_int = downsample_frames(frames, alpha=3)
    sampled_frac = fractional_downsample(frames, alpha=2.5)
"""

import math
import copy
from typing import Any, Dict, List, Set, Tuple


# ===================================================================
# 帧间线性内差（从 lance_utils.py 移植）
# ===================================================================

# -- 哪些 ego 字段适合线性内差 ------------------------------------------------
_LERP_EGO_SCALAR = {
    "vel",              # 标量速度
    "yaw_rate",         # 偏航率
    "steering_angle",   # 方向盘角度
    "steering_angle_offset",
}
_LERP_EGO_VEC2 = {
    "acc",              # (x, y) 加速度
    "pos_utm",          # (x, y) UTM 坐标
    "pos_enu",          # (x, y) ENU 坐标
}
_LERP_EGO_ANGLE = {"heading_angle_global"}  # 角度（特殊处理）
_LERP_EGO_FLOAT = {"longitude", "latitude"} # 经纬度

# -- 哪些 agent 字段适合线性内差 ---------------------------------------------
_LERP_AGENT_SCALAR = {
    "yaw_rate",
    "confidence",
}
_LERP_AGENT_VEC2 = {
    "pos_utm",
    "pos_vcs",
    "pos_enu",
    "vel",
    "acc",
}
_LERP_AGENT_VEC3 = {"height"}
_LERP_AGENT_VEC2_FLOAT = {"width", "lenght"}  # 尺寸（近似常数）
_LERP_AGENT_ANGLE = {"heading_angle_global", "heading_angle_local"}


def _lerp_angle(a_deg: float, b_deg: float, t: float) -> float:
    """
    角度线性内差，取张角小的一侧。
    例如 350° → 10° 走 360°→10° 方向（只差 20°），不走 350°→10° 的 340° 绕路。
    """
    diff = (b_deg - a_deg) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return (a_deg + diff * t) % 360.0


def _lerp_num(a: float, b: float, t: float) -> float:
    """标量线性内差"""
    return a + (b - a) * t


def lerp_snapshots(s0: Dict[str, Any],
                   s1: Dict[str, Any],
                   t: float) -> Dict[str, Any]:
    """
    对两帧 Snapshot 做时间比率为 t ∈ [0, 1] 的内差。

    规则：
    - 可连续数值量 → 线性内差
    - 角度 → 取张角小的一侧内差
    - 轨迹数组（ego_states、agent_state*、targets_states*）→ 就近取
    - 地图 / algo_env / sd_map / 元信息 → 就近取
    - agent 障碍物通过 track_id 做对应，再逐 field 内差；
      只出现在单帧的 agent 不保留（移除）
    - sensor / 图片等 → 就近取（不做内差）

    Args:
        s0: 前一帧 Snapshot (dict)
        s1: 后一帧 Snapshot (dict)
        t:  时间比率 [0, 1]，0=s0 时刻，1=s1 时刻

    Returns:
        内差后的新 Snapshot dict（不会修改入参）
    """
    if t <= 0.0:
        return copy.deepcopy(s0)
    if t >= 1.0:
        return copy.deepcopy(s1)

    # 以就近帧为底板，在上面覆盖需要内差的字段
    result = copy.deepcopy(s0) if t < 0.5 else copy.deepcopy(s1)

    # ---- 顶层元信息 ----
    result["timestamp"] = _lerp_num(s0["timestamp"], s1["timestamp"], t)

    # ---- ego ----
    e0, e1 = s0.get("ego", {}), s1.get("ego", {})
    if e0 and e1:
        result["ego"] = _lerp_ego(e0, e1, t)

    # ---- agents（按 track_id 配对）----
    agents0 = {a["track_id"]: a for a in s0.get("agents", [])}
    agents1 = {a["track_id"]: a for a in s1.get("agents", [])}
    common_ids: Set[str] = set(agents0.keys()) & set(agents1.keys())
    result["agents"] = []
    for tid in common_ids:
        result["agents"].append(_lerp_agent(agents0[tid], agents1[tid], t))

    return result


def _lerp_ego(e0: Dict[str, Any], e1: Dict[str, Any], t: float) -> Dict[str, Any]:
    """ego 层内差"""
    out = copy.deepcopy(e0)

    for k in _LERP_EGO_SCALAR:
        if k in e1:
            out[k] = _lerp_num(e0[k], e1[k], t)

    for k in _LERP_EGO_VEC2:
        if k in e1:
            out[k] = (_lerp_num(e0[k][0], e1[k][0], t),
                      _lerp_num(e0[k][1], e1[k][1], t))

    for k in _LERP_EGO_ANGLE:
        if k in e1:
            out[k] = _lerp_angle(e0[k], e1[k], t)

    for k in _LERP_EGO_FLOAT:
        if k in e1:
            out[k] = _lerp_num(e0[k], e1[k], t)

    # 轨迹类字段就近取
    for traj_key in ("ego_states", "ego_states_from_odom", "ego_pose_odo_future"):
        src = e1 if t >= 0.5 else e0
        if traj_key in src:
            out[traj_key] = copy.deepcopy(src[traj_key])

    return out


def _lerp_agent(a0: Dict[str, Any], a1: Dict[str, Any], t: float) -> Dict[str, Any]:
    """agent 层内差"""
    out = copy.deepcopy(a0)
    out["track_id"] = a0["track_id"]

    for k in _LERP_AGENT_SCALAR:
        if k in a1:
            out[k] = _lerp_num(a0[k], a1[k], t)

    for k in _LERP_AGENT_VEC2:
        if k in a1:
            out[k] = (_lerp_num(a0[k][0], a1[k][0], t),
                      _lerp_num(a0[k][1], a1[k][1], t))

    for k in _LERP_AGENT_VEC3:
        if k in a1:
            out[k] = _lerp_num(a0[k], a1[k], t)

    for k in _LERP_AGENT_VEC2_FLOAT:
        if k in a1:
            out[k] = _lerp_num(a0[k], a1[k], t)

    for k in _LERP_AGENT_ANGLE:
        if k in a1:
            out[k] = _lerp_angle(a0[k], a1[k], t)

    # 轨迹类字段就近取
    for traj_key in ("agent_state", "agent_state_from_odom",
                     "agent_state_from_odom_refined",
                     "traj_valid", "traj_quality", "lane_info"):
        src = a1 if t >= 0.5 else a0
        if traj_key in src:
            out[traj_key] = copy.deepcopy(src[traj_key])

    return out


# ===================================================================
# 整数版降采样
# ===================================================================

def downsample_frames(
    frames: List[Dict[str, Any]],
    alpha: int,
) -> List[Dict[str, Any]]:
    """
    整数倍率降采样：每 alpha 帧取 1，不做内差。

    Args:
        frames: 原始帧列表（按时间升序）
        alpha:  快放倍速，正整数。

    Returns:
        降采样后的帧列表（deepcopy，不修改入参）

    修改内容：
        - frameindex: 重新编号为 0, 1, 2 ...
        - meta_decision / last_meta_decision: 移除
        - 其他字段不变（含 timestamp）
    """
    if isinstance(alpha, float):
        alpha = int(alpha)
    if not isinstance(alpha, int) or alpha < 1:
        raise ValueError(f"alpha must be a positive integer >= 1, got {alpha}")

    if alpha == 1:
        return copy.deepcopy(frames)

    sampled: List[Dict[str, Any]] = []
    for new_idx, old_idx in enumerate(range(0, len(frames), alpha)):
        frame = copy.deepcopy(frames[old_idx])
        frame["frameindex"] = new_idx
        frame.pop("meta_decision", None)
        frame.pop("last_meta_decision", None)
        sampled.append(frame)

    return sampled


# ===================================================================
# 小数版降采样（基于 lerp 重采样）
# ===================================================================

def fractional_downsample(
    frames: List[Dict[str, Any]],
    alpha: float,
) -> List[Dict[str, Any]]:
    """
    小数倍率降采样：通过 lerp 内差在目标时间点生成新帧。

    流程：
      1. 计算原始平均帧间隔 orig_dt
      2. 目标间隔 target_dt = orig_dt * alpha
      3. 从 t0 开始每 target_dt 生成目标时间点
      4. 对每个时间点找前后帧，调 lerp_snapshots 内差
      5. frameindex 重编号，meta_decision 移除

    Args:
        frames: 原始帧列表（按时间升序，至少 2 帧才能内差）
        alpha:  快放倍速，正实数 >= 1。
                 alpha=2.5 → 从 10fps 降到 4fps

    Returns:
        降采样后的帧列表（deepcopy，不修改入参）
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if alpha < 1.0:
        raise ValueError(f"alpha must be >= 1 for downsampling, got {alpha}")
    if not frames:
        return []
    if len(frames) == 1 or alpha == 1.0:
        return copy.deepcopy(frames)

    orig_dt = _avg_dt(frames)
    target_dt = orig_dt * alpha
    t0 = frames[0]["timestamp"]
    t_end = frames[-1]["timestamp"]

    # 目标帧数 = ceil(len(frames) / alpha)，与整数版帧数一致
    # 例：3 帧 alpha=2 → ceil(1.5)=2 → frame 0 和 frame 2
    n_expected = math.ceil(len(frames) / alpha)
    if n_expected <= 1:
        return copy.deepcopy([frames[0]])

    # 生成 n_expected 个目标时间点，均匀覆盖 [t0, t_end]
    target_times = [t0 + i * target_dt for i in range(n_expected)]
    # 最后一帧不超出 t_end
    target_times = [min(t, t_end) for t in target_times]

    result: List[Dict[str, Any]] = []
    for new_idx, target_ts in enumerate(target_times):
        prev_idx, next_idx, t_ratio = _find_segment(frames, target_ts)

        if prev_idx == next_idx or t_ratio is None:
            frame = copy.deepcopy(frames[prev_idx])
        else:
            frame = lerp_snapshots(frames[prev_idx], frames[next_idx], t_ratio)

        frame["timestamp"] = target_ts
        frame["frameindex"] = new_idx
        frame.pop("meta_decision", None)
        frame.pop("last_meta_decision", None)
        result.append(frame)

    return result


def _avg_dt(frames: List[Dict[str, Any]]) -> float:
    """原始帧序列的平均帧间隔"""
    total = frames[-1]["timestamp"] - frames[0]["timestamp"]
    return total / (len(frames) - 1)


def _find_segment(
    frames: List[Dict[str, Any]],
    target_ts: float,
) -> Tuple[int, int, float | None]:
    """
    找到 target_ts 落在哪两帧之间。

    Returns:
        (prev_idx, next_idx, t_ratio)
        - t_ratio ∈ [0, 1]
        - 目标正好落在帧上时 prev_idx==next_idx, t_ratio=None
    """
    n = len(frames)
    if target_ts <= frames[0]["timestamp"] + 1e-9:
        return (0, 0, None)
    if target_ts >= frames[-1]["timestamp"] - 1e-9:
        return (n - 1, n - 1, None)

    # 二分查找
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if frames[mid]["timestamp"] <= target_ts:
            lo = mid
        else:
            hi = mid - 1

    prev_idx = lo
    next_idx = lo + 1 if lo < n - 1 else lo

    if prev_idx == next_idx:
        return (prev_idx, next_idx, None)

    seg_start = frames[prev_idx]["timestamp"]
    seg_end = frames[next_idx]["timestamp"]
    dt = seg_end - seg_start
    if dt <= 1e-12:
        return (prev_idx, prev_idx, None)

    t_ratio = (target_ts - seg_start) / dt
    t_ratio = max(0.0, min(1.0, t_ratio))

    return (prev_idx, next_idx, t_ratio)
