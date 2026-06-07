"""
Lance 数据处理工具

读写 Lance 列式存储文件 + 帧间线性内差。

使用方式：
    pip install lancedb pyarrow>=14

    from lance_utils import read_lance_frames, write_lance_frames, lerp_snapshots
"""

import math
import pickle
import copy
import pyarrow as pa
import lancedb
from typing import Any, Dict, List, Optional, Set
from pathlib import Path


_DEFAULT_TABLE = "frames"
_DEFAULT_COLUMN = "src_label_stag_1"


# ---------------------------------------------------------------------------
# 读写 Lance
# ---------------------------------------------------------------------------

def read_lance_frames(
    lance_path: str,
    table_name: str = _DEFAULT_TABLE,
    column: str = _DEFAULT_COLUMN,
) -> List[Dict[str, Any]]:
    """从 LanceDB 中读取所有帧的 dict 数据。"""
    db = lancedb.connect(lance_path)
    tbl = db.open_table(table_name)
    arrow_table = tbl.to_arrow()
    raw_bytes: List[bytes] = arrow_table.column(column).to_pylist()
    return [pickle.loads(b) for b in raw_bytes]


def write_lance_frames(
    lance_path: str,
    frames: List[Dict[str, Any]],
    table_name: str = _DEFAULT_TABLE,
    column: str = _DEFAULT_COLUMN,
    mode: str = "overwrite",
) -> str:
    """将帧 dict 列表写入 LanceDB。"""
    pickled = pa.array([pickle.dumps(f) for f in frames], type=pa.binary())
    schema = pa.schema([pa.field(column, pa.binary())])
    tbl = pa.table({column: pickled}, schema=schema)
    db = lancedb.connect(lance_path)
    db.create_table(table_name, tbl, mode=mode)
    return lance_path


# ---------------------------------------------------------------------------
# 内差核心函数
# ---------------------------------------------------------------------------

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
_LERP_AGENT_VEC2_FLOAT = {"width", "lenght"}  # 尺寸（常数）
_LERP_AGENT_ANGLE = {"heading_angle_global", "heading_angle_local"}

# -- 精确切分角度 -----------------------------------------------------------------


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


def _get_closest_snapshot(s0: Dict[str, Any],
                          s1: Dict[str, Any],
                          t: float) -> Dict[str, Any]:
    """返回离 query 时间最近的原始帧（用 deepcopy 防止污染）"""
    if t <= 0.5:
        return copy.deepcopy(s0)
    return copy.deepcopy(s1)


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------

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
    - agent 障碍物通过 track_id 做对应，再逐 field 内差；只在 s0 或 s1 中的一个 agent，直接取

    Args:
        s0: 前一帧 Snapshot (dict)
        s1: 后一帧 Snapshot (dict)
        t:  时间比率 [0, 1]，0=s0 时刻，1=s1 时刻

    Returns:
        内差后的新 Snapshot dict（不会修改入参）
    """
    # t = 0 / 1 → 直接返回最近帧
    if t <= 0.0:
        return copy.deepcopy(s0)
    if t >= 1.0:
        return copy.deepcopy(s1)

    # t ∈ (0,1)：以就近帧为底板，然后在上面覆盖需要内差的字段
    # t < 0.5 用 s0，t >= 0.5 用 s1
    result = copy.deepcopy(s0) if t < 0.5 else copy.deepcopy(s1)

    # ---- 顶层元信息 ----
    result["timestamp"] = _lerp_num(s0["timestamp"], s1["timestamp"], t)

    # ---- ego ----
    e0, e1 = s0.get("ego", {}), s1.get("ego", {})
    if e0 and e1:
        result["ego"] = _lerp_ego(e0, e1, t)

    # ---- agents（按 track_id 配对）----
    # 只在两帧都出现的 agent 才保留并内差；只出现在单帧的 agent 移除
    # （agent 是连续运动对象，落单的帧代表另一端不存在，内差位置无意义）
    agents0 = {a["track_id"]: a for a in s0.get("agents", [])}
    agents1 = {a["track_id"]: a for a in s1.get("agents", [])}
    common_ids: Set[str] = set(agents0.keys()) & set(agents1.keys())
    result["agents"] = []
    for tid in common_ids:
        result["agents"].append(_lerp_agent(agents0[tid], agents1[tid], t))

    return result


# ---------------------------------------------------------------------------
# ego 内差
# ---------------------------------------------------------------------------

def _lerp_ego(e0: Dict[str, Any], e1: Dict[str, Any], t: float) -> Dict[str, Any]:
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

    # ------------------------------------------------------------------
    # ego_states / ego_states_from_odom / ego_pose_odo_future → 就近取
    # 轨迹类字段在帧间不做逐元素内差，取离 t 最近的那一帧
    # ------------------------------------------------------------------
    for traj_key in ("ego_states", "ego_states_from_odom", "ego_pose_odo_future"):
        src = e1 if t >= 0.5 else e0
        if traj_key in src:
            out[traj_key] = copy.deepcopy(src[traj_key])

    return out


# ---------------------------------------------------------------------------
# agent 内差
# ---------------------------------------------------------------------------

def _lerp_agent(a0: Dict[str, Any], a1: Dict[str, Any], t: float) -> Dict[str, Any]:
    out = copy.deepcopy(a0)
    # track_id 不变
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

    # agent_state* / traj_valid / traj_quality / lane_info → 就近取
    for traj_key in ("agent_state", "agent_state_from_odom",
                     "agent_state_from_odom_refined",
                     "traj_valid", "traj_quality", "lane_info"):
        src = a1 if t >= 0.5 else a0
        if traj_key in src:
            out[traj_key] = copy.deepcopy(src[traj_key])

    return out
