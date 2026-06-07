"""
Lance 帧序列降采样工具

从原始 Lance 帧列表中按倍速 alpha 抽取子集，重新编号 frameindex，
并清理帧号相关的元信息（meta_decision / last_meta_decision）。

alpha 含义与 time_scale 统一：alpha = 倍速因子，仅接受正整数。
  - alpha = 2 → 2x 快放，每 2 帧取 1
  - alpha = 3 → 3x 快放，每 3 帧取 1
  - alpha = 1 → 不变

使用方法：
    from down_sample import downsample_frames

    frames = read_lance_frames("path/to/lance")
    sampled = downsample_frames(frames, alpha=3)  # 3x 快放
"""

import copy
from typing import Any, Dict, List


def downsample_frames(
    frames: List[Dict[str, Any]],
    alpha: int,
) -> List[Dict[str, Any]]:
    """
    从 frames 列表中按倍速 alpha 降采样。

    Args:
        frames: 原始帧列表（按时间升序）
        alpha:  快放倍速，正整数。
                 alpha=2 → 2x 快放（每 2 帧取 1）
                 alpha=1 → 返回原列表（deepcopy）

    Returns:
        降采样后的帧列表（deepcopy，不修改入参）

    修改内容：
        - frameindex: 重新编号为 0, 1, 2 ...
        - meta_decision: 移除（降采样后帧号不再对应原始序列）
        - last_meta_decision: 移除

    其他字段不变：
        - timestamp 不变（物理信息不变，只是观看速度加快）
        - ego / agents / map / sensor 等所有字段原样保留
    """
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
