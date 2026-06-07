#!/usr/bin/env python3
"""
speedup.py — Lance 数据倍速脚本

流程:
  1. 从 input_path 读取全部帧
  2. 按 alpha 降采样（整数 downsample_frames / 小数 fractional_downsample）
  3. 时间轴 + 物理量缩放（time_scale_frames）
  4. 写入 output_path

用法:
  python speedup.py /path/to/lance/ --alpha 3       # 3x 整数倍
  python speedup.py /path/to/lance/ -a 2.5 -o ./out # 2.5x 小数倍
  python speedup.py /path/to/lance/ -a 3 --dry-run  # 只看概要
"""

import argparse
import sys
import os
from typing import List, Dict, Any

from lance_utils import read_lance_frames, write_lance_frames
from down_sample import downsample_frames, fractional_downsample
from time_scale import time_scale_frames


def run(input_path: str, output_path: str, alpha: float, dry_run: bool) -> None:
    # ---------- read ----------
    print(f"📖 读取: {input_path}")
    frames = read_lance_frames(input_path)
    print(f"   共 {len(frames)} 帧")

    # ---------- downsample ----------
    if isinstance(alpha, float) and not alpha.is_integer():
        print(f"⚠️  非整数 alpha={alpha}，将用 lerp 内差降采样")
        print(f"   风险说明: 帧间的图片(sensor bytes)不做内差，就近取近帧")
        print(f"   帧内轨迹数组(ego_states/agent_state等)就近取，不做内差\n")
        sampled = fractional_downsample(frames, alpha)
    else:
        sampled = downsample_frames(frames, int(alpha))

    print(f"🔽 降采样后: {len(sampled)} 帧")

    # ---------- time scale ----------
    scaled = time_scale_frames(sampled, alpha)
    print(f"⏩ 倍速缩放: alpha={alpha}")

    # ---------- dry-run / write ----------
    if dry_run:
        print(f"\n📋 预览 (dry-run)")
        print(f"   输入:  {len(frames)} 帧")
        print(f"   输出:  {len(scaled)} 帧 ({alpha}x)")
        print(f"   写入:  {output_path}")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write_lance_frames(output_path, scaled)
    print(f"💾 写入: {output_path} ({len(scaled)} 帧)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lance 数据倍速脚本 — 先降采样再时间缩放",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  speedup.py ./raw_lance -a 3                  # 3x 整数倍速（直接抽帧）
  speedup.py ./raw_lance -a 2.5 -o ./out       # 2.5x 小数倍速（lerp 内差）
  speedup.py ./raw_lance -a 4 --dry-run        # 只看输出概要不写文件
        """,
    )
    parser.add_argument("input_path", help="Lance 数据路径（目录）")
    parser.add_argument("-a", "--alpha", type=float, required=True,
                        help="倍速因子，>= 1.0")
    parser.add_argument("-o", "--output_path", default=None,
                        help="输出路径 (默认: ./<input_dirname>_<alpha>x.lance)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印概要，不写入文件")

    args = parser.parse_args()

    if args.alpha < 1.0:
        print(f"❌ alpha 必须 >= 1.0，当前 {args.alpha}")
        sys.exit(1)

    if args.output_path is None:
        base_dir = os.path.dirname(args.input_path.rstrip("/"))
        base_name = os.path.basename(args.input_path.rstrip("/"))
        args.output_path = os.path.join(base_dir or ".", f"{base_name}_{args.alpha}x.lance")

    run(args.input_path, args.output_path, args.alpha, args.dry_run)


if __name__ == "__main__":
    main()
