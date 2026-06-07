# model_training — Lance 数据倍速工具

## 用法

```bash
python speedup.py <lance_path> -a <alpha> [-o <output>] [--dry-run]
```

参数：

| 参数 | 说明 |
|------|------|
| `lance_path` | Lance 数据目录（read_lance_frames 的输入路径） |
| `-a` / `--alpha` | 倍速因子，必须 >= 1.0 |
| `-o` / `--output_path` | 输出路径（默认: `<input>_<alpha>x.lance`） |
| `--dry-run` | 只看概要不写文件 |

示例：

```bash
# 3x 整数倍速（直接抽帧）
python speedup.py /data/raw_lance -a 3

# 2.5x 小数倍速（lerp 内差，会提示风险）
python speedup.py /data/raw_lance -a 2.5 -o /data/out.lance

# 先预览输出结果
python speedup.py /data/raw_lance -a 3 --dry-run
```

## 依赖

- **Python >= 3.10**（type hint 用 `float | None` 语法）
- **lancedb** — Lance 列式存储读写（内部依赖 `lance-format` C 扩展，而非 pypi 上假的 `lance` 包）
- **pyarrow >= 14** — Arrow 列式内存格式

安装：

```bash
pip install 'pyarrow>=14' lancedb
```

> ℹ️ 如果遇到 `externally-managed-environment`，加 `--break-system-packages`。

---

## 模块说明

```
lance_utils.py   Lance 读写（read / write）—— 最底层 I/O
down_sample.py   降采样（整数抽帧 + 小数 lerp 内差）—— frame 数减少
time_scale.py    时间轴 + 物理量缩放 —— 保留帧数不变，时间戳/速度改变
speedup.py       编排脚本：先降采样 → 再时间缩放
```

---

## ⚠️ 已知坑点与设计决策

### 1. `meta_decision` / `last_meta_decision` 被移除

降采样后帧号重编号，原始 `meta_decision` 中的帧引用会指向错误的位置。与其保留一个注定错误的决策字段，不如直接移除。如果下游流程需要决策信息，需重新计算。

### 2. 非整数 alpha 时图片不做内差

帧间的 sensor bytes（图片、点云等）`lerp_snapshots` 不做插值，而是就近取近帧。对于增广（augmentation）场景此行为是安全的——生成训练数据时更关心语义连续性而非传感器数据的精确插值。如果下游需要 sensor 时序插值，需单独处理。

### 3. 帧内轨迹数组不缩放、不内差

`ego_states`（历史+未来121帧）、`agent_state`、`targets_states` 等轨迹数组在 `lerp_snapshots` 中就近取近帧，在 `stretch_timestamps` 中不做时间戳调整。这意味着这些内部的 timestep 在倍速后与全局时间轴不同步。但对于模型训练（QCNet 等）它们通常仅用于 visual 输入或注意力机制，节奏变化的影响有限。

### 4. 三套时间各自独立零点

全局 `timestamp`、`algo_env.sync_utc`、`sensor.*.timestamp_tsn` 三者各自以初始帧为基准缩放——因为它们有各自独立的零点和物理含义（UTC、系统时间、硬件时钟），不应混用一个偏移。

### 5. 仅单帧出现的 agent 不保留

`lerp_snapshots` 中，只出现在两帧中某一帧的 agent（track_id 不匹配）会被丢弃。这是设计选择：内差无法产生合理的中间态，保留反而会引入伪影。

### 6. `write_lance_frames` 使用 `mode="overwrite"`

每次写入都是全量替换，不会追加到已有 dataset。行为确定，可重复执行。
