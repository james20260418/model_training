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


