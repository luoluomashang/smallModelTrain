"""第一阶段管线的通用读写工具。

本项目的数据在各个步骤之间主要以 JSONL 和纯文本流转：
- JSONL 适合一行保存一个章节、一个评分、一个偏好样本，方便断点续跑。
- 纯文本原稿可能来自不同写作软件，所以读取时需要兼容常见中文编码。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


# 按“最常见且最安全”到“中文 Windows 老文件兼容”的顺序尝试。
ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


def read_text_auto(path: str | Path) -> str:
    """自动尝试常见编码读取小说原稿。

    utf-8-sig 可以去掉 UTF-8 BOM；gb18030 覆盖 GBK/GB2312 场景。
    如果所有候选编码都失败，最后抛出最接近真实原因的解码错误。
    """

    file_path = Path(path)
    last_error: UnicodeDecodeError | None = None
    for encoding in ENCODINGS:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return file_path.read_text()


def read_jsonl(path: str | Path) -> list[dict]:
    """读取 JSONL 文件。

    JSONL 是“一行一个 JSON 对象”。空行会被跳过；遇到坏行时把文件名和
    行号放进错误信息，方便定位是哪条中间数据损坏。
    """

    rows: list[dict] = []
    file_path = Path(path)
    if not file_path.exists():
        return rows
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_number} is not valid JSON") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """写出 JSONL，并自动创建父目录。

    ensure_ascii=False 保留中文原文，便于人工检查数据；newline="\n" 保证
    Windows 和 Unix 上产物一致。
    """

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
