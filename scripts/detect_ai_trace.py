from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from small_model_train.scoring import detect_ai_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", required=True)
    args = parser.parse_args()
    text = Path(args.text_file).read_text(encoding="utf-8")
    print(json.dumps(detect_ai_trace(text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
