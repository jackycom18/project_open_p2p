#!/usr/bin/env python
"""统计多个片段的按键分布与鼠标位移方向，辅助设计文本指令任务。"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402


def main():
    base = Path("/home/yue/open-p2p/dataset/dataset")
    for ep in sys.argv[1:]:
        p = base / ep
        anns = evaluate.get_annotations(evaluate.load_annotation(p), 0, 200)
        keys = Counter()
        left = sum(1 for a in anns if a["mouse_delta_x"] < 0)
        right = sum(1 for a in anns if a["mouse_delta_x"] > 0)
        zero = sum(1 for a in anns if a["mouse_delta_x"] == 0)
        has_left_key = sum(1 for a in anns if "LeftArrow" in a["keys"])
        has_right_key = sum(1 for a in anns if "RightArrow" in a["keys"])
        has_space = sum(1 for a in anns if "Space" in a["keys"])
        for a in anns:
            keys.update(a["keys"])
        print(f"=== {ep[:8]} ===")
        print(f"  按键分布: {dict(keys)}")
        print(f"  鼠标: 左移帧={left}, 右移帧={right}, 静止帧={zero}")
        print(f"  特殊键: LeftArrow={has_left_key}, RightArrow={has_right_key}, Space={has_space}")


if __name__ == "__main__":
    main()
