#!/usr/bin/env python
"""诊断鼠标位移：对比标注 vs 预测的分布，定位相关系数为 0 的原因。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402


def main():
    episode = Path(sys.argv[1])
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    num = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    pred_csv = sys.argv[4] if len(sys.argv) > 4 else None

    proto = evaluate.load_annotation(episode)
    anns = evaluate.get_annotations(proto, start, num)

    gt_x = np.array([a["mouse_delta_x"] for a in anns], dtype=float)
    gt_y = np.array([a["mouse_delta_y"] for a in anns], dtype=float)

    print("=== 标注 (GT) 鼠标位移分布 ===")
    print(f"x: 非零帧数={np.count_nonzero(gt_x)}/{len(gt_x)}, "
          f"std={gt_x.std():.3f}, 唯一值数={len(np.unique(gt_x))}, "
          f"前20={gt_x[:20].tolist()}")
    print(f"y: 非零帧数={np.count_nonzero(gt_y)}/{len(gt_y)}, "
          f"std={gt_y.std():.3f}, 唯一值数={len(np.unique(gt_y))}, "
          f"前20={gt_y[:20].tolist()}")

    if pred_csv:
        import csv
        pr_x, pr_y = [], []
        with open(pred_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pr_x.append(float(row["mouse_delta_x"]) if row.get("mouse_delta_x") not in (None, "") else 0.0)
                pr_y.append(float(row["mouse_delta_y"]) if row.get("mouse_delta_y") not in (None, "") else 0.0)
        pr_x = np.array(pr_x[:num], dtype=float)
        pr_y = np.array(pr_y[:num], dtype=float)
        print("\n=== 预测 (Pred) 鼠标位移分布 ===")
        print(f"x: 非零帧数={np.count_nonzero(pr_x)}/{len(pr_x)}, "
              f"std={pr_x.std():.3f}, 唯一值数={len(np.unique(pr_x))}, "
              f"前20={pr_x[:20].tolist()}")
        print(f"y: 非零帧数={np.count_nonzero(pr_y)}/{len(pr_y)}, "
              f"std={pr_y.std():.3f}, 唯一值数={len(np.unique(pr_y))}, "
              f"前20={pr_y[:20].tolist()}")


if __name__ == "__main__":
    main()
