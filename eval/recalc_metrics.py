#!/usr/bin/env python
"""用已保存的预测 CSV + 标注重算完整指标（无需重启推理服务器）。"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402


def main():
    episode = Path(sys.argv[1])
    pred_csv = sys.argv[2]
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    num = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    proto = evaluate.load_annotation(episode)
    annotations = evaluate.get_annotations(proto, start, num)

    predictions = []
    with open(pred_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys = frozenset(k for k in (row["keys"] or "").split(",") if k.strip())
            predictions.append({
                "keys": keys,
                "mouse_delta_x": float(row["mouse_delta_x"]) if row.get("mouse_delta_x") not in (None, "") else 0.0,
                "mouse_delta_y": float(row["mouse_delta_y"]) if row.get("mouse_delta_y") not in (None, "") else 0.0,
            })
    predictions = predictions[:num]

    m = evaluate.compute_metrics(annotations, predictions)
    print(f"帧数: {len(annotations)}")
    print(f"按键准确率: {m['key_accuracy']:.4f}")
    print(f"鼠标相关系数: x={m['mouse_corr_x']:.4f}, y={m['mouse_corr_y']:.4f}, mean={m['mouse_corr_mean']:.4f}")
    print(f"位移MAE: x={m['mouse_mae_x']:.4f}, y={m['mouse_mae_y']:.4f}, 欧氏={m['mouse_euclidean_mae']:.4f}")


if __name__ == "__main__":
    main()
