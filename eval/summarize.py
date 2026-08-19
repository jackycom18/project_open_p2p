#!/usr/bin/env python
"""summarize.py —— 评测指标汇总与归档工具（课题六 · MVP5 指标表）

功能：
  1. 扫描指定目录（默认 eval_results/）下的所有指标 CSV（evaluate.py /
     run_instruct_experiment.py 的输出）；
  2. 计算平均按键准确率、鼠标相关系数、任务完成率；
  3. 输出控制台汇总表 + Markdown 指标表（可直接并入实验报告）。

用法：
  uv run python eval/summarize.py --input-dir eval_results --out eval_results/指标汇总.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

HEADERS = {
    "episode", "instruction", "start_frame", "frames", "key_accuracy",
    "mouse_corr_x", "mouse_corr_y", "mouse_corr_mean", "mouse_mae_x",
    "mouse_mae_y", "mouse_euclidean_mae", "task", "task_done",
}


def read_csvs(input_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(input_dir.rglob("*.csv")):
        if p.name == "summary.csv":
            continue  # 已聚合的结果跳过，避免重复
        try:
            with p.open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("episode"):
                        r["_src"] = p.name
                        rows.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 跳过 {p}: {e}")
    return rows


def fmt(x: str, cast=float, digits=3) -> float:
    try:
        return round(cast(x), digits) if x not in ("", None) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default="eval_results", help="指标 CSV 所在目录")
    ap.add_argument("--out", default="", help="Markdown 输出路径（默认只打印控制台）")
    ap.add_argument("--min-rows", type=int, default=1, help="每个片段至少要有多少行才纳入汇总")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"目录不存在: {input_dir}")
    rows = read_csvs(input_dir)
    if not rows:
        print(f"[warn] {input_dir} 下没有找到指标 CSV（先运行 evaluate.py / run_instruct_experiment.py）")
        return

    # 按 episode 分组（指令片段按 instruction 再细分）
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r.get("episode", ""), r.get("instruction", ""))].append(r)

    print(f"\n共 {len(rows)} 条评测记录，涉及 {len(groups)} 个片段\n")
    summary_rows = []
    for (ep, instr), g in sorted(groups.items()):
        if len(g) < args.min_rows:
            continue
        done = [r for r in g if str(r.get("task_done", "")).lower() == "true"]
        n = len(g)
        key_acc = sum(fmt(r.get("key_accuracy", ""), float, 4) for r in g) / n
        corr = sum(fmt(r.get("mouse_corr_mean", ""), float, 4) for r in g) / n
        mae = sum(fmt(r.get("mouse_euclidean_mae", ""), float, 2) for r in g) / n
        completion = len(done) / n
        summary_rows.append({
            "episode": ep, "instruction": instr, "n": n,
            "key_accuracy": round(key_acc, 4),
            "mouse_corr_mean": round(corr, 4),
            "mouse_mae": round(mae, 2),
            "task_done": f"{len(done)}/{n}",
            "task_completion": round(completion, 3),
        })
        print(f"{ep:<48} n={n:>3}  按键率={key_acc:.3f}  鼠标相关={corr:.3f}  "
              f"MAE={mae:.2f}px  完成={len(done)}/{n}")

    if not summary_rows:
        sys.exit("没有满足 min-rows 的分组")

    # 总体平均（按键率/鼠标相关按 n 加权）
    total_n = sum(s["n"] for s in summary_rows)
    mean_key = sum(s["key_accuracy"] * s["n"] for s in summary_rows) / total_n
    mean_corr = sum(s["mouse_corr_mean"] * s["n"] for s in summary_rows) / total_n
    mean_mae = sum(s["mouse_mae"] * s["n"] for s in summary_rows) / total_n
    total_done = sum(int(s["task_done"].split("/")[0]) for s in summary_rows)
    print("-" * 100)
    print(f"总体（{total_n} 条记录）: 按键准确率={mean_key:.3f}  鼠标相关系数={mean_corr:.3f}  "
          f"位移MAE={mean_mae:.2f}px  任务完成率={total_done}/{total_n}={total_done/total_n:.1%}")

    # 与课题验收线对比
    print("\n验收对照（MVP3 基线: 按键率≥55%、鼠标相关≥0.5）:")
    print(f"  按键准确率 {mean_key:.1%}  {'达标' if mean_key >= 0.55 else '未达标'}")
    print(f"  鼠标相关系数 {mean_corr:.3f}  {'达标' if mean_corr >= 0.5 else '未达标'}")
    if any(s["task_completion"] > 0 for s in summary_rows):
        print("扩展验收（带指令完成率≥60%、不带指令≤35%）需对比 run_instruct_experiment.py 的输出")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# 评测指标汇总表", ""]
        lines.append("| 片段 | 指令 | 记录数 | 按键准确率 | 鼠标相关系数 | 位移MAE(px) | 任务完成 | 完成率 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for s in summary_rows:
            lines.append(f"| {s['episode']} | {s['instruction'] or '-'} | {s['n']} "
                         f"| {s['key_accuracy']:.3f} | {s['mouse_corr_mean']:.3f} "
                         f"| {s['mouse_mae']:.2f} | {s['task_done']} | {s['task_completion']:.1%} |")
        lines.append(f"| **总体** | - | {total_n} | **{mean_key:.3f}** | **{mean_corr:.3f}** "
                     f"| **{mean_mae:.2f}** | **{total_done}/{total_n}** | **{total_done/total_n:.1%}** |")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[done] 指标表已写入 {out}")


if __name__ == "__main__":
    main()
