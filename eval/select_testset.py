"""select_testset.py —— OpenP2P 数据浏览与测试集选择工具

用途（对应课题 MVP2/扩展）：
  1. 递归扫描数据集目录，对每个 episode 统计帧数、标注有效比例、按键分布、
     鼠标位移分布等，输出候选清单 CSV。
  2. 从候选片段中确定性（seed）抽取指定数量的测试单元，生成 testset.json，
     供 evaluate.py / run_instruct_experiment.py 直接使用。

用法：
  uv run python eval/select_testset.py --dataset dataset --output-dir eval_results
  uv run python eval/select_testset.py --dataset dataset --pick 6 --frames 200 \
      --seed 42 --testset-out eval/testset.json

测试集与训练/微调数据的隔离：
  - 官方 toy 小样本身不参与官方训练，天然与 150M 预训练权重隔离；
  - 若后续用 toy 微调，应把微调所用片段写入 --exclude 排除清单，
    再从剩余片段 / 不同 start_frame 取测试集；
  - 同一 seed 可完全复现测试集。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# 让 eval 目录内模块可直接 import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import load_annotation  # noqa: E402

VIDEO_NAMES = ("192x192.mp4", "video.mp4", "data.mp4")


def find_episodes(dataset: Path) -> list[Path]:
    """递归查找所有含 annotation.proto 的目录，视为一个 episode。"""
    episodes = []
    for proto in dataset.rglob("annotation.proto"):
        episodes.append(proto.parent)
    return sorted(episodes, key=lambda p: str(p).lower())


def _stats_episode(episode: Path) -> dict:
    annotation = load_annotation(episode / "annotation.proto")
    frames = list(annotation.frame_annotations)

    n = len(frames)
    is_known = sum(1 for f in frames if f.is_known)
    keys_counter: Counter[str] = Counter()
    move_count = 0
    move_abs = 0.0
    click_count = 0
    scroll_count = 0
    for f in frames:
        for k in f.user_action.keyboard.keys:
            keys_counter[k] += 1
        dx = f.user_action.mouse.mouse_delta_px.x
        dy = f.user_action.mouse.mouse_delta_px.y
        if dx or dy:
            move_count += 1
            move_abs += abs(dx) + abs(dy)
        if f.user_action.mouse.buttons_down:
            click_count += 1
        if f.user_action.mouse.scroll_delta_px.y or f.user_action.mouse.scroll_delta_px.x:
            scroll_count += 1

    video = next((episode / v for v in VIDEO_NAMES if (episode / v).exists()), None)
    return {
        "episode": episode.as_posix(),
        "frames": n,
        "is_known_ratio": round(is_known / n, 3) if n else 0.0,
        "active_ratio": round((move_count + click_count + scroll_count) / n, 3) if n else 0.0,
        "mouse_move_ratio": round(move_count / n, 3) if n else 0.0,
        "mouse_abs_delta": round(move_abs / max(1, move_count), 2),
        "click_ratio": round(click_count / n, 3) if n else 0.0,
        "scroll_ratio": round(scroll_count / n, 3) if n else 0.0,
        "top_keys": ",".join(f"{k}:{c}" for k, c in keys_counter.most_common(5)),
        "n_distinct_keys": len(keys_counter),
        "has_video": video is not None,
        "video_name": video.name if video else "",
    }


def action_stats(episode: Path, start: int, num: int) -> dict:
    """统计一段 [start, start+num) 帧内的动作特征，辅助挑选指令任务片段。"""
    annotation = load_annotation(episode / "annotation.proto")
    frames = list(annotation.frame_annotations)[start : start + num]
    keys: Counter[str] = Counter()
    move = 0
    for f in frames:
        for k in f.user_action.keyboard.keys:
            keys[k] += 1
        dx = f.user_action.mouse.mouse_delta_px.x
        dy = f.user_action.mouse.mouse_delta_px.y
        if dx or dy:
            move += 1
    n = len(frames)
    return {
        "keys": dict(keys.most_common(8)),
        "n_keys": len(keys),
        "move_ratio": round(move / n, 3) if n else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenP2P 数据浏览与测试集选择工具")
    ap.add_argument("--dataset", default="dataset", help="数据集根目录")
    ap.add_argument("--output-dir", default="eval_results", help="候选清单 CSV 输出目录")
    ap.add_argument("--exclude", default="", help="排除清单文件，每行一个 episode 相对路径")
    ap.add_argument("--pick", type=int, default=0, help="从中抽取 N 个测试单元写入 testset.json")
    ap.add_argument("--frames", type=int, default=200, help="每个测试单元帧数（<=200）")
    ap.add_argument("--start", type=int, default=0, help="默认起始帧偏移")
    ap.add_argument("--min-frames", type=int, default=200, help="候选片段最短帧数")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（确定性复现）")
    ap.add_argument("--testset-out", default="eval/testset.json", help="testset.json 输出路径")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not dataset.is_dir():
        sys.exit(f"[ERROR] 数据集目录不存在: {dataset}")

    excluded = set()
    if args.exclude:
        excluded = {line.strip() for line in Path(args.exclude).read_text().splitlines() if line.strip()}
        print(f"[info] 已加载排除清单 {len(excluded)} 条")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = [e for e in find_episodes(dataset) if e.as_posix() not in excluded]
    print(f"[info] 共发现 {len(episodes)} 个 episode")

    rows = []
    for ep in episodes:
        try:
            s = _stats_episode(ep)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 跳过 {ep}: {e}")
            continue
        rows.append(s)

    rows.sort(key=lambda r: r["frames"], reverse=True)
    csv_path = output_dir / "candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f"[info] 候选清单已写入 {csv_path}")

    # 控制台表格
    print("\n{:<52} {:>6} {:>8} {:>10} {:>9} {:>9} {:>10} {:>6}".format(
        "episode", "frames", "known%", "active%", "move%", "click%", "keys", "video"))
    print("-" * 120)
    for r in rows[:40]:
        print("{:<52} {:>6} {:>7.1f}% {:>9.1f}% {:>8.1f}% {:>8.1f}% {:>10} {:>6}".format(
            r["episode"][-52:], r["frames"],
            100 * r["is_known_ratio"], 100 * r["active_ratio"],
            100 * r["mouse_move_ratio"], 100 * r["click_ratio"],
            r["n_distinct_keys"], "Y" if r["has_video"] else "-"))
    print("-" * 120)

    if args.pick <= 0:
        print("\n[done] 未生成测试集。可加 --pick N 自动生成 testset.json")
        return

    # 生成 testset.json：优先动作丰富且帧数足够的片段
    rng = random.Random(args.seed)
    qualified = [r for r in rows if r["frames"] >= args.min_frames and r["has_video"]]
    if len(qualified) < args.pick:
        print(f"[warn] 满足条件(>= {args.min_frames} 帧且有视频)的片段仅 {len(qualified)} 个，"
              f"实际抽取 {len(qualified)} 个")
    rng.shuffle(qualified)
    picked = qualified[: args.pick]

    testset = []
    for r in picked:
        n_frames = min(args.frames, r["frames"] - args.start)
        testset.append({
            "episode": r["episode"],
            "start_frame": args.start,
            "num_frames": n_frames,
            "instruction": "",          # 文本指令，由用户填写
            "task": "",                 # 任务完成判定规则，如 "move_left"
        })

    doc = {
        "dataset": str(dataset),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "note": "测试集与训练/微调数据隔离说明见 README.md。instruction/task 字段需人工填写。",
        "testset": testset,
    }
    out = Path(args.testset_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] testset.json 已写入 {out}（{len(testset)} 个测试单元）")
    print("      请编辑其中的 instruction（文本指令）与 task（完成判定规则）字段")


if __name__ == "__main__":
    main()
