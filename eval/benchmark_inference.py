#!/usr/bin/env python
"""benchmark_inference.py —— 推理性能评测（课题六 · 扩展 C）

测量固定测试集（约 200 帧）的批量推理性能，对比多种推理配置的耗时与帧率，
同时输出按键一致率（按键准确率）以证明优化不影响动作正确性。

优化方案（三组配置，同一份 200 帧测试集各跑一遍）：
  * kvcache     ：优化后，KV Cache 增量缓存（官方默认，逐帧复用历史，最快）；
  * full        ：优化前，--use_full_inference 全量重算（每次重算整段历史，慢）；
  * no_compile  ：对照组，--no-compile 关闭 torch.compile（仍用 KV Cache），
                  用于佐证编译加速的额外收益。

核心结论输出：
  * KV Cache 加速比 = full 耗时 / kvcache 耗时（预期 >1，越大越说明增量缓存的收益）；
  * 按键一致率变化 = kvcache 按键率 - full 按键率（预期接近 0，证明优化不牺牲正确性）。

用法（WSL，~/open-p2p 下）：
  uv run python eval/benchmark_inference.py \
      --dataset dataset \
      --episode <游戏>/<片段> \
      --num-frames 200 \
      --config checkpoints/150M/model_config.yaml \
      --checkpoint-path checkpoints/150M/checkpoint-step=00500000.ckpt

输出：eval_results/benchmark_<时间戳>/
      benchmark.csv   （每组配置的耗时/帧率/按键准确率）
      report.md       （性能对比表 + 优化结论）
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluate  # noqa: E402
from p2p_client import make_frame, action_to_tuple  # noqa: E402
from run_instruct_experiment import resolve_video_name  # noqa: E402
from elefant.data.proto import video_inference_pb2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("benchmark_inference")

DEFAULT_UDS = "/tmp/uds.recap"


def build_server_cmd(uv_cmd: list[str], config: str, ckpt: str,
                     no_compile: bool, full_inference: bool,
                     fps_test_frames: int | None = None) -> list[str]:
    """构造推理服务器启动命令（可切换 compile 与 full inference）。"""
    cmd = uv_cmd + [
        "elefant/policy_model/inference.py",
        "--config", config,
        "--checkpoint_path", ckpt,
    ]
    if no_compile:
        cmd.append("--no-compile")
    if full_inference:
        cmd.append("--use_full_inference")
    if fps_test_frames is not None:
        cmd.extend(["--fps_test_frames", str(fps_test_frames)])
    return cmd


def start_server(uv_cmd: list[str], config: str, ckpt: str,
                 no_compile: bool, full_inference: bool, log_path: Path,
                 fps_test_frames: int | None = None):
    """启动一个服务器子进程，返回 Popen 对象与日志文件句柄。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    cmd = build_server_cmd(uv_cmd, config, ckpt, no_compile, full_inference,
                           fps_test_frames)
    mode = "full" if full_inference else ("nocompile" if no_compile else "kvcache")
    logger.info("启动服务器（%s）: %s", mode, " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def stop_server(proc, log_file, grace: float = 15.0) -> None:
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    log_file.close()


async def run_frames_timed(uds_path: str, frames: list) -> tuple[list[dict], float]:
    """同一 UDS 连接逐帧发/收，返回 (动作列表, 总耗时秒)。"""
    reader, writer = await asyncio.open_unix_connection(uds_path)
    results: list[dict] = []
    try:
        start = time.perf_counter()
        for frame in frames:
            data = frame.SerializeToString()
            writer.write(len(data).to_bytes(4, byteorder="little"))
            writer.write(data)
            await writer.drain()

            len_bytes = await reader.readexactly(4)
            action_len = int.from_bytes(len_bytes, byteorder="little")
            action_data = await reader.readexactly(action_len)
            action = video_inference_pb2.Action.FromString(action_data)
            results.append(action_to_tuple(action))
        elapsed = time.perf_counter() - start
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass
    return results, elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--episode", required=True, help="片段目录（相对 --dataset）")
    ap.add_argument("--video-name", default=None)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=200)
    ap.add_argument("--config", required=True, help="模型配置 yaml")
    ap.add_argument("--checkpoint-path", required=True, help="150M 检查点路径")
    ap.add_argument("--uds-path", default=DEFAULT_UDS)
    ap.add_argument("--output-dir", default="eval_results")
    ap.add_argument("--uv-cmd", default="uv run")
    ap.add_argument("--server-timeout", type=float, default=900.0)
    ap.add_argument("--repeats", type=int, default=1,
                    help="每组配置重复测量次数（取平均）")
    ap.add_argument("--only", type=str, default=None,
                    help="只跑指定配置（kvcache / full / no_compile），逗号分隔可多个")
    args = ap.parse_args()

    episode_dir = Path(args.dataset) / args.episode
    if not episode_dir.is_dir():
        sys.exit(f"片段目录不存在: {episode_dir}")

    # 读人类标注 + 视频帧（两组配置共用同一份 200 帧）
    proto = evaluate.load_annotation(episode_dir)
    annotations = evaluate.get_annotations(proto, args.start_frame, args.num_frames)
    if not annotations:
        sys.exit("没有可评测的标注帧")
    vname = resolve_video_name(episode_dir, args.video_name)
    frame_bytes = evaluate.load_video_frames(episode_dir, vname, args.start_frame, len(annotations))
    # 官方 150M 模型固定 192x192 输入
    width, height = 192, 192
    frames = [make_frame(fb, width, height, i) for i, fb in enumerate(frame_bytes)]
    logger.info("准备评测 %d 帧（%s）", len(frames), args.episode)

    out_dir = Path(args.output_dir) / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    uv_cmd = args.uv_cmd.split()

    # 三组配置：
    #   kvcache     ：优化后（KV Cache 增量缓存，官方默认）
    #   full        ：优化前（--use_full_inference 全量重算，短 warmup 以免超时）
    #   no_compile  ：对照组（--no-compile 关闭编译，仍用 KV Cache）
    configs = [
        ("kvcache", False, False, None, "优化后（KV Cache 增量缓存）"),
        ("full", False, True, 200, "优化前（全量重算 --use_full_inference）"),
        ("no_compile", True, False, None, "对照组（关闭编译 --no-compile）"),
    ]

    rows = []
    only = set(args.only.split(",")) if args.only else None
    for tag, no_compile, full_inference, fps_frames, desc in configs:
        if only and tag not in only:
            continue
        for rep in range(args.repeats):
            proc, log_file = start_server(
                uv_cmd, args.config, args.checkpoint_path,
                no_compile, full_inference, out_dir / f"server_{tag}_{rep}.log",
                fps_test_frames=fps_frames,
            )
            try:
                # 等待服务器就绪：循环尝试 UDS 连接直到成功或超时
                import socket
                deadline = time.time() + args.server_timeout
                ready = False
                while time.time() < deadline:
                    if Path(args.uds_path).exists():
                        try:
                            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                            s.connect(args.uds_path)
                            s.close()
                            ready = True
                            break
                        except (ConnectionRefusedError, FileNotFoundError):
                            pass
                    time.sleep(2.0)
                if not ready:
                    logger.error("%s 服务器未就绪", desc)
                    continue

                predictions, elapsed = asyncio.run(run_frames_timed(args.uds_path, frames))
            finally:
                stop_server(proc, log_file)

            metrics = evaluate.compute_metrics(annotations, predictions)
            n = len(frames)
            fps = n / elapsed if elapsed > 0 else 0.0
            avg_ms = elapsed / n * 1000 if n > 0 else 0.0
            row = {
                "config": tag,
                "description": desc,
                "repeat": rep,
                "frames": n,
                "total_time_s": round(elapsed, 3),
                "fps": round(fps, 2),
                "avg_per_frame_ms": round(avg_ms, 2),
                "key_accuracy": round(metrics["key_accuracy"], 4),
            }
            rows.append(row)
            logger.info("%s 第%d次：耗时=%.3fs，fps=%.2f，按键一致率=%.4f",
                        desc, rep + 1, elapsed, fps, metrics["key_accuracy"])

    if not rows:
        sys.exit("没有任何有效的 benchmark 结果")

    # 写 CSV
    csv_path = out_dir / "benchmark.csv"
    fieldnames = ["config", "description", "repeat", "frames",
                  "total_time_s", "fps", "avg_per_frame_ms", "key_accuracy"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # 汇总（按 config 取平均）
    agg: dict[str, dict] = {}
    for r in rows:
        a = agg.setdefault(r["config"], {
            "description": r["description"], "n": 0, "time": 0.0,
            "fps": 0.0, "key": 0.0,
        })
        a["n"] += 1
        a["time"] += r["total_time_s"]
        a["fps"] += r["fps"]
        a["key"] += r["key_accuracy"]
    for a in agg.values():
        a["time"] /= a["n"]
        a["fps"] /= a["n"]
        a["key"] /= a["n"]

    kvcache = agg.get("kvcache")
    full = agg.get("full")
    no_compile = agg.get("no_compile")

    print("\n" + "=" * 70)
    print("推理性能评测报告（扩展 C）")
    print("=" * 70)
    for tag, a in agg.items():
        print(f"{a['description']}: 平均耗时 {a['time']:.3f}s, "
              f"fps {a['fps']:.2f}, 按键一致率 {a['key']:.4f}")

    # 主优化对比：KV Cache vs 全量重算
    speedup = 0.0
    key_delta = 0.0
    if kvcache and full:
        speedup = full["time"] / kvcache["time"]
        key_delta = kvcache["key"] - full["key"]
        print("-" * 70)
        print(f"KV Cache 加速比（全量重算耗时 / KV Cache 耗时）: {speedup:.2f}x")
        print(f"按键一致率变化（KV Cache - 全量重算）          : {key_delta:+.4f} "
              f"{'（无显著影响）' if abs(key_delta) < 0.02 else '（存在差异，需关注）'}")

    # 次对比：关闭编译对 KV Cache 的影响
    if kvcache and no_compile:
        cs = no_compile["time"] / kvcache["time"]
        print(f"关闭编译对 KV Cache 的影响（no-compile 耗时 / kvcache 耗时）: {cs:.2f}x")

    # 写 Markdown 报告
    report_path = out_dir / "report.md"
    lines = ["# 推理性能评测报告（扩展 C）", ""]
    lines.append("## 1. 各配置性能对比")
    lines.append("")
    lines.append("| 配置 | 平均耗时(s) | 帧率(fps) | 按键一致率 |")
    lines.append("|---|---|---|---|")
    for tag, a in agg.items():
        lines.append(f"| {a['description']} | {a['time']:.3f} | {a['fps']:.2f} | {a['key']:.4f} |")
    lines.append("")
    lines.append("## 2. 优化结论")
    lines.append("")
    if kvcache and full:
        lines.append(f"- **主优化（KV Cache vs 全量重算）**：加速比 {speedup:.2f}x；"
                     f"按键一致率变化 {key_delta:+.4f}，"
                     f"{'优化不影响按键一致率' if abs(key_delta) < 0.02 else '优化对按键一致率存在影响，需关注'}。")
    if kvcache and no_compile:
        lines.append(f"- **对照组（关闭编译）**：耗时比为 {no_compile['time'] / kvcache['time']:.2f}x，"
                     f"用于佐证 torch.compile 的额外加速效果。")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[done] 结果目录: {out_dir}")
    print(f"      指标表: {csv_path.name} / {report_path.name}")


if __name__ == "__main__":
    main()
