#!/usr/bin/env python
"""run_instruct_experiment.py —— 文本指令对照实验编排器（课题六 · 立项扩展）

实验设计（与立项书一致）：
  对照组（不带指令）：启动官方推理服务器（无 --input_text），模型文本输入为
                      训练默认的 zeros 嵌入，评测全部指令测试单元。
  实验组（带指令）  ：启动官方推理服务器（--input_text），评测每个单元前向
                      服务器 stdin 注入对应指令文本，评测同样单元。
  对比每组每个单元的"任务完成率"，报告带指令 / 不带指令两组指标。

设计要点（代码中已落实）：
  * 两组实验各使用独立服务器进程，避免 KV cache / 文本状态跨组污染；
  * 组内单元顺序与 testset.json 一致，保证两组变量除"指令"外完全相同；
  * 指令通过官方支持的 stdin 按行动态注入（SharedTextInputState 机制）。

用法（WSL，~/open-p2p 下）：
  uv run python eval/run_instruct_experiment.py \
      --testset eval/testset.json \
      --config checkpoints/150M/model_config.yaml \
      --checkpoint-path checkpoints/150M/checkpoint-step=00500000.ckpt \
      --dataset dataset

输出：eval_results/experiment_<时间戳>/
      units_no_instr.csv / units_with_instr.csv / summary.csv / report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import socket
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
from p2p_client import InferenceClient, make_frame  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("instruct_experiment")

DEFAULT_UDS = "/tmp/uds.recap"
VIDEO_NAMES = ("192x192.mp4", "video.mp4", "data.mp4")


# ---------------------------------------------------------------------------
# 服务器进程管理
# ---------------------------------------------------------------------------


def wait_until_ready(uds_path: str, timeout: float = 900.0) -> None:
    """轮询等待 UDS socket 可连接（服务器模型加载 + warmup 已完成）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(uds_path):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(2.0)
                s.connect(uds_path)
                s.close()
                return
            except OSError:
                pass
        time.sleep(2.0)
    raise TimeoutError(f"等待推理服务器超时（{uds_path}）")


class ServerSession:
    """管理一个官方推理服务器进程的生命周期。"""

    def __init__(self, use_input_text: bool, config: Path, ckpt: Path, uds_path: str,
                 log_path: Path, uv_cmd: list[str]) -> None:
        self.use_input_text = use_input_text
        self.uds_path = uds_path
        cmd = uv_cmd + [
            "elefant/policy_model/inference.py",
            "--config", str(config),
            "--checkpoint_path", str(ckpt),
        ]
        if use_input_text:
            cmd.append("--input_text")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")
        logger.info("启动推理服务器: %s", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def wait_ready(self, timeout: float = 900.0) -> None:
        wait_until_ready(self.uds_path, timeout)

    def send_instruction(self, text: str) -> None:
        """向服务器 stdin 注入一行指令（官方 SharedTextInputState 机制）。"""
        assert self.proc.stdin is not None, "服务器未开启 stdin（需 --input_text）"
        self.proc.stdin.write(text.strip() + "\n")
        self.proc.stdin.flush()
        logger.info("已注入指令: %r", text)

    def stop(self, grace: float = 15.0) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.log_file.close()


# ---------------------------------------------------------------------------
# 单个测试单元的评测（复用 evaluate 模块）
# ---------------------------------------------------------------------------


def resolve_video_name(episode_dir: Path, video_name: str | None) -> str:
    if video_name:
        return video_name
    for v in VIDEO_NAMES:
        if (episode_dir / v).exists():
            return v
    raise FileNotFoundError(f"{episode_dir} 下找不到视频文件（尝试 {VIDEO_NAMES}）")


def run_unit(unit: dict, dataset: Path, video_name: str | None,
             uds_path: str, client: InferenceClient) -> dict:
    episode_dir = dataset / unit["episode"]
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"片段目录不存在: {episode_dir}")

    start = int(unit.get("start_frame", 0))
    num = int(unit.get("num_frames", 200))

    proto = evaluate.load_annotation(episode_dir)
    annotations = evaluate.get_annotations(proto, start, num)
    if not annotations:
        raise ValueError(f"{unit['episode']} 无可评测帧")

    vname = resolve_video_name(episode_dir, video_name)
    frame_bytes = evaluate.load_video_frames(episode_dir, vname, start, len(annotations))
    width = proto.metadata.frame_width if proto.metadata.HasField("frame_width") else 192
    height = proto.metadata.frame_height if proto.metadata.HasField("frame_height") else 192

    frames = [make_frame(fb, width, height, i) for i, fb in enumerate(frame_bytes)]
    predictions = client.run(frames)
    if len(predictions) != len(annotations):
        logger.warning("%s 预测帧数 %d != 标注帧数 %d，按较短者对齐",
                       unit["episode"], len(predictions), len(annotations))

    metrics = evaluate.compute_metrics(annotations, predictions)

    task_done, task_reason = None, ""
    if unit.get("task"):
        task_done, task_reason = evaluate.check_instruction(
            unit["task"], predictions, len(predictions)
        )

    row = {
        "episode": unit["episode"],
        "start_frame": start,
        "num_frames": len(annotations),
        "instruction": unit.get("instruction", ""),
        "task": unit.get("task", ""),
        "key_accuracy": round(metrics["key_accuracy"], 4),
        "mouse_corr_x": round(metrics["mouse_corr_x"], 4),
        "mouse_corr_y": round(metrics["mouse_corr_y"], 4),
        "mouse_corr_mean": round(metrics["mouse_corr_mean"], 4),
        "mouse_mae_x": round(metrics["mouse_mae_x"], 2),
        "mouse_mae_y": round(metrics["mouse_mae_y"], 2),
        "mouse_euclidean_mae": round(metrics["mouse_euclidean_mae"], 2),
        "task_done": task_done if task_done is not None else "",
        "task_reason": task_reason,
    }
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info("已写入 %s（%d 行）", path, len(rows))


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def summarize_full(rows_no: list[dict], rows_instr: list[dict]) -> list[dict]:
    """聚合对照组/实验组，输出最终汇总表。"""
    def group(rows):
        g = {}
        for r in rows:
            key = (r["episode"], r["instruction"], r["task"])
            if key not in g:
                g[key] = {"units": 0, "task_done": 0, "key_acc": 0.0, "corr": 0.0}
            gg = g[key]
            gg["units"] += 1
            gg["task_done"] += 1 if str(r.get("task_done", "")) == "True" else 0
            gg["key_acc"] += float(r["key_accuracy"] or 0)
            gg["corr"] += float(r["mouse_corr_mean"] or 0)
        return g

    gn = group(rows_no)
    gi = group(rows_instr)
    keys = list(gi.keys())
    out = []
    for k in keys:
        a = gn.get(k)
        b = gi.get(k)
        if a is None or b is None:
            continue
        ep, instr, task = k
        out.append({
            "episode": ep,
            "instruction": instr,
            "task": task,
            "units": a["units"],
            "no_instr_done": f"{a['task_done']}/{a['units']}",
            "no_instr_completion": round(a["task_done"] / a["units"], 3),
            "with_instr_done": f"{b['task_done']}/{b['units']}",
            "with_instr_completion": round(b["task_done"] / b["units"], 3),
            "no_instr_key_acc": round(a["key_acc"] / a["units"], 4),
            "with_instr_key_acc": round(b["key_acc"] / b["units"], 4),
            "no_instr_mouse_corr": round(a["corr"] / a["units"], 4),
            "with_instr_mouse_corr": round(b["corr"] / b["units"], 4),
        })
    return out


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("文本指令对照实验结果（指令片段完成率）")
    print("=" * 100)
    header = "{:<40} {:>10} {:>14} {:>14} {:>14} {:>14}".format(
        "episode", "units", "no_instr完成", "with_instr完成", "no_instr键率", "with_instr键率")
    print(header)
    print("-" * 100)
    for r in rows:
        print("{:<40} {:>10} {:>9} ({:>6.1f}%) {:>9} ({:>6.1f}%) {:>14.3f} {:>14.3f}".format(
            r["episode"][-40:], r["units"], r["no_instr_done"],
            100 * r["no_instr_completion"], r["with_instr_done"],
            100 * r["with_instr_completion"], r["no_instr_key_acc"], r["with_instr_key_acc"]))
    print("-" * 100)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = ["# 文本指令对照实验指标表", ""]
    lines.append("| 片段 | 指令 | 单元数 | 不带指令完成率 | 带指令完成率 | 不带指令按键率 | 带指令按键率 | 不带指令鼠标相关 | 带指令鼠标相关 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['episode']} | {r['instruction'] or '-'} | {r['units']} "
            f"| {r['no_instr_completion']:.1%} ({r['no_instr_done']}) "
            f"| {r['with_instr_completion']:.1%} ({r['with_instr_done']}) "
            f"| {r['no_instr_key_acc']:.3f} | {r['with_instr_key_acc']:.3f} "
            f"| {r['no_instr_mouse_corr']:.3f} | {r['with_instr_mouse_corr']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("指标表已写入 %s", path)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--testset", default="eval/testset.json", help="测试集清单")
    p.add_argument("--config", default="checkpoints/150M/model_config.yaml",
                   help="模型配置（与官方推理脚本一致）")
    p.add_argument("--checkpoint-path", default="checkpoints/150M/checkpoint-step=00500000.ckpt",
                   help="150M 检查点路径")
    p.add_argument("--uds-path", default=DEFAULT_UDS)
    p.add_argument("--video-name", default=None, help="覆盖视频文件名")
    p.add_argument("--output-dir", default="eval_results")
    p.add_argument("--uv-cmd", default="uv run", help="服务器启动命令前缀（默认 'uv run'）")
    p.add_argument("--server-timeout", type=float, default=900.0, help="服务器就绪等待秒数")
    p.add_argument("--only", choices=["no_instr", "with_instr"], default=None,
                   help="只跑其中一组（默认两组都跑）")
    return p.parse_args()


def main():
    args = parse_args()

    testset_path = Path(args.testset)
    if not testset_path.exists():
        sys.exit(f"找不到 testset.json: {testset_path}（先用 select_testset.py 生成）")
    spec = json.loads(testset_path.read_text(encoding="utf-8"))
    units = spec["testset"]
    if not units:
        sys.exit("testset.json 中没有测试单元")

    # 校验指令与任务
    no_instr_units = [u for u in units if u.get("task")]
    with_instr_units = [u for u in units if u.get("task") and u.get("instruction")]
    if not with_instr_units:
        logger.warning("没有同时含 task 与 instruction 的单元，带指令组将为空")
    logger.info("对照组单元 %d 个，实验组单元 %d 个", len(no_instr_units), len(with_instr_units))

    dataset = Path(args.dataset)
    out_dir = Path(args.output_dir) / f"experiment_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    uv_cmd = args.uv_cmd.split()
    rows_no, rows_instr = [], []

    # ---- 对照组：不带指令 ----
    if args.only in (None, "no_instr") and no_instr_units:
        srv = ServerSession(False, Path(args.config), Path(args.checkpoint_path),
                            args.uds_path, out_dir / "server_no_instr.log", uv_cmd)
        try:
            srv.wait_ready(args.server_timeout)
            client = InferenceClient(args.uds_path)
            for u in no_instr_units:
                try:
                    rows_no.append(run_unit(u, dataset, args.video_name, args.uds_path, client))
                except Exception as e:  # noqa: BLE001
                    logger.error("对照组单元 %s 失败: %s", u["episode"], e)
        finally:
            srv.stop()
        write_csv(out_dir / "units_no_instr.csv", rows_no)

    # ---- 实验组：带指令 ----
    if args.only in (None, "with_instr") and with_instr_units:
        srv = ServerSession(True, Path(args.config), Path(args.checkpoint_path),
                            args.uds_path, out_dir / "server_with_instr.log", uv_cmd)
        try:
            srv.wait_ready(args.server_timeout)
            client = InferenceClient(args.uds_path)
            for u in with_instr_units:
                # 每个单元评测前注入其指令（官方 stdin 机制，逐帧生效）
                srv.send_instruction(u["instruction"])
                try:
                    rows_instr.append(run_unit(u, dataset, args.video_name, args.uds_path, client))
                except Exception as e:  # noqa: BLE001
                    logger.error("实验组单元 %s 失败: %s", u["episode"], e)
        finally:
            srv.stop()
        write_csv(out_dir / "units_with_instr.csv", rows_instr)

    # ---- 汇总 ----
    if not rows_no and not rows_instr:
        sys.exit("没有任何评测结果")
    summary = summarize_full(rows_no, rows_instr)
    write_csv(out_dir / "summary.csv", summary)
    write_markdown(out_dir / "report.md", summary)
    print_summary(summary)
    print(f"\n[done] 结果目录: {out_dir}")


if __name__ == "__main__":
    main()
