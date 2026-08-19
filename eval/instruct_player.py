#!/usr/bin/env python
"""instruct_player.py —— 按帧推送文本指令的离线播放器（课题六 · 扩展）

解决"写一大串指令，按每一帧推送不同指令"的需求：
  读一段数据集录像 + 一份"指令时间轴"，逐帧把画面发给推理服务器；
  在发到时间轴里指定的那一帧之前，把对应新指令写进服务器 stdin
  （--input_text 模式），实现"同一段视频里按帧切换不同指令"。

指令文件（JSON）格式：
  {
    "timeline": [
      {"frame": 0,   "text": "向左走"},
      {"frame": 60,  "text": "跳一下"},
      {"frame": 120, "text": "向右走"}
    ]
  }
  frame 是相对片段起始的帧号（0 = 片段第一帧），text 是该帧起生效的指令。

用法（WSL，~/open-p2p 下）：
  uv run python eval/instruct_player.py \
      --episode roblox-rivals/0001_01_01_005 \
      --instructions eval/instructions.json \
      --config checkpoints/150M/model_config.yaml \
      --checkpoint-path checkpoints/150M/checkpoint-step=00500000.ckpt

输出：eval_results/instruct_<时间戳>/frames.csv
      （每帧：frame / instruction / keys / mouse_delta_x / mouse_delta_y）

关键点：
  * 脚本自己启动推理服务器子进程（持有 stdin），因此能中途注入指令；
  * 逐帧在同一个 UDS 连接里收发，保持模型 KV cache 不重置；
  * 注入指令后等待 --instruction-delay 秒，让服务器 stdin 监听线程生效。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from p2p_client import make_frame, action_to_tuple  # noqa: E402
from elefant.data.proto import video_inference_pb2  # noqa: E402
import evaluate  # noqa: E402
from run_instruct_experiment import ServerSession, resolve_video_name  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("instruct_player")

DEFAULT_UDS = "/tmp/uds.recap"


def load_timeline(path: Path, num_frames: int) -> dict[int, str]:
    """读取指令时间轴，返回 {相对帧号: 指令文本}。"""
    spec = json.loads(path.read_text(encoding="utf-8"))
    timeline: dict[int, str] = {}
    for item in spec["timeline"]:
        frame = int(item["frame"])
        if 0 <= frame < num_frames:
            timeline[frame] = str(item["text"]).strip()
    return timeline


async def stream_with_timeline(
    uds_path: str,
    frames: list,
    timeline: dict[int, str],
    send_instruction,
    instruction_delay: float,
) -> list[dict]:
    """逐帧发画面；发到指定帧前先注入该帧对应指令。返回每帧动作+指令。"""
    reader, writer = await asyncio.open_unix_connection(uds_path)
    results: list[dict] = []
    current_instr = ""
    try:
        for i, frame in enumerate(frames):
            # 当前相对帧号命中了时间轴 -> 先切换指令，再发这一帧
            if i in timeline:
                current_instr = timeline[i]
                send_instruction(current_instr)
                # 等服务器 stdin 监听线程读到并更新指令，避免这一帧仍用旧指令
                await asyncio.sleep(instruction_delay)

            # 发送帧
            data = frame.SerializeToString()
            writer.write(len(data).to_bytes(4, byteorder="little"))
            writer.write(data)
            await writer.drain()

            # 接收动作
            len_bytes = await reader.readexactly(4)
            action_len = int.from_bytes(len_bytes, byteorder="little")
            action_data = await reader.readexactly(action_len)
            action = video_inference_pb2.Action.FromString(action_data)
            row = action_to_tuple(action)
            row["instruction"] = current_instr
            results.append(row)

            if (i + 1) % 50 == 0:
                logger.info("已处理 %d/%d 帧", i + 1, len(frames))
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--episode", required=True, help="片段目录（相对 --dataset）")
    ap.add_argument("--video-name", default=None)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=200)
    ap.add_argument("--instructions", required=True, help="指令时间轴 JSON 文件")
    ap.add_argument("--config", required=True, help="模型配置 yaml")
    ap.add_argument("--checkpoint-path", required=True, help="150M 检查点路径")
    ap.add_argument("--uds-path", default=DEFAULT_UDS)
    ap.add_argument("--output-dir", default="eval_results")
    ap.add_argument("--uv-cmd", default="uv run")
    ap.add_argument("--instruction-delay", type=float, default=0.1,
                    help="注入指令后等待秒数（让 stdin 监听线程生效）")
    ap.add_argument("--server-timeout", type=float, default=900.0)
    args = ap.parse_args()

    episode_dir = Path(args.dataset) / args.episode
    if not episode_dir.is_dir():
        sys.exit(f"片段目录不存在: {episode_dir}")

    # 1. 读视频帧（画面来源：数据集录像）
    proto = evaluate.load_annotation(episode_dir)
    annotations = evaluate.get_annotations(proto, args.start_frame, args.num_frames)
    if not annotations:
        sys.exit("没有可评测的标注帧")

    vname = resolve_video_name(episode_dir, args.video_name)
    frame_bytes = evaluate.load_video_frames(
        episode_dir, vname, args.start_frame, len(annotations)
    )
    width = proto.metadata.frame_width if proto.metadata.HasField("frame_width") else 192
    height = proto.metadata.frame_height if proto.metadata.HasField("frame_height") else 192
    frames = [make_frame(fb, width, height, i) for i, fb in enumerate(frame_bytes)]

    # 2. 读指令时间轴
    timeline = load_timeline(Path(args.instructions), len(annotations))
    if not timeline:
        sys.exit("指令时间轴为空，或所有指令帧号都落在评测区间外")
    logger.info("指令时间轴共 %d 条，将在这些相对帧切换指令: %s",
                len(timeline), sorted(timeline))

    out_dir = Path(args.output_dir) / f"instruct_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. 启动服务器（带 --input_text，脚本持有 stdin）
    srv = ServerSession(True, Path(args.config), Path(args.checkpoint_path),
                        args.uds_path, out_dir / "server.log", args.uv_cmd.split())
    try:
        srv.wait_ready(args.server_timeout)
        logger.info("开始逐帧推送指令...")
        predictions = asyncio.run(stream_with_timeline(
            args.uds_path, frames, timeline,
            srv.send_instruction, args.instruction_delay,
        ))
    finally:
        srv.stop()

    # 4. 保存逐帧结果（frame / instruction / 动作）
    csv_path = out_dir / "frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "instruction", "keys", "mouse_delta_x", "mouse_delta_y"])
        for i, p in enumerate(predictions):
            w.writerow([
                i,
                p["instruction"],
                ",".join(sorted(p["keys"])),
                p["mouse_delta_x"] if p["mouse_delta_x"] is not None else 0,
                p["mouse_delta_y"] if p["mouse_delta_y"] is not None else 0,
            ])
    logger.info("逐帧结果已写入 %s", csv_path)

    print("\n指令时间轴执行摘要:")
    for frame, text in sorted(timeline.items()):
        print(f"  第 {frame} 帧起 -> {text!r}")
    print(f"\n[done] 结果目录: {out_dir}")


if __name__ == "__main__":
    main()
