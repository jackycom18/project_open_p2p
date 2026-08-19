#!/usr/bin/env python
"""open-p2p 行为克隆评测脚本（课题六 · MVP2）

功能：
  1. 加载指定测试片段的视频帧（torchcodec 解码）与人类标注（annotation.proto）；
  2. 通过 UDS 将帧逐帧发送给官方推理服务器，收集模型动作；
  3. 计算并报告：按键准确率、鼠标位移 Pearson 相关系数、鼠标位移 MAE；
  4. （可选）文本指令任务完成率判定（立项扩展方向）。

运行前提（WSL / Ubuntu，在 ~/open-p2p 目录下）：
  终端1：启动推理服务器
      uv run elefant/policy_model/inference.py \
          --config checkpoints/150M/model_config.yaml \
          --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
      带文本指令时追加 --input_text
  终端2：运行评测
      uv run python eval/evaluate.py --episode <game>/<episode_id> --num-frames 200

输出：控制台指标报告 + eval_results/<tag>.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# 允许本脚本独立于仓库路径运行（eval/ 目录与 open-p2p 仓库同级时）
from p2p_client import InferenceClient, make_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")

try:
    from torchcodec.decoders import VideoDecoder
except Exception:  # pragma: no cover
    logger.warning("torchcodec 不可用，将无法解码视频帧")
    VideoDecoder = None

from elefant.data.proto import video_annotation_pb2

# ---------------------------------------------------------------------------
# 一、数据读取
# ---------------------------------------------------------------------------


def load_annotation(episode_dir: Path) -> video_annotation_pb2.VideoAnnotation:
    """加载 annotation.proto（与官方 playback.py 相同的解析方式）。"""
    proto_path = episode_dir / "annotation.proto"
    if not proto_path.exists():
        raise FileNotFoundError(f"找不到标注文件: {proto_path}")
    proto = video_annotation_pb2.VideoAnnotation()
    with open(proto_path, "rb") as f:
        proto.ParseFromString(f.read())
    logger.info("已加载标注 %s，共 %d 帧", proto_path, len(proto.frame_annotations))
    return proto


def load_video_frames(
    episode_dir: Path, video_name: str, start_frame: int, num_frames: int
) -> list[bytes]:
    """用 torchcodec 解码视频帧，返回 HWC 原始 RGB 字节列表（0-255）。

    与官方数据管线一致（elefant/data/video_proto_dataset.py 使用 torchcodec）。
    """
    if VideoDecoder is None:
        raise RuntimeError("torchcodec 未安装，无法解码视频")
    video_path = episode_dir / video_name
    if not video_path.exists():
        # 回退到常用命名
        video_path = episode_dir / "video.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"找不到视频文件: {video_path}")

    decoder = VideoDecoder(str(video_path), device="cpu", num_ffmpeg_threads=1)
    n_frames = len(decoder)
    end = min(start_frame + num_frames, n_frames)
    if end <= start_frame:
        raise ValueError(
            f"帧区间越界: start={start_frame}, num={num_frames}, 视频共 {n_frames} 帧"
        )
    # decoder[start:end] -> [N, C, H, W] uint8
    chunk = decoder[start_frame:end]
    frames_hwc = [frame.permute(1, 2, 0).numpy().tobytes() for frame in chunk]
    logger.info("已解码 %d 帧视频（%s，start=%d）", len(frames_hwc), video_path, start_frame)
    return frames_hwc


def get_annotations(
    proto: video_annotation_pb2.VideoAnnotation, start_frame: int, num_frames: int
) -> list[dict]:
    """提取 [start, start+num_frames) 的人类标注为统一 dict 列表。"""
    result = []
    for i in range(start_frame, min(start_frame + num_frames, len(proto.frame_annotations))):
        user_action = proto.frame_annotations[i].user_action
        assert user_action.is_known, f"第 {i} 帧用户动作未知"
        mouse = user_action.mouse
        result.append(
            {
                "frame": i,
                "keys": frozenset(user_action.keyboard.keys),
                "mouse_delta_x": mouse.mouse_delta_px.x,
                "mouse_delta_y": mouse.mouse_delta_px.y,
            }
        )
    return result


# ---------------------------------------------------------------------------
# 二、指标计算
# ---------------------------------------------------------------------------


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson 相关系数；序列无方差时返回 0.0 并告警。"""
    if len(x) == 0 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_metrics(annotations: list[dict], predictions: list[dict]) -> dict:
    """计算按键准确率、鼠标相关系数、鼠标位移误差。"""
    n = min(len(annotations), len(predictions))
    if n == 0:
        raise ValueError("评测帧数为 0")

    key_matches = sum(
        annotations[i]["keys"] == predictions[i]["keys"] for i in range(n)
    )
    key_accuracy = key_matches / n

    # 鼠标位移：None 视为 0（模型/标注在该帧无位移）
    gt_x = np.array([a["mouse_delta_x"] or 0 for a in annotations[:n]], dtype=float)
    gt_y = np.array([a["mouse_delta_y"] or 0 for a in annotations[:n]], dtype=float)
    pr_x = np.array([p["mouse_delta_x"] if p["mouse_delta_x"] is not None else 0 for p in predictions[:n]], dtype=float)
    pr_y = np.array([p["mouse_delta_y"] if p["mouse_delta_y"] is not None else 0 for p in predictions[:n]], dtype=float)

    r_x = pearson_r(gt_x, pr_x)
    r_y = pearson_r(gt_y, pr_y)
    r_mean = (r_x + r_y) / 2.0

    mae_x = float(np.mean(np.abs(gt_x - pr_x)))
    mae_y = float(np.mean(np.abs(gt_y - pr_y)))

    # 位移向量欧氏距离误差
    euc_err = float(np.mean(np.linalg.norm(np.stack([gt_x - pr_x, gt_y - pr_y], axis=1), axis=1)))

    return {
        "frames": n,
        "key_accuracy": key_accuracy,
        "mouse_corr_x": r_x,
        "mouse_corr_y": r_y,
        "mouse_corr_mean": r_mean,
        "mouse_mae_x": mae_x,
        "mouse_mae_y": mae_y,
        "mouse_euclidean_mae": euc_err,
    }


# ---------------------------------------------------------------------------
# 三、文本指令任务完成率判定（立项扩展方向）
# ---------------------------------------------------------------------------


def check_instruction(
    task: str, predictions: list[dict], n_frames: int
) -> tuple[bool, str]:
    """根据任务规则判断模型是否完成任务。返回 (是否完成, 判定依据)。

    支持的规则（见立项书）：
      move_left             ：片段内累计左移（mouse_delta_x < 0）超过阈值且按下左移键
      jump_then_turn_right  ：片段内按顺序检测到"跳跃(Space) → 右移(RightArrow)"动作
    """
    if task == "move_left":
        left_moves = [p for p in predictions if (p["mouse_delta_x"] or 0) < 0]
        total_left = sum(-(p["mouse_delta_x"] or 0) for p in left_moves)
        used_left_key = any("LeftArrow" in p["keys"] for p in predictions)
        ok = total_left >= 50 and used_left_key
        reason = (
            f"累计左移位移={total_left:.1f}px(阈值50)，"
            f"使用左移键={used_left_key}"
        )
        return ok, reason

    if task == "jump_then_turn_right":
        jump_at = next(
            (i for i, p in enumerate(predictions) if "Space" in p["keys"]), None
        )
        turn_after = (
            next(
                (i for i, p in enumerate(predictions) if i > (jump_at or -1) and "RightArrow" in p["keys"]),
                None,
            )
            if jump_at is not None
            else None
        )
        ok = jump_at is not None and turn_after is not None
        reason = (
            f"跳跃帧={jump_at}，跳跃后右转帧={turn_after}（片段 {n_frames} 帧）"
        )
        return ok, reason

    raise ValueError(f"未知任务规则: {task}")


# ---------------------------------------------------------------------------
# 四、主流程
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="dataset", help="数据集根目录（默认 dataset）")
    p.add_argument(
        "--episode",
        required=True,
        help="测试片段目录（相对 --dataset，如 roblox-rivals/0001_01_01_005）",
    )
    p.add_argument("--video-name", default="192x192.mp4", help="视频文件名（默认 192x192.mp4）")
    p.add_argument("--start-frame", type=int, default=0, help="起始帧（默认 0）")
    p.add_argument(
        "--num-frames",
        type=int,
        default=200,
        help="评测帧数（MVP 要求 ≤200，默认 200）",
    )
    p.add_argument("--uds-path", default="/tmp/uds.recap", help="推理服务器 UDS 路径")
    p.add_argument("--instruction", default=None, help="文本指令（带指令实验组）")
    p.add_argument(
        "--task",
        choices=["move_left", "jump_then_turn_right"],
        default=None,
        help="指令片段完成率判定规则（需同时提供 --instruction）",
    )
    p.add_argument("--tag", default=None, help="实验标签，用于输出文件名")
    p.add_argument("--output-dir", default="eval_results", help="指标表输出目录")
    p.add_argument(
        "--save-predictions",
        default=None,
        metavar="CSV",
        help="将逐帧预测动作保存为 CSV（供 record_demo.py 复用）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    episode_dir = Path(args.dataset) / args.episode
    if not episode_dir.is_dir():
        sys.exit(f"片段目录不存在: {episode_dir}")

    # 1. 人类标注
    proto = load_annotation(episode_dir)
    annotations = get_annotations(proto, args.start_frame, args.num_frames)
    if not annotations:
        sys.exit("没有可评测的人类标注帧")

    # 2. 视频帧（与标注对齐）
    frame_bytes = load_video_frames(episode_dir, args.video_name, args.start_frame, len(annotations))
    width = proto.metadata.frame_width if proto.metadata.HasField("frame_width") else 192
    height = proto.metadata.frame_height if proto.metadata.HasField("frame_height") else 192

    # 3. 推理
    logger.info("连接推理服务器 %s，发送 %d 帧...", args.uds_path, len(frame_bytes))
    client = InferenceClient(args.uds_path)
    frames = [make_frame(fb, width, height, i) for i, fb in enumerate(frame_bytes)]
    predictions = client.run(frames)
    if len(predictions) != len(annotations):
        logger.warning(
            "预测帧数(%d)与标注帧数(%d)不一致，按较短者对齐",
            len(predictions),
            len(annotations),
        )

    # 4. 指标
    metrics = compute_metrics(annotations, predictions)
    tag = args.tag or f"{args.episode.replace('/', '_')}_{args.start_frame}_{len(annotations)}"
    if args.instruction is not None:
        tag += "_with_instr"
    else:
        tag += "_no_instr"

    # 5. 文本指令任务完成率
    task_done, task_reason = None, None
    if args.task is not None:
        if args.instruction is None:
            logger.warning("--task 但未提供 --instruction，仍按规则判定（对照实验）")
        task_done, task_reason = check_instruction(args.task, predictions, len(predictions))
        metrics["task_done"] = task_done

    # 6. 报告
    print("\n" + "=" * 70)
    print(f"评测报告  episode={args.episode}  frames={metrics['frames']}")
    if args.instruction is not None:
        print(f"文本指令  {args.instruction!r}")
    print("=" * 70)
    print(f"按键准确率            : {metrics['key_accuracy']:.4f}  ({metrics['key_accuracy']*100:.1f}%)")
    print(f"鼠标相关系数 x        : {metrics['mouse_corr_x']:.4f}")
    print(f"鼠标相关系数 y        : {metrics['mouse_corr_y']:.4f}")
    print(f"鼠标相关系数 mean     : {metrics['mouse_corr_mean']:.4f}")
    print(f"鼠标位移 MAE (x,px)   : {metrics['mouse_mae_x']:.2f}")
    print(f"鼠标位移 MAE (y,px)   : {metrics['mouse_mae_y']:.2f}")
    print(f"位移欧氏误差 MAE (px) : {metrics['mouse_euclidean_mae']:.2f}")
    if task_done is not None:
        print(f"任务完成率判定        : {'完成' if task_done else '未完成'}  ({task_reason})")
    print("=" * 70)

    # 7. 指标表 CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"{tag}.csv"
    row = {
        "episode": args.episode,
        "instruction": args.instruction or "",
        "start_frame": args.start_frame,
        "frames": metrics["frames"],
        "key_accuracy": round(metrics["key_accuracy"], 4),
        "mouse_corr_x": round(metrics["mouse_corr_x"], 4),
        "mouse_corr_y": round(metrics["mouse_corr_y"], 4),
        "mouse_corr_mean": round(metrics["mouse_corr_mean"], 4),
        "mouse_mae_x": round(metrics["mouse_mae_x"], 2),
        "mouse_mae_y": round(metrics["mouse_mae_y"], 2),
        "mouse_euclidean_mae": round(metrics["mouse_euclidean_mae"], 2),
        "task": args.task or "",
        "task_done": task_done if task_done is not None else "",
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    logger.info("指标表已写入 %s", csv_path)

    # 8. （可选）逐帧预测动作 CSV，供 record_demo.py 复用
    if args.save_predictions:
        pred_path = Path(args.save_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "keys", "mouse_delta_x", "mouse_delta_y"])
            for i, p in enumerate(predictions[: len(annotations)]):
                keys = ",".join(sorted(p["keys"]))
                writer.writerow([
                    i,
                    keys,
                    p["mouse_delta_x"] if p["mouse_delta_x"] is not None else 0,
                    p["mouse_delta_y"] if p["mouse_delta_y"] is not None else 0,
                ])
        logger.info("逐帧预测已写入 %s", pred_path)


if __name__ == "__main__":
    main()
