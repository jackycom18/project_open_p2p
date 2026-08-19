#!/usr/bin/env python
"""record_demo.py —— 模型输出 vs 人类标注 逐帧对比视频（课题六 · MVP4 录屏演示）

生成一段 mp4：
  * 上半屏：原始视频帧（放大）+ 帧号 / 指令文本；
  * 下半屏：每帧的 人类标注按键/鼠标位移 vs 模型预测按键/鼠标位移。

运行前提：
  1. 官方推理服务器已启动（终端1，见 README）；
  2. 本脚本连 UDS 拉取模型预测（与评测一致的链路）。

用法（WSL，~/open-p2p 下）：
  uv run python eval/record_demo.py --episode <game>/<episode_id> \
      --start-frame 0 --num-frames 120 --instruction "向左移动"
  uv run python eval/record_demo.py --episode <game>/<episode_id> \
      --predictions-csv eval_results/xxx.csv   # 或从已有评测结果读预测

输出：demo/<episode>_<start>_<n>.mp4（帧率 --fps，默认 10）
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    sys.exit("需要 Pillow（官方依赖已包含）：uv sync 后可用")

import evaluate  # noqa: E402
from p2p_client import InferenceClient, make_frame  # noqa: E402

VIDEO_NAMES = ("192x192.mp4", "video.mp4", "data.mp4")
UP_H, DOWN_H = 384, 168
CANVAS_W = 768
BAR_W = (CANVAS_W - 40) // 2  # 左右动作栏宽度
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for fp in FONT_PATHS:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def resolve_video(episode_dir: Path, video_name: str | None) -> Path:
    if video_name:
        return episode_dir / video_name
    for v in VIDEO_NAMES:
        if (episode_dir / v).exists():
            return episode_dir / v
    raise FileNotFoundError(f"{episode_dir} 下找不到视频文件")


def draw_bar(draw: ImageDraw.ImageDraw, x0: int, y0: int, title: str,
             keys: frozenset, dx: float, dy: float, font, color: tuple) -> None:
    """绘制一个动作栏：标题 + 按键 + 鼠标位移。"""
    w, h = BAR_W, DOWN_H - 20
    draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=8, fill=(245, 245, 245), outline=color, width=2)
    draw.text((x0 + 12, y0 + 8), title, fill=color, font=font)
    key_str = "+".join(sorted(keys)) if keys else "·"
    draw.text((x0 + 12, y0 + 44), f"按键: {key_str}", fill=(20, 20, 20), font=font)
    draw.text((x0 + 12, y0 + 78), f"鼠标: dx={dx:+.1f} dy={dy:+.1f}", fill=(20, 20, 20), font=font)
    draw.text((x0 + 12, y0 + 112), "按住=同键按下持续帧", fill=(120, 120, 120), font=font)


def compose_frame(frame_hwc: np.ndarray, instruction: str, idx: int, total: int,
                  human: dict, model: dict, font, font_small) -> Image.Image:
    canvas = Image.new("RGB", (CANVAS_W, UP_H + DOWN_H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 上半屏：视频帧
    img = Image.fromarray(frame_hwc, "RGB").resize((UP_H, UP_H), Image.BILINEAR)
    canvas.paste(img, ((CANVAS_W - UP_H) // 2, 0))
    draw.text((10, 10), f"帧 {idx}/{total}  指令: {instruction or '(无)'}",
              fill=(220, 40, 40), font=font)

    # 分隔线
    draw.line([(0, UP_H), (CANVAS_W, UP_H)], fill=(0, 0, 0), width=2)

    # 下半屏：动作对比
    draw_bar(draw, 20, UP_H + 10, "人类标注 (GT)",
             human["keys"], human["mouse_delta_x"] or 0, human["mouse_delta_y"] or 0,
             font, (30, 120, 220))
    draw_bar(draw, 20 + BAR_W + 20, UP_H + 10, "模型输出 (Pred)",
             model["keys"], model["mouse_delta_x"] or 0, model["mouse_delta_y"] or 0,
             font, (220, 90, 30))
    return canvas


def load_predictions_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "keys": frozenset(r.get("keys", "").split(",")) if r.get("keys") else frozenset(),
                "mouse_delta_x": float(r.get("mouse_delta_x") or 0),
                "mouse_delta_y": float(r.get("mouse_delta_y") or 0),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--episode", required=True, help="片段目录（相对 --dataset）")
    ap.add_argument("--video-name", default=None)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=120)
    ap.add_argument("--instruction", default=None, help="演示时展示的指令文本")
    ap.add_argument("--predictions-csv", default=None,
                    help="若提供，从该 CSV 读模型预测（键 mouse_delta_x/mouse_delta_y/keys）")
    ap.add_argument("--uds-path", default="/tmp/uds.recap")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out-dir", default="demo")
    args = ap.parse_args()

    episode_dir = Path(args.dataset) / args.episode
    if not episode_dir.is_dir():
        sys.exit(f"片段目录不存在: {episode_dir}")

    proto = evaluate.load_annotation(episode_dir)
    annotations = evaluate.get_annotations(proto, args.start_frame, args.num_frames)
    video_path = resolve_video(episode_dir, args.video_name)
    frame_bytes = evaluate.load_video_frames(
        episode_dir, video_path.name, args.start_frame, len(annotations)
    )

    # 模型预测
    if args.predictions_csv:
        predictions = load_predictions_csv(Path(args.predictions_csv))
        logger_hint = f"从 {args.predictions_csv} 读取预测"
    else:
        width = proto.metadata.frame_width if proto.metadata.HasField("frame_width") else 192
        height = proto.metadata.frame_height if proto.metadata.HasField("frame_height") else 192
        client = InferenceClient(args.uds_path)
        frames = [make_frame(fb, width, height, i) for i, fb in enumerate(frame_bytes)]
        predictions = client.run(frames)
        logger_hint = f"连接 {args.uds_path} 实时推理"

    n = min(len(annotations), len(predictions), len(frame_bytes))
    print(f"[info] {logger_hint}，共 {n} 帧")

    font = load_font(20)
    images = []
    for i in range(n):
        fhwc = np.frombuffer(frame_bytes[i], dtype=np.uint8).reshape(
            192, 192, 3) if len(frame_bytes[i]) == 192 * 192 * 3 else None
        if fhwc is None:
            print("[warn] 无法解析帧，使用纯色占位")
            fhwc = np.full((192, 192, 3), 200, dtype=np.uint8)
        canvas = compose_frame(
            fhwc, args.instruction, i + 1, n,
            annotations[i], predictions[i], font, font,
        )
        images.append(np.asarray(canvas, dtype=np.uint8))

    # ffmpeg 合成 mp4
    out_path = Path(args.out_dir) / f"{args.episode.replace('/', '_')}_{args.start_frame}_{n}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H, W = images[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for arr in images:
        proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        sys.exit(f"[ERROR] ffmpeg 合成失败: {out_path}")
    print(f"[done] 对比视频已生成: {out_path}（{n} 帧, {args.fps}fps, {W}x{H}）")


if __name__ == "__main__":
    main()
