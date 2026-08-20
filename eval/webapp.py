#!/usr/bin/env python
"""webapp —— Open P2P 行为克隆游戏智能体 Web 演示界面。

提供一个浏览器可访问的前端页面，用于"软件系统展示"录屏：
  * 启动 / 停止 150M 推理服务器（支持 --input_text 文本指令模式）；
  * 选择测试片段，逐帧推理，实时展示游戏画面 + 模型预测动作 vs 人类标注；
  * 输入自然语言指令，观察模型动作变化。

运行（Linux，~/open-p2p 下）：
  export PATH="$HOME/.local/bin:$PATH"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  uv run python eval/webapp.py --port 8000

访问：http://<linux-ip>:8000
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# 让 webapp 能 import eval/ 下的模块
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate  # noqa: E402
from p2p_client import InferenceClient, make_frame  # noqa: E402
from run_instruct_experiment import resolve_video_name  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset" / "dataset"
CONFIG = "checkpoints/150M/model_config.yaml"
CKPT = "checkpoints/150M/checkpoint-step=00500000.ckpt"
UDS_PATH = "/tmp/uds.recap"

app = Flask(__name__)

# 全局状态：推理服务器进程
_server_proc: subprocess.Popen | None = None
_server_log_file = None
_server_input_text: bool = False


def _env() -> dict:
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
    return env


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    global _server_proc
    running = _server_proc is not None and _server_proc.poll() is None
    uds_ready = os.path.exists(UDS_PATH)
    return jsonify({
        "running": running,
        "uds_ready": uds_ready,
        "input_text": _server_input_text,
    })


@app.route("/api/server/start", methods=["POST"])
def api_start():
    global _server_proc, _server_log_file, _server_input_text
    if _server_proc is not None and _server_proc.poll() is None:
        return jsonify({"ok": False, "msg": "服务器已在运行"})

    input_text = request.json.get("input_text", False) if request.json else False
    cmd = [
        "uv", "run", "elefant/policy_model/inference.py",
        "--config", CONFIG,
        "--checkpoint_path", CKPT,
    ]
    if input_text:
        cmd.append("--input_text")

    _server_log_file = open("/tmp/webapp_server.log", "w", encoding="utf-8")
    _server_proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=_server_log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        env=_env(),
    )
    _server_input_text = input_text
    return jsonify({"ok": True, "msg": "服务器启动中，加载约需 1-2 分钟", "input_text": input_text})


@app.route("/api/server/stop", methods=["POST"])
def api_stop():
    global _server_proc, _server_log_file
    if _server_proc is not None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
        _server_proc = None
    if _server_log_file is not None:
        _server_log_file.close()
        _server_log_file = None
    if os.path.exists(UDS_PATH):
        try:
            os.unlink(UDS_PATH)
        except OSError:
            pass
    return jsonify({"ok": True, "msg": "已停止"})


@app.route("/api/instruction", methods=["POST"])
def api_instruction():
    global _server_proc
    if _server_proc is None or _server_proc.poll() is not None:
        return jsonify({"ok": False, "msg": "服务器未运行"})
    if not _server_input_text:
        return jsonify({"ok": False, "msg": "当前服务器未启用指令模式（需以 input_text 启动）"})
    text = (request.json or {}).get("text", "")
    if not text.strip():
        return jsonify({"ok": False, "msg": "指令为空"})
    _server_proc.stdin.write(text.strip().encode("utf-8") + b"\n")
    _server_proc.stdin.flush()
    return jsonify({"ok": True, "msg": f"指令已注入: {text}"})


@app.route("/api/episodes")
def api_episodes():
    eps = sorted(d.name for d in DATASET_DIR.iterdir() if d.is_dir())
    return jsonify({"episodes": eps})


@app.route("/api/infer", methods=["POST"])
def api_infer():
    global _server_proc
    if _server_proc is None or _server_proc.poll() is not None:
        return jsonify({"ok": False, "msg": "请先启动推理服务器"})
    if not os.path.exists(UDS_PATH):
        return jsonify({"ok": False, "msg": "服务器加载中，请稍候（UDS 未就绪）"})

    body = request.json or {}
    episode = body.get("episode")
    start = int(body.get("start_frame", 0))
    num = int(body.get("num_frames", 60))
    num = max(1, min(num, 300))

    episode_dir = DATASET_DIR / episode
    if not episode_dir.is_dir():
        return jsonify({"ok": False, "msg": f"片段不存在: {episode}"})

    try:
        proto = evaluate.load_annotation(episode_dir)
        annotations = evaluate.get_annotations(proto, start, num)
        vname = resolve_video_name(episode_dir, None)
        frame_bytes = evaluate.load_video_frames(episode_dir, vname, start, len(annotations))
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "msg": f"读取数据失败: {e}"})

    frames = [make_frame(fb, 192, 192, i) for i, fb in enumerate(frame_bytes)]
    try:
        predictions = InferenceClient(timeout=600).run(frames)
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "msg": f"推理失败: {e}"})

    import numpy as np
    from PIL import Image

    results = []
    for i, (fb, ann, pred) in enumerate(zip(frame_bytes, annotations, predictions)):
        arr = np.frombuffer(fb, dtype=np.uint8).reshape(192, 192, 3)
        img = Image.fromarray(arr, "RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        results.append({
            "frame_id": i,
            "image": "data:image/jpeg;base64," + b64,
            "pred_keys": sorted(pred["keys"]),
            "pred_mx": pred["mouse_delta_x"] if pred["mouse_delta_x"] is not None else 0,
            "pred_my": pred["mouse_delta_y"] if pred["mouse_delta_y"] is not None else 0,
            "gt_keys": sorted(ann["keys"]),
            "gt_mx": ann["mouse_delta_x"],
            "gt_my": ann["mouse_delta_y"],
        })

    return jsonify({"ok": True, "frames": results})


@app.route("/api/server/log")
def api_log():
    try:
        with open("/tmp/webapp_server.log", "r", encoding="utf-8") as f:
            lines = f.readlines()
        return jsonify({"ok": True, "tail": lines[-30:]})
    except FileNotFoundError:
        return jsonify({"ok": True, "tail": []})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
