"""open-p2p 推理服务器 UDS 客户端封装。

协议依据官方实现（elefant/inference/unix_socket_server.py）：
  客户端发送: 4 字节小端长度 + 序列化的 video_inference_pb2.Frame
  服务端返回: 4 字节小端长度 + 序列化的 video_inference_pb2.Action

用法（配合官方推理服务器）:
  终端1: uv run elefant/policy_model/inference.py --config ... --checkpoint_path ...
  终端2: python -c "from p2p_client import InferenceClient; ..."
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Sequence

from elefant.data.proto import video_inference_pb2

logger = logging.getLogger(__name__)

DEFAULT_UDS_PATH = "/tmp/uds.recap"


def make_frame(
    frame_hwc_bytes: bytes, width: int, height: int, frame_id: int
) -> video_inference_pb2.Frame:
    """构造 Frame 消息（data 为 HWC 顺序的原始 RGB 字节，与官方客户端一致）。"""
    return video_inference_pb2.Frame(
        data=frame_hwc_bytes, width=width, height=height, id=frame_id
    )


def action_to_tuple(action: video_inference_pb2.Action) -> dict:
    """把 Action 消息转成便于比较的 dict。

    返回:
      {
        "id": int,
        "keys": frozenset[str],
        "mouse_delta_x": int | None,
        "mouse_delta_y": int | None,
      }
    """
    mouse = action.mouse_action
    has_delta = mouse.HasField("mouse_delta_px")
    return {
        "id": action.id,
        "keys": frozenset(action.keys),
        "mouse_delta_x": mouse.mouse_delta_px.x if has_delta else None,
        "mouse_delta_y": mouse.mouse_delta_px.y if has_delta else None,
    }


class InferenceClient:
    """通过 Unix Domain Socket 连接 open-p2p 推理服务器的同步客户端。"""

    def __init__(self, uds_path: str = DEFAULT_UDS_PATH, timeout: float = 120.0):
        self.uds_path = uds_path
        self.timeout = timeout

    def run(
        self,
        frames: Sequence[video_inference_pb2.Frame],
        sleep_between_frames: float = 0.0,
    ) -> List[dict]:
        """发送帧序列，返回每帧的模型动作（list[dict]，见 action_to_tuple）。"""
        return asyncio.run(self._run_async(frames, sleep_between_frames))

    async def _run_async(
        self,
        frames: Sequence[video_inference_pb2.Frame],
        sleep_between_frames: float,
    ) -> List[dict]:
        reader, writer = await asyncio.open_unix_connection(self.uds_path)
        try:
            return await asyncio.wait_for(
                self._exchange(reader, writer, frames, sleep_between_frames),
                timeout=self.timeout * max(1, len(frames)),
            )
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _exchange(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        frames: Sequence[video_inference_pb2.Frame],
        sleep_between_frames: float,
    ) -> List[dict]:
        results: List[dict] = []
        for i, frame in enumerate(frames):
            # 发送
            data = frame.SerializeToString()
            writer.write(len(data).to_bytes(4, byteorder="little"))
            writer.write(data)
            await writer.drain()

            # 接收
            len_bytes = await reader.readexactly(4)
            action_len = int.from_bytes(len_bytes, byteorder="little")
            action_data = await reader.readexactly(action_len)
            action = video_inference_pb2.Action.FromString(action_data)
            results.append(action_to_tuple(action))

            if (i + 1) % 50 == 0:
                logger.info("已处理 %d/%d 帧", i + 1, len(frames))
            if sleep_between_frames > 0:
                await asyncio.sleep(sleep_between_frames)
        return results
