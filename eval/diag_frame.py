#!/usr/bin/env python
"""诊断视频帧内容：解码前几帧，打印 shape 与像素统计，并保存首帧为 PNG 供人工检查。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402


def main():
    episode = Path(sys.argv[1])
    video_name = sys.argv[2] if len(sys.argv) > 2 else "192x192.mp4"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    frames = evaluate.load_video_frames(episode, video_name, 0, n)
    for i, fb in enumerate(frames):
        arr = np.frombuffer(fb, dtype=np.uint8).reshape(192, 192, 3)
        print(f"帧 {i}: shape={arr.shape}, mean={arr.mean():.1f}, "
              f"min={arr.min()}, max={arr.max()}, "
              f"Rmean={arr[:,:,0].mean():.1f}, Gmean={arr[:,:,1].mean():.1f}, Bmean={arr[:,:,2].mean():.1f}")

    # 保存首帧 PNG
    from PIL import Image
    arr0 = np.frombuffer(frames[0], dtype=np.uint8).reshape(192, 192, 3)
    out = Path("/tmp/frame0.png")
    Image.fromarray(arr0, "RGB").save(out)
    print(f"已保存首帧到 {out}")


if __name__ == "__main__":
    main()
