# -*- coding: utf-8 -*-
"""生成 Linux 一键启动器（自解压 shell 脚本）。

把 webapp.py + templates/index.html 打包进一个自解压脚本，
在已配置好环境的 Linux 上执行 `bash p2p_demo.dat` 即可启动 Web 演示系统。
"""
import base64
import gzip
import io
import tarfile

HEADER = r"""#!/bin/bash
# ============================================================
#  Open P2P 行为克隆游戏智能体 —— Web 演示系统 一键启动器
#  用法: bash p2p_demo.dat
#  前提: 已在 ~/open-p2p 配好环境（uv / 模型权重 / 数据集）
# ============================================================
set -e

DEST="$HOME/open-p2p/eval"
REPO="$HOME/open-p2p"

echo "[1/3] 解压 Web 演示代码到 $DEST ..."
mkdir -p "$DEST/templates"
ARCHIVE_LINE=$(awk '/^__ARCHIVE_BELOW__$/{print NR+1; exit}' "$0")
tail -n +"$ARCHIVE_LINE" "$0" | base64 -d | tar -xz -C "$DEST"

echo "[2/3] 准备环境变量 ..."
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[3/3] 启动 Web 服务 (http://<本机IP>:8000) ..."
cd "$REPO"
exec uv run python eval/webapp.py --port 8000
exit 0
__ARCHIVE_BELOW__
"""


def main():
    # 1. 打包 webapp.py + templates/index.html 成 tar.gz
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add("eval/webapp.py", arcname="webapp.py")
        tar.add("eval/templates/index.html", arcname="templates/index.html")
    tar_bytes = buf.getvalue()

    # 2. base64 编码
    b64 = base64.b64encode(tar_bytes).decode()

    # 3. 拼接脚本头 + base64
    # 脚本头里的 r-string 包含字面 \n，需还原为真实换行
    header = HEADER.replace("\\n", "\n")
    launcher = header + b64 + "\n"

    out = "p2p_demo.dat"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(launcher)
    print(f"已生成 {out}，大小 {len(launcher)/1024:.1f} KB")


if __name__ == "__main__":
    main()
