#!/usr/bin/env bash
# ============================================================
# open-p2p 环境搭建脚本（在 WSL / Ubuntu 24.04 终端内执行）
#
# 严格依据官方 README 原文，逐步执行：
#   README.md -> "Training and Offline Inference (Linux)"
#     - Prerequisites
#     - Download Model Checkpoints
#     - Download Sample Dataset
#     - Inference（含 huggingface-cli login）
#
# 用法：
#   bash /mnt/c/Users/Xzj13/CodeBuddy/project/setup_wsl.sh
#
# 可选环境变量：
#   HF_TOKEN=<token>  设置后自动完成 HuggingFace 登录（否则最后会提示手动登录）
#   SKIP_DOWNLOADS=1  跳过模型与数据下载（仅搭建环境）
# ============================================================
set -euo pipefail

LOG="$HOME/p2p_setup.log"
P2P_DIR="$HOME/open-p2p"

log()  { echo -e "\n\033[1;36m[$(date '+%H:%M:%S')] $*\033[0m" | tee -a "$LOG"; }
done() { echo -e "\033[1;32m  -> OK\033[0m" | tee -a "$LOG"; }
skip() { echo -e "\033[1;33m  -> SKIP (已存在/已完成)\033[0m" | tee -a "$LOG"; }

touch "$LOG"
log "===== open-p2p 环境搭建开始 ($(date)) ====="

# ---------- 1. 官方 README Prerequisites ----------
log "[1/8] 安装系统依赖 build-essential git nvtop htop"
sudo apt update -y
sudo apt install -y build-essential git nvtop htop software-properties-common
done

log "[2/8] 安装 FFmpeg 7 (官方 PPA: ubuntuhandbook1/ffmpeg7)"
if ffmpeg -version 2>/dev/null | grep -q "7\."; then
    skip
else
    sudo add-apt-repository -y ppa:ubuntuhandbook1/ffmpeg7
    sudo apt update -y
    sudo apt install -y ffmpeg
    done
fi

log "[3/8] 安装 FFmpeg 开发库 (libavcodec-dev 等)"
sudo apt install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavdevice-dev libavfilter-dev
done

log "[4/8] 安装 Clang"
sudo apt install -y clang libclang-dev
done

log "[5/8] 安装 Rust (官方 sh.rustup.rs)"
if command -v cargo >/dev/null 2>&1; then
    skip
else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
    done
fi

log "[6/8] 安装 socat (推理服务器通信)"
sudo apt install -y socat
done

log "[7/8] 提高文件描述符上限 (ulimit -n 65535)"
ulimit -n 65535
if ! grep -q "65535" "$HOME/.bashrc"; then
    echo "ulimit -n 65535" >> "$HOME/.bashrc"
fi
done

log "[8/8] 安装 uv 包管理器 (官方 astral.sh)"
if command -v uv >/dev/null 2>&1; then
    skip
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    done
fi

# ---------- 2. 克隆官方仓库并安装 Python 依赖 ----------
log "[9] 克隆 open-p2p 仓库"
if [ -d "$P2P_DIR/.git" ]; then
    skip
else
    git clone https://github.com/elefant-ai/open-p2p.git "$P2P_DIR"
    done
fi
cd "$P2P_DIR"

log "[10] uv sync (安装全部依赖，自动管理 Python 3.13.2)"
# 首次可能耗时较长（含 torch cu128 与 Rust 组件编译）
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv sync
done

# ---------- 3. 官方 README: Download Model Checkpoints ----------
if [ "${SKIP_DOWNLOADS:-0}" != "1" ]; then
    log "[11] 下载 150M 预训练检查点 (官方脚本)"
    if [ -d "$P2P_DIR/checkpoints/150M" ]; then
        skip
    else
        uv run python scripts/download_checkpoints.py 150M
        done
    fi

    # ---------- 4. 官方 README: Download Sample Dataset ----------
    log "[12] 下载 p2p-toy-examples 小样数据集 (官方脚本)"
    if [ -d "$P2P_DIR/dataset" ] && [ -n "$(ls -A "$P2P_DIR/dataset" 2>/dev/null)" ]; then
        skip
    else
        uv run python scripts/download_data.py --toy
        done
    fi

    # ---------- 5. 官方 README: Hugging Face 登录 (Gemma tokenizer 认证) ----------
    log "[13] Hugging Face 登录"
    if uv run huggingface-cli whoami >/dev/null 2>&1; then
        skip
    elif [ -n "${HF_TOKEN:-}" ]; then
        uv run huggingface-cli login --token "$HF_TOKEN"
        done
    else
        echo -e "\033[1;33m  -> 未提供 HF_TOKEN，请在 open-p2p 目录手动执行:\033[0m" | tee -a "$LOG"
        echo -e "\033[1;33m     cd ~/open-p2p && uv run huggingface-cli login\033[0m" | tee -a "$LOG"
    fi
else
    log "[11-13] 已设置 SKIP_DOWNLOADS=1，跳过模型/数据下载与登录"
fi

# ---------- 6. 同步课题评测代码 ----------
log "[14] 同步课题评测代码 (eval/) 到仓库"
if [ -d "$P2P_DIR/eval" ]; then
    log "  -> eval/ 已存在，增量同步新文件"
    cp -ru /mnt/c/Users/Xzj13/CodeBuddy/project/eval/. "$P2P_DIR/eval/"
    rm -rf "$P2P_DIR/eval/__pycache__"
else
    cp -r /mnt/c/Users/Xzj13/CodeBuddy/project/eval "$P2P_DIR/eval"
fi
done

# ---------- 7. 验证推理链路 ----------
log "[15] 验证推理链路 (随机权重，无需下载模型)"
cd "$P2P_DIR"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
ulimit -n 65535
uv run elefant/policy_model/inference.py \
  --config config/policy_model/150M.yaml \
  --use_random_weights 2>&1 | tee -a "$LOG"
done

log "===== 环境搭建完成 ====="
echo -e "\033[1;32m
下一步操作（见课题 README）：
  1) 推理服务器（无文本指令）：
       cd ~/open-p2p && uv run elefant/policy_model/inference.py \\
         --config checkpoints/150M/model_config.yaml \\
         --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
  2) 推理服务器（带文本指令，末尾追加 --input_text；
       运行时在服务器终端输入一行文本即可动态更新指令）：
  3) 完整日志见: $LOG\033[0m"
