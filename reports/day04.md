# 《AI开发实践》实践报告（第 4 天）

| 项目 | 内容 |
|---|---|
| 课题名称 | 行为克隆游戏智能体（文本指令 + 推理性能方向） |
| 组名/成员 | 向子坚（独立完成） |
| 日期 | 2026-08-22 |
| 当日任务 | 工程/环境起步，能按说明运行 |
| 阶段成果 | 环境可运行推理脚本（含启动说明与审查记录） |

## 1. 今日目标

按第 2 天选型把环境搭起来，目标只有一个：**任何人按说明操作，就能把 `inference.py` 跑起来**（M1 的第一步）。同时完成可复现性审查。

## 2. 环境搭建内容

### 2.1 部署目标机

在独立 Linux 服务器（实验室网络，`100.92.15.14`）上部署；Windows 端保留 `setup_wsl.ps1` / `setup_wsl.sh` 脚本，便于 WSL2 场景复用同一套流程。

### 2.2 安装清单（与 setup 脚本一致）

| 软件 | 版本要求 | 用途 |
|---|---|---|
| NVIDIA 驱动 + CUDA | GPU 可用（`nvidia-smi` 正常） | 推理底座，显存 ≥8GB |
| uv | latest | Python 包管理与 Rust 组件编译 |
| FFmpeg | ≥7 | 视频帧解码 |
| Clang | latest | 依赖编译 |
| Rust（cargo） | stable | elefant_rust 编译 |
| HuggingFace CLI | latest | 下载权重/认证（Gemma tokenizer） |

关键步骤：

```bash
# 1) 安装系统依赖（uv / ffmpeg / clang / rust）
# 2) 克隆仓库
git clone https://github.com/elefant-ai/open-p2p.git ~/open-p2p
cd ~/open-p2p
# 3) 安装 Python 依赖并编译 Rust 组件
uv sync
# 4) 下载 150M 权重（需先 huggingface-cli login）
uv run python scripts/download_checkpoints.py 150M
# 5) 下载 toy 小样数据
uv run python scripts/download_data.py --toy
```

### 2.3 验证运行（先假后真）

按第 3 天约定先用随机权重打通链路：

```bash
uv run elefant/policy_model/inference.py \
  --config config/policy_model/150M.yaml \
  --use_random_weights
```

观察点：

- 模型加载成功，无 import/编译报错；
- 启动后自动执行 **FPS 测试**并打印结果（随机权重下链路已通）；
- 服务器监听 `/tmp/uds.recap` 成功，无显存不足报错。

## 3. 启动说明（可复现性审查）

- **前置**：`nvidia-smi` 可见 GPU；`huggingface-cli whoami` 已登录；
- **启动**：按上文命令即可；首次运行需等待 Rust 组件编译与权重加载；
- **文本指令模式**：启动命令加 `--input_text`，在服务器终端输入指令按回车生效；
- **审查结果**：命令行步骤均可复现；`uv.lock` 锁定依赖版本，跨机一致。

## 4. 遇到的问题与解决

| 问题 | 现象 | 处理 |
|---|---|---|
| HuggingFace 认证失败 | 下载权重/tokenizer 401 | 提前 `huggingface-cli login`；检查点本地缓存，避免重复下载 |
| Rust 组件编译慢 | `uv sync` 首次耗时较长 | 属预期行为，仅首次需要；保留编译缓存 |
| 网络受限 | 下载超时 | 使用代理/镜像源；toy 数据小，优先下载 |
| 显存不确定 | 担心 8GB 不够 | 用 150M 档 + 单批推理验证，实际占用远低于 8GB |

## 5. 自查与明日计划

- [x] 按说明可启动 `inference.py`（随机权重模式）；
- [x] 启动说明已形成文档，可复现；
- [ ] 明日：加载真实 150M 权重，主路径贯通，形成可演示版本（评测指标 + 对比录屏）。

> 备注：如教师调整栏目，按当次布置为准。
