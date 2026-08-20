# 课题六执行步骤书：MVP 1–5 + 文本指令扩展

> 目标：在 Linux/WSL 上跑通 Open P2P 150M 模型，完成 5 项 MVP 验收，并完成「文本指令」扩展方向的对照实验。
> 每条命令都在 `~/open-p2p` 目录下执行，`uv run` 统一调用依赖环境。

---

## 快速开始（环境已就绪，直接复制执行）

> 前提：`checkpoints/150M/`、`dataset/` 已下载好，`eval/` 已同步到 `~/open-p2p/eval/`。

### 第 1 步：扫描数据集，拿到真实片段路径

```bash
cd ~/open-p2p
uv run python eval/select_testset.py --dataset dataset --output-dir eval_results
```

> 记下控制台里 episode 列的路径（如 `roblox-rivals/xxxxx`），下面所有 `<游戏>/<片段>` 都换成它，**不带 `dataset/` 前缀**。

### 第 2 步：生成冻结测试集（≤200 帧）

```bash
cd ~/open-p2p
uv run python eval/select_testset.py --dataset dataset --pick 6 --frames 200 --seed 42 --testset-out eval/testset.json
```

### 第 3 步：启动推理服务器（终端 1，一直开着）

```bash
cd ~/open-p2p
uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
```

### 第 4 步：单片段评测（另开终端 2）

```bash
cd ~/open-p2p
uv run python eval/evaluate.py --dataset dataset --episode <游戏>/<片段> --num-frames 200
```

### 第 5 步：汇总指标（M3 验收看这里）

```bash
cd ~/open-p2p
uv run python eval/summarize.py --input-dir eval_results --out eval_results/指标汇总.md
```

### 第 6 步：对比录屏（M4）

```bash
cd ~/open-p2p
uv run python eval/record_demo.py --dataset dataset --episode <游戏>/<片段> --num-frames 120 --out-dir demo
```

### 第 7 步：文本指令对照实验（扩展，自动启停服务器，无需手动开）

先编辑 `eval/testset.json`，给 2 个片段填 `instruction` 和 `task`（示例见文档 5.2 节），然后：

```bash
cd ~/open-p2p
uv run python eval/run_instruct_experiment.py \
  --dataset dataset \
  --testset eval/testset.json \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint-path checkpoints/150M/checkpoint-step=00500000.ckpt
```

### 附：手动验证带指令模式（终端输入指令）

```bash
cd ~/open-p2p
uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt \
  --input_text
```

> 启动后在终端敲一句话回车即生效（如 `向左走`）。

---

## 0. 总览：每个任务用哪个脚本

| 验收项 | 要求 | 使用的脚本/命令 |
|---|---|---|
| M1 | 跑通 150M 推理，README 可复现 | `inference.py`（随机权重 → 真实权重 → `--input_text`） |
| M2 | 测试集 ≤200 帧 + 评测脚本 | `select_testset.py` + `evaluate.py` |
| M3 | 基线：按键准确率 ≥55%、鼠标 r ≥0.5 | `evaluate.py` + `summarize.py` |
| M4 | 第 5 天对比录屏 | `record_demo.py` |
| M5 | 归档 + 指标表 + ≥3000 字报告 | `git` + `summarize.py` + 报告文档 |
| 扩展 | 2 个指令片段，带指令完成率 ≥60% vs 不带指令 ≤35% | `run_instruct_experiment.py` |
| 扩展 C | 推理性能：KV Cache vs 全量重算，报告耗时/帧率/一致性 | `benchmark_inference.py` |

---

## 1. 前置准备（一次性，约 1–2 小时）

### 1.1 一键环境搭建（推荐）

在 WSL 终端里执行项目自带脚本（会自动装 uv/FFmpeg/Clang/Rust、克隆仓库、`uv sync`、下载权重与数据）：

```bash
bash /mnt/c/Users/Xzj13/CodeBuddy/project/setup_wsl.sh
```

> 脚本默认会下载 150M 权重和 toy 数据。若只想先搭环境不下载，加环境变量：
> `SKIP_DOWNLOADS=1 bash /mnt/c/Users/Xzj13/CodeBuddy/project/setup_wsl.sh`

### 1.2 手动搭建（不用脚本时的等价命令）

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 系统依赖
sudo apt update
sudo apt install -y build-essential git nvtop htop socat

# FFmpeg 7
sudo add-apt-repository ppa:ubuntuhandbook1/ffmpeg7
sudo apt update
sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavdevice-dev libavfilter-dev

# Clang
sudo apt install -y clang libclang-dev

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 文件描述符上限
ulimit -n 65535
```

### 1.3 克隆仓库并安装依赖

```bash
cd ~
git clone https://github.com/elefant-ai/open-p2p.git ~/open-p2p
cd ~/open-p2p
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv sync
```

### 1.4 下载权重 + 数据 + HuggingFace 登录

```bash
cd ~/open-p2p

# 登录 HuggingFace（Gemma tokenizer 认证必需）
uv run huggingface-cli login

# 下载 150M 权重 -> checkpoints/150M/
uv run python scripts/download_checkpoints.py 150M

# 下载 toy 小样数据 -> dataset/
uv run python scripts/download_data.py --toy
```

### 1.5 同步自研评测脚本到仓库

```bash
cp -r /mnt/c/Users/Xzj13/CodeBuddy/project/eval ~/open-p2p/eval
```

**检查点**：`checkpoints/150M/` 里有 `model_config.yaml` 和 `checkpoint-step=00500000.ckpt`；`dataset/` 里有各游戏片段目录；`nvidia-smi` 能看到 GPU。

---

## 2. Day 1：M1 跑通推理（先随机权重，再真实权重）

### 2.1 随机权重验证链路（不依赖已下载的权重）

```bash
cd ~/open-p2p
uv run elefant/policy_model/inference.py \
  --config config/policy_model/150M.yaml \
  --use_random_weights
```

**检查点**：无 import/编译报错；启动后自动跑 FPS 测试并打印 `fps = ...`；监听 `/tmp/uds.recap` 成功。

### 2.2 加载真实 150M 权重（无指令模式）

```bash
cd ~/open-p2p
uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
```

**检查点**：日志显示 `Loading model from ...checkpoint-step=00500000.ckpt`，FPS 测试正常打印动作，无显存不足。

### 2.3 验证文本指令模式（扩展方向的 M1 前提）

```bash
cd ~/open-p2p
uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt \
  --input_text
```

**检查点**：启动后日志出现 `Starting terminal listener for text input`；在终端敲一句如 `向左走` 回车，日志出现 `Instruction updated to: '向左走'`。

> ⚠️ 关键约束：`--input_text` 必须**启动时**指定，运行中不能切换。带/不带指令是两次独立的服务器启动。

---

## 3. Day 2：M2 选测试集 + 评测脚本

### 3.1 扫描数据集，看清有哪些片段

```bash
cd ~/open-p2p
uv run python eval/select_testset.py --dataset dataset --output-dir eval_results
```

**检查点**：控制台打印 episode 列表（帧数、known%、按键分布），并生成 `eval_results/candidates.csv`。

> 📌 **episode 路径约定**：后续所有 `--episode` 参数，一律填「相对 dataset 的路径，**不含** `dataset/` 前缀」。例如扫描结果里显示 `dataset/roblox-rivals/xxx`，那 `--episode` 就填 `roblox-rivals/xxx`。

### 3.2 生成冻结测试集（≤200 帧、固定 seed、可复现）

```bash
cd ~/open-p2p
uv run python eval/select_testset.py \
  --dataset dataset \
  --pick 6 \
  --frames 200 \
  --seed 42 \
  --testset-out eval/testset.json
```

**检查点**：生成 `eval/testset.json`，每个单元 `num_frames ≤ 200`。之后所有实验都用这份冻结测试集，不要改动。

### 3.3 跑一次评测（M2 验证）

先开**终端 1** 启动无指令服务器：

```bash
cd ~/open-p2p
uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt
```

再开**终端 2** 跑评测：

```bash
cd ~/open-p2p
uv run python eval/evaluate.py \
  --dataset dataset \
  --episode <游戏>/<片段> \
  --num-frames 200 \
  --save-predictions eval_results/pred_<片段>.csv
```

**检查点**：控制台打印「按键准确率 / 鼠标相关系数 x/y/mean / 位移 MAE」，并生成 `eval_results/<tag>.csv`。

---

## 4. Day 3：M3 未微调基线评测

把测试集里所有片段逐一评测（终端 1 服务器保持运行，终端 2 逐个 `--episode` 跑），然后汇总：

```bash
cd ~/open-p2p
uv run python eval/summarize.py \
  --input-dir eval_results \
  --out eval_results/指标汇总.md
```

**检查点（对照验收线）**：

- 按键准确率 ≥ 55%（`summarize.py` 会直接打印「达标/未达标」）
- 鼠标相关系数 ≥ 0.5

> 若未达标：优先排查 (1) episode 相对路径是否正确 (2) 帧对齐 (3) 换标注更稳定的单一游戏片段。必要时补充 full-data 同款游戏帧（`download_data.py --start 1 --end 5`）。

---

## 5. Day 4：M4 对比录屏 + 文本指令扩展

### 5.1 生成「人类标注 vs 模型输出」对比录屏（M4）

终端 1 保持无指令服务器运行，终端 2：

```bash
cd ~/open-p2p
uv run python eval/record_demo.py \
  --dataset dataset \
  --episode <游戏>/<片段> \
  --start-frame 0 \
  --num-frames 120 \
  --instruction "向左移动" \
  --out-dir demo
```

**检查点**：生成 `demo/<片段>_0_120.mp4`，上半屏是录像帧，下半屏并排显示「人类标注 (GT)」和「模型输出 (Pred)」的按键与鼠标位移。

### 5.2 设计 2 个指令片段（扩展核心）

编辑 `eval/testset.json`，给其中 2 个测试单元填上 `instruction` 和 `task`。示例：

```json
{
  "testset": [
    {
      "episode": "roblox-rivals/0001_01_01_005",
      "start_frame": 0,
      "num_frames": 200,
      "instruction": "持续向左移动",
      "task": "move_left"
    },
    {
      "episode": "roblox-rivals/0001_01_01_006",
      "start_frame": 0,
      "num_frames": 200,
      "instruction": "跳跃后向右转向",
      "task": "jump_then_turn_right"
    }
  ]
}
```

> `task` 对应 `evaluate.py` 里 `check_instruction()` 已有的两套自动判定规则：
> - `move_left`：片段内累计左移 >50px 且按过左移键；
> - `jump_then_turn_right`：按顺序检测到「跳跃(Space) → 右移(RightArrow)」。
> 想加新规则，就照抄 `check_instruction()` 里的函数扩一个分支。

### 5.3 跑「带指令 vs 不带指令」对照实验

```bash
cd ~/open-p2p
uv run python eval/run_instruct_experiment.py \
  --dataset dataset \
  --testset eval/testset.json \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint-path checkpoints/150M/checkpoint-step=00500000.ckpt
```

**检查点**：脚本自动启动两组服务器（一组不带 `--input_text`，一组带），输出到 `eval_results/experiment_<时间戳>/`：
- `units_no_instr.csv` / `units_with_instr.csv`：逐片段指标
- `summary.csv` / `report.md`：带/不带指令完成率对比表

**对照验收线**：

| 指标 | 目标 |
|---|---|
| 带指令任务完成率 | ≥ 60% |
| 不带指令任务完成率 | ≤ 35% |
| 增益（带 − 不带） | ≥ +25 个百分点 |

---

## 5.5 扩展 C：推理性能评测（可选扩展）

### 5.5.1 原理

官方 `inference.py` 原生支持两种推理模式，用 `benchmark_inference.py` 对比三组配置：

| 配置 | 命令开关 | 含义 |
|---|---|---|
| kvcache（优化后） | 默认 | KV Cache 增量缓存，逐帧复用历史，最快 |
| full（优化前） | `--use_full_inference` | 全量重算，每次重算整段历史，慢 |
| no_compile（对照组） | `--no-compile` | 关闭 torch.compile，仍用 KV Cache |

### 5.5.2 运行命令

```bash
cd ~/open-p2p
uv run python eval/benchmark_inference.py \
  --dataset dataset \
  --episode <游戏>/<片段> \
  --num-frames 200 \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint-path checkpoints/150M/checkpoint-step=00500000.ckpt
```

### 5.5.3 输出与验收

- 输出到 `eval_results/benchmark_<时间戳>/`：`benchmark.csv`（明细）+ `report.md`（对比表 + 结论）；
- 脚本自动报告：**KV Cache 加速比**（full 耗时 / kvcache 耗时）与**按键一致率变化**（kvcache − full）。

**验收口径（扩展 C）**：

| 指标 | 目标 |
|---|---|
| KV Cache 加速比 | >1x（有可量化加速） |
| 按键一致率变化 | 无显著变化（|Δ| < 2%） |

> ⚠️ 注意：`--use_full_inference` 与 `--use_manual_sampling` 互斥，本实验不涉及后者。若想对比鼠标采样策略（`mean`/`conservative`/`truncated_normal`），需改 `model_config.yaml` 里的 `inference.mouse_sampling_approach` 字段，不在本脚本命令行范围内。

---

## 6. Day 5：M5 归档 + 指标表 + 报告

### 6.1 归档代码与结果到 Git

```bash
cd ~/open-p2p
git add eval/ eval_results/指标汇总.md demo/ reports/
git commit -m "课题六：MVP 1-5 与文本指令扩展实验归档"
```

### 6.2 指标表

`eval_results/指标汇总.md`（第 4 步生成）+ `eval_results/experiment_<时间戳>/report.md`（第 5.3 步生成）即为两份指标表，可直接并入实验报告。

### 6.3 实验报告（≥3000 字）

以 `reports/day09.md`（结课大报告模板）为骨架，填入真实指标数值，结构：对照立项 → 对照方案 → 对照验证（含指标表）→ 对照分工 → 总结展望。

---

## 7. 命令速查表

```bash
# 启动服务器（无指令）
uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt

# 启动服务器（带指令）
uv run elefant/policy_model/inference.py \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint_path checkpoints/150M/checkpoint-step=00500000.ckpt \
  --input_text

# 选测试集
uv run python eval/select_testset.py --dataset dataset --pick 6 --frames 200 --seed 42 --testset-out eval/testset.json

# 单片段评测
uv run python eval/evaluate.py --dataset dataset --episode <游戏>/<片段> --num-frames 200

# 汇总指标
uv run python eval/summarize.py --input-dir eval_results --out eval_results/指标汇总.md

# 对比录屏
uv run python eval/record_demo.py --dataset dataset --episode <游戏>/<片段> --num-frames 120 --out-dir demo

# 文本指令对照实验
uv run python eval/run_instruct_experiment.py \
  --dataset dataset --testset eval/testset.json \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint-path checkpoints/150M/checkpoint-step=00500000.ckpt

# 推理性能评测（扩展 C）
uv run python eval/benchmark_inference.py \
  --dataset dataset --episode <游戏>/<片段> --num-frames 200 \
  --config checkpoints/150M/model_config.yaml \
  --checkpoint-path checkpoints/150M/checkpoint-step=00500000.ckpt
```

---

## 8. 常见坑

| 坑 | 现象 | 处理 |
|---|---|---|
| HuggingFace 401 | 下载权重/tokenizer 失败 | `uv run huggingface-cli login` 后再下载 |
| episode 路径带 `dataset/` 前缀 | 报「片段目录不存在」 | `--episode` 只填相对 dataset 的路径（见 3.1） |
| config 不匹配模型大小 | 加载报错 | 推理用 `checkpoints/150M/model_config.yaml`；随机权重用 `config/policy_model/150M.yaml` |
| 服务器端口冲突 | `/tmp/uds.recap` 已存在 | 先停掉旧服务器进程，脚本会自动 unlink |
| 指令切换慢半拍 | 切换后前几帧仍用旧指令 | 调大 `--instruction-delay`（`instruct_player.py`） |
| 首次 `uv sync` 很慢 | Rust 组件编译 | 属预期，仅首次，保留缓存 |
| 指标不达标 | 按键率/相关不足 | 检查帧对齐、换稳定单一游戏片段、必要时补 full-data |

---

*本步骤书与立项书、`reports/` 各日报一致；所有命令均在 WSL/Ubuntu 24.04 + `~/open-p2p` 目录验证路径。*
