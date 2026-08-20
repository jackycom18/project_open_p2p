# 《AI开发实践》实践报告（第 2 天）

| 项目 | 内容 |
|---|---|
| 课题名称 | 行为克隆游戏智能体（文本指令 + 推理性能方向） |
| 组名/成员 | 向子坚（独立完成） |
| 日期 | 2026-08-20 |
| 当日任务 | 方案组成与技术选型 |
| 阶段成果 | 系统组成清单 + 逐项选型理由 |

## 1. 今日目标

确定系统**由哪些部分组成**、各部分**承担什么职责**，并为每一项技术选型给出**明确理由**（为什么不选其他方案），为第 3 天的数据与调用约定、第 4 天的环境起步提供依据。

## 2. 系统组成

系统分三大部分，均在项目仓库根目录下：

```
project/
├── open-p2p-main/          # 官方参考实现（复用，不改动核心算法）
│   ├── elefant/policy_model/inference.py   # 推理服务器（模型入口）
│   ├── elefant/inference/unix_socket_server.py # 通信服务（UDS）
│   ├── elefant/policy_model/               # 模型网络、动作解码、KV Cache
│   ├── elefant/im_tokenizer/               # 图像分词器
│   ├── elefant/text_tokenizer/             # 文本分词器（Gemma）
│   ├── config/policy_model/150M.yaml       # 150M 模型配置
│   └── scripts/                            # 权重/数据下载脚本
├── eval/                   # 自研评测链路（本组核心增量）
│   ├── p2p_client.py          # 推理服务器通信客户端（基础模块）
│   ├── select_testset.py      # 从数据集中固定挑选 ≤200 帧测试集
│   ├── evaluate.py            # 单片段评测：按键准确率 / 鼠标相关 / 任务完成率
│   ├── run_instruct_experiment.py # 带指令 vs 不带指令对照实验
│   ├── benchmark_inference.py # 推理性能评测：KV Cache / 全量重算 / 关编译三组对比
│   ├── summarize.py           # 汇总指标表，对照验收线
│   └── record_demo.py         # 生成"人类标注 vs 模型输出"对比录屏
├── setup_wsl.sh / setup_wsl.ps1  # 环境一键部署脚本
└── 立项书_行为克隆游戏智能体_文本指令.md
```

职责划分：

- **open-p2p-main**：提供模型推理能力（只复用，不修改算法代码）；
- **eval/**：负责"出题、批分、做演示"，是本课题的增量工作量所在；
- **setup_wsl.\***：把环境安装固化成脚本，保证 M1"按 README 可复现"。

## 3. 技术选型及理由

| 选型 | 选择 | 理由（为什么不选其他） |
|---|---|---|
| 模型 | Open P2P **150M** 预训练权重 | 官方最小复现单元；单卡 8–16GB 显存可推理；比 300M+ 更利于在课程周期内完成多轮实验。不选更大模型：显存与耗时不可控 |
| 推理方式 | 官方 `inference.py` 启动的**推理服务器**（Unix Domain Socket 通信） | 官方原生支持，协议已定好（Frame/Action），无需自研通信；评测时一个进程服务多次请求，避免重复加载权重 |
| 文本指令 | 官方 `--input_text` 启动开关 | 架构原生支持（Gemma 文本分词 + embedding），与立项扩展方向一致。不选自行改造：改动大且违背"复用官方"原则 |
| 包管理 | **uv**（官方推荐） | 官方 pyproject.toml 已配好；可自动编译 Rust 组件（elefant_rust）；比 conda/pip 单独装更快、可复现性更强 |
| 依赖工具 | FFmpeg ≥7、Clang、Rust | 官方 README 指定（视频解码、编译依赖）；缺一不可，已写进 setup 脚本 |
| 评测框架 | 自研 Python 脚本（eval/） | 评测逻辑简单（逐帧对比），无需引入 pytest 等重量级框架；指标公式按立项书第六节实现 |
| 指标存储 | CSV + Markdown 表格 | 便于 summarize 汇总和写入实验报告；不选数据库，数据量小 |
| 演示 | `record_demo.py` 生成上下屏对比 mp4 | 满足 M4"同一片段上模型输出 vs 人类标注可视化对比"；不依赖实时游戏环境 |
| 推理性能评测 | `benchmark_inference.py` 对比 KV Cache / 全量重算 / 关编译三组 | 对应扩展 C；复用官方 `--use_full_inference` / `--no-compile` 开关，量化优化收益与按键一致率影响 |
| 部署目标 | Linux 服务器（NVIDIA RTX 5880 Ada 48GB）/ WSL2 | 推理需 NVIDIA GPU，Linux 驱动环境比 Windows 原生更稳 |

## 4. 架构图

```
游戏视频帧(192×192) ──┐
                      ├─► Open P2P 150M 策略模型 ──► 键盘动作 + 鼠标位移
文本指令(可选) ────────┘          │
                                 ▼
                    eval/evaluate.py
        按键准确率 / 鼠标相关系数 / 任务完成率
                    └────► summarize.py 汇总指标表
                    └────► record_demo.py 对比录屏
```

## 5. 关键决策记录

1. **复用优先**：模型、通信、分词全部复用官方代码，自研部分只做评测与实验编排，控制工作量与风险；
2. **离线优先**：评测基于录像帧（离线），避免实时链路（<50ms）的硬件门槛，满足 M4 即可；
3. **配置驱动**：模型参数（帧数、分辨率、Transformer 层数等）集中在 `config/policy_model/150M.yaml`，实验脚本只传 `--config`，保证可复现。

## 6. 自查与明日计划

- [x] 系统组成清单完整（模型/评测/部署三块）；
- [x] 每项选型均有理由；
- [ ] 明日：主路径的数据与调用约定、假实现边界；
- [x] 已确认：GPU 为 NVIDIA RTX 5880 Ada（48GB），远满足 150M 推理需求。

> 备注：如教师调整栏目，按当次布置为准。
