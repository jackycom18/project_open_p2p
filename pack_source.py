# -*- coding: utf-8 -*-
"""打包源码 zip：学号-姓名-课题名-源码.zip（≤10MB）。

包含：eval/（自研脚本 + Web 演示 + 启动器）、open-p2p-main/（官方代码，排除 banner）、
      reports/、立项书、步骤书、README、部署脚本。
排除：临时诊断脚本、临时截图/结果快照、答辩 PPT（单独提交）、重复压缩包。
"""
import os
import sys
import zipfile

# 待排除的临时文件（相对项目根）
EXCLUDE_FILES = {
    "eval/diag_mouse.py", "eval/diag_frame.py", "eval/recalc_metrics.py",
    "eval/stats_episodes.py", "eval/gen_launcher.py", "eval/make_ppt.py",
    "eval/check_ppt.py",
    "gen_launcher.py", "make_ppt.py", "check_ppt.py", "pack_source.py",
    "frame0.png", "frame500.png", "demo_frame1.png", "demo_frame2.png",
    "results_benchmark.csv", "results_benchmark_full.csv",
    "results_benchmark_report.md", "results_instruct.md",
    "立项书_行为克隆游戏智能体_文本指令.zip", "open-p2p.tar.gz",
    "课题六_行为克隆游戏智能体_答辩.pptx",
}

EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache"}

# 额外排除的大文件（与实验无关）
EXCLUDE_SUFFIX = ("banner.png",)


def main():
    student_id = sys.argv[1] if len(sys.argv) > 1 else "学号"
    name = sys.argv[2] if len(sys.argv) > 2 else "向子坚"
    topic = sys.argv[3] if len(sys.argv) > 3 else "行为克隆游戏智能体"

    out = f"{student_id}-{name}-{topic}-源码.zip"
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    include_dirs = ["eval", "open-p2p-main", "reports"]
    include_files = [
        "立项书_行为克隆游戏智能体_文本指令.md",
        "MVP与文本指令_执行步骤书.md",
        "README.md",
        "setup_wsl.sh",
        "setup_wsl.ps1",
        "p2p_demo.dat",
    ]

    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in include_dirs:
            for r, dirs, files in os.walk(d):
                dirs[:] = [x for x in dirs if x not in EXCLUDE_DIRS]
                for f in files:
                    rel = os.path.join(r, f).replace("\\", "/")
                    if rel in EXCLUDE_FILES or f in EXCLUDE_SUFFIX:
                        continue
                    z.write(os.path.join(r, f), rel)
                    total += 1
        for f in include_files:
            if os.path.exists(f):
                z.write(f, f)
                total += 1

    size_kb = os.path.getsize(out) / 1024
    print(f"已打包 {total} 个文件 -> {out}")
    print(f"大小 {size_kb:.1f} KB ({'OK' if size_kb < 10240 else '超10MB!'})")


if __name__ == "__main__":
    main()
