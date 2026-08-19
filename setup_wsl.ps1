# ============================================================
# Windows 侧：启用 WSL 并安装 Ubuntu 24.04
# 依据官方 README "Real-Time Game Inference (Windows)" 第 3 步：
#   wsl --install -d Ubuntu-24.04
# 必须【以管理员身份运行】本脚本，否则 wsl --install 无效。
# ============================================================
#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

Write-Host "[1/3] 启用 WSL 功能并安装 Ubuntu 24.04 ..." -ForegroundColor Cyan
wsl --install -d Ubuntu-24.04

Write-Host "[2/3] 配置 WSL 内存限制 (.wslconfig) ..." -ForegroundColor Cyan
# 官方 README 建议将 WSL 内存设为系统内存的较大比例（原文档示例 52GB）。
# 这里按本机总内存的 75% 自动生成，避免固定 52GB 超出笔记本内存导致 WSL 无法启动。
$totalGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$wslGB = [math]::Max(8, [math]::Floor($totalGB * 0.75))
$wslconfig = "$env:USERPROFILE\.wslconfig"
if (Test-Path $wslconfig) {
    Write-Host "已存在 $wslconfig，跳过写入。如需调整请手动编辑 (memory=${wslGB}GB 为建议值)" -ForegroundColor Yellow
} else {
    "[wsl2]`nmemory=${wslGB}GB" | Out-File -FilePath $wslconfig -Encoding ascii
    Write-Host "已写入 $wslconfig (memory=${wslGB}GB)" -ForegroundColor Green
}

Write-Host "[3/3] 完成。" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步：请【重启计算机】。" -ForegroundColor Cyan
Write-Host "重启后打开 Ubuntu 24.04 终端（首次启动需设置 Linux 用户名/密码），然后运行：" -ForegroundColor Yellow
Write-Host ""
Write-Host "    bash /mnt/c/Users/Xzj13/CodeBuddy/project/setup_wsl.sh" -ForegroundColor Green
Write-Host ""
Write-Host "该脚本会自动完成官方 README 的全部环境搭建、模型与数据下载。" -ForegroundColor Yellow
