# 微基准脚本：同进程交替采样 CPU / GPU 各路径，报告 min/median/mean。
# 用法（在仓库根目录或任意位置运行）：
#   .\run_bench.ps1                # 默认 20 次矩阵乘迭代 / 6 次卷积迭代
#   .\run_bench.ps1 -Iters 50      # 更多采样
# 输出同时保存到 docs\bench\bench_<时间戳>.txt
param(
    [int]$Iters = 20,
    [int]$ConvIters = 6
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$exe = Join-Path $root "x64\Release\高等代数.exe"
if (-not (Test-Path $exe)) { Write-Error "未找到 $exe，请先构建 Release|x64"; exit 1 }

$outDir = Join-Path $root "docs\bench"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$file = Join-Path $outDir "bench_$stamp.txt"
"# 微基准 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $file -Encoding utf8

$env:NN_BENCH = "matmul"
$env:NN_BENCH_ITERS = "$Iters"
& $exe 2>&1 | Tee-Object -FilePath $file -Append

$env:NN_BENCH = "conv"
$env:NN_BENCH_ITERS = "$ConvIters"
& $exe 2>&1 | Tee-Object -FilePath $file -Append

Remove-Item Env:\NN_BENCH -ErrorAction SilentlyContinue
Remove-Item Env:\NN_BENCH_ITERS -ErrorAction SilentlyContinue
Write-Host "输出已保存到: $file"
