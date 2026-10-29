#pragma once
/// <summary>性能微基准入口：按环境变量 NN_BENCH 选择 matmul / conv / all。
/// 设计目标：单算子层级、同进程交替采样、报告 min/median/mean，避免 epoch 级噪声。</summary>
int RunBench(const char* op);
