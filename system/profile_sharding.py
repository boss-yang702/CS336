import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import AdamW
import time
import argparse
import sys
import functools

# 将项目根目录添加到Python路径，以允许导入 'system.shared_optimizer'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from system.shared_optimizer import ShardedOptimizer

# --- 模型定义: 一个用于分析的“XL”尺寸模型 ---
class LargeTransformer(nn.Module):
    """一个 'XL' 尺寸的模型，用于性能分析。"""
    def __init__(self, num_layers=24, d_model=1024, d_ff=4096, vocab_size=50257):
        super().__init__()
        # 为内存分析简化的Transformer层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = x + layer(x) # 残差连接
        x = self.lm_head(x)
        return x

# --- 分布式环境设置 ---
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501' # 使用一个不同的端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def print_memory_usage(rank, stage):
    # 只在rank 0上打印，避免信息重复
    if rank == 0:
        # torch.cuda.max_memory_allocated() 返回自上次重置以来的峰值内存
        peak_mem_gb = torch.cuda.max_memory_allocated(rank) / (1024**3)
        print(f"[{stage:^25s}] Rank 0 Peak Memory: {peak_mem_gb:.2f} GB")

# --- 工作进程函数 ---
def worker_fn(rank, world_size, args):
    setup(rank, world_size)
    
    # --- 模型和优化器设置 ---
    torch.cuda.reset_peak_memory_stats(rank)
    model = LargeTransformer().to(rank)
    
    # 同步初始模型参数
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    if args.mode == 'sharded':
        optimizer = ShardedOptimizer(model.parameters(), AdamW, lr=0.001)
    else: # standard
        optimizer = AdamW(model.parameters(), lr=0.001)

    print_memory_usage(rank, "After Model & Optimizer Init")

    # --- 用于性能分析的训练循环 ---
    input_ids = torch.randint(0, 50257, (args.batch_size, 128), device=rank)
    
    total_time = 0
    num_iters = 10
    warmup_iters = 2

    for i in range(num_iters):
        # 在热身迭代后重置统计数据，以获得更准确的测量结果
        if i == warmup_iters:
            torch.cuda.reset_peak_memory_stats(rank)
            total_time = 0

        iter_start_time = time.time()

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = outputs.sum() # 使用一个简单的伪损失
        loss.backward()

        # 梯度同步 (两种模式都需要)
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        
        if i == warmup_iters: print_memory_usage(rank, "Before Optimizer Step")
        optimizer.step()
        if i == warmup_iters: print_memory_usage(rank, "After Optimizer Step")

        if i >= warmup_iters:
            total_time += (time.time() - iter_start_time)

    if rank == 0:
        avg_iter_time = total_time / (num_iters - warmup_iters)
        print(f"\n--- Timings for '{args.mode}' mode ---")
        print(f"Average time per iteration: {avg_iter_time * 1000:.2f} ms")

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['sharded', 'standard'], help="Optimizer mode: 'sharded' or 'standard'")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size per GPU")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("此脚本需要至少2块GPU来运行。")
        sys.exit(1)
    
    mp.spawn(worker_fn, args=(world_size, args), nprocs=world_size, join=True)