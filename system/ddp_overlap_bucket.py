import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Dict
import time
import argparse

# ==============================================================================
# 1. 模型定义区域
# 这里使用一个简单的玩具模型来演示和验证DDP类的功能。
# ==============================================================================
class ToyModel(nn.Module):
    """一个简单的玩具模型，用于功能验证。"""
    def __init__(self):
        super().__init__()
        # 创建几个线性层来模拟一个更深的模型的梯度计算顺序
        self.layers = nn.ModuleList([
            nn.Linear(10, 20), nn.ReLU(),
            nn.Linear(20, 20), nn.ReLU(),
            nn.Linear(20, 10), nn.ReLU(),
            nn.Linear(10, 5)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ==============================================================================
# 2. 分桶DDP的实现 (最终优化方案)
# 这是本次作业的核心实现。
# ==============================================================================
class DDPOverlapBucketed:
    """
    一个DDP容器，它使用梯度分桶和后向钩子来实现最优的计算与通信重叠。
    """
    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0):
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        
        # --- 核心逻辑 1: 参数分桶 ---
        self.buckets: List[List[nn.Parameter]] = []
        self.param_to_bucket_map: Dict[nn.Parameter, int] = {}
        
        # 梯度是以后向传播的逆序计算的，所以我们反向遍历参数以获得最佳分桶顺序
        params_in_reverse = list(self.module.parameters())[::-1]
        
        current_bucket = []
        current_bucket_size = 0
        for p in params_in_reverse:
            if p.requires_grad:
                param_size = p.numel() * p.element_size()
                # 如果当前桶非空且加入新参数会超限，则开启新桶
                if current_bucket and current_bucket_size + param_size > self.bucket_size_bytes:
                    self.buckets.append(current_bucket)
                    current_bucket = []
                    current_bucket_size = 0
                
                current_bucket.append(p)
                current_bucket_size += param_size
        
        # 添加最后一个桶
        if current_bucket:
            self.buckets.append(current_bucket)

        # --- 核心逻辑 2: 准备钩子和状态追踪 ---
        self.bucket_grad_ready_counts: List[int] = []
        self.handles: List[dist.Work] = []
        
        # 为每个桶初始化一个计数器
        self.bucket_grad_ready_counts = [len(b) for b in self.buckets]
        
        # 创建一个从参数到其所在桶索引的快速查找映射
        for i, bucket in enumerate(self.buckets):
            for p in bucket:
                self.param_to_bucket_map[p] = i

        # (第零步) 广播模型权重
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        # --- 核心逻辑 3: 注册后向钩子 ---
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._hook(p))
    
    def _hook(self, p: nn.Parameter):
        """返回一个为特定参数p定制的钩子函数。"""
        def hook_fn(*args, **kwargs):
            bucket_idx = self.param_to_bucket_map[p]
            # 将对应桶的“待处理梯度”计数减一
            self.bucket_grad_ready_counts[bucket_idx] -= 1
            # 如果这个桶的所有梯度都已准备好，则触发通信
            if self.bucket_grad_ready_counts[bucket_idx] == 0:
                self._reduce_bucket(bucket_idx)
        return hook_fn

    def _reduce_bucket(self, bucket_idx: int):
        """对一个已准备好的桶进行打包、通信和解包。"""
        bucket = self.buckets[bucket_idx]
        
        # 1. 打包/拉平: 将桶内所有梯度打包成一个连续的大张量
        flat_grad = torch._utils._flatten_dense_tensors([p.grad.data for p in bucket])
        
        # 2. 异步通信: 对这个大张量发起一个异步的all_reduce
        handle = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)

        # 3. 注册回调: 使用.then()注册一个回调函数，该函数将在通信完成后被自动执行
        def unflatten_and_copy(fut):
            # fut.value()[0] 是all_reduce求和后的扁平化张量
            reduced_flat_grad = fut.value()[0]
            reduced_flat_grad /= self.world_size # 求平均
            
            # 解包: 将扁平化张量恢复成与原始梯度形状相同的张量列表
            unflattened_grads = torch._utils._unflatten_dense_tensors(reduced_flat_grad, [p.grad.data for p in bucket])
            
            # 拷贝: 将解包后的梯度拷贝回每个参数的.grad属性
            for p, new_grad in zip(bucket, unflattened_grads):
                p.grad.data.copy_(new_grad)

        handle.get_future().then(unflatten_and_copy)

    def forward(self, *args, **kwargs):
        """前向传播直接调用被包装的模型的forward方法。"""
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """在优化器步骤之前，等待所有已发起的通信完成。"""
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        # 为下一次迭代重置计数器
        self.bucket_grad_ready_counts = [len(b) for b in self.buckets]

# ==============================================================================
# 3. 分布式环境设置与工作进程函数
# ==============================================================================
def setup(rank, world_size):
    """初始化分布式环境。"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """销毁进程组。"""
    dist.destroy_process_group()

def worker_fn(rank, world_size, args):
    """每个GPU进程执行的工作函数。"""
    print(f"开始运行 Rank {rank}...")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # --- 使用我们实现的DDP分桶类来包装模型 ---
    ddp_model = DDPOverlapBucketed(model, bucket_size_mb=args.bucket_size_mb)
    print(f"Rank {rank}: DDP分桶模型已初始化，桶大小: {args.bucket_size_mb}MB。")

    # 模拟训练循环
    for i in range(20):
        inputs = torch.randn(20, 10).to(rank)
        labels = torch.randint(0, 5, (20,)).to(rank)

        optimizer.zero_grad()
        outputs = ddp_model.forward(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        
        ddp_model.finish_gradient_synchronization()
        
        optimizer.step()

        if rank == 0:
            print(f"Rank {rank}, Iter {i}: Loss = {loss.item():.4f}")
        
        time.sleep(0.1)

    cleanup()
    print(f"Rank {rank} 已完成。")

# ==============================================================================
# 4. 主程序入口
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bucket_size_mb', 
        type=float, 
        default=25.0, 
        help="梯度分桶的大小 (MB)"
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("此脚本需要至少2块GPU来运行。")
    else:
        print(f"将在所有 {world_size} 块GPU上运行...")
        mp.spawn(worker_fn, args=(world_size, args), nprocs=world_size, join=True)
