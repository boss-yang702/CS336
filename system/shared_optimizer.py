import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Optimizer, AdamW
from typing import List, Dict, Any, Type, Iterator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
# ==============================================================================
# 1. 模型定义区域
# 这里使用一个简单的玩具模型来演示和验证分片优化器的功能。
# ==============================================================================
class ToyModel(nn.Module):
    """一个简单的玩具模型，用于功能验证。"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 20, bias=False), nn.ReLU(),
            nn.Linear(20, 20, bias=False), nn.ReLU(),
            nn.Linear(20, 10, bias=False), nn.ReLU(),
            nn.Linear(10, 5, bias=False)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ==============================================================================
# 2. 优化器状态分片 (ShardedOptimizer) 的实现
# 这是本次作业的核心实现。
# ==============================================================================
class ShardedOptimizer(Optimizer):
    """
    一个实现了优化器状态分片（ZeRO-1）的优化器包装类。
    它将模型参数分配给不同的rank，每个rank只维护一部分参数的优化器状态。
    """
    def __init__(self, params: Iterator[nn.Parameter], optimizer_cls: Type[Optimizer], **kwargs: Any):
        # 1. 获取分布式环境信息
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # 2. 参数分片：将所有可训练参数分组，并为每个rank分配一个分片
        self.param_groups = list(params) # 将迭代器转换为列表
        if not isinstance(self.param_groups[0], dict):
             # 如果传入的不是标准的参数组字典，为其创建一个
            self.param_groups = [{'params': self.param_groups}]

        # 遍历每个参数组，进行分片
        for group in self.param_groups:
            group['original_params'] = list(group['params']) # 保存原始顺序的完整参数列表
            # 确保 'params' 是一个列表
            group['params'] = list(group['params'])
            
            # 这是我们为当前rank分配到的参数
            sharded_params = []
            # 这是其他rank负责的参数
            non_sharded_params = []
            for i, p in enumerate(group['params']):
                if i % self.world_size == self.rank:
                    sharded_params.append(p)
            
            # 更新参数组，使其只包含当前rank负责的参数
            group['params'] = sharded_params
        
        # 3. 初始化内部的、真正的优化器
        # 这个优化器只会看到被分配到当前rank的参数子集
        self.optimizer = optimizer_cls(self.param_groups, **kwargs)
        
        # 调用PyTorch优化器基类的构造函数，这是必需的
        # 我们传递一个空的参数列表，因为实际的参数已经由内部优化器管理
        super().__init__([], {})

    def step(self, closure=None):
        """
        执行一步优化。这个方法包含两个核心步骤：
        1. 本地更新：调用内部优化器，只更新本地分片上的参数。
        2. 全局同步：将更新后的本地参数广播给所有其他rank。
        """
        # --- 步骤1: 本地更新 ---
        # 调用内部优化器的step，它只会更新当前rank负责的那一部分参数
        self.optimizer.step(closure)
        
        # --- 步骤2: 全局同步 ---
        # 遍历每个参数组中的所有原始参数，并将更新后的值从其属主rank广播出去
        for group in self.param_groups:
            # 对组内的每一个参数进行广播
            for i, p in enumerate(group['original_params']):
                # 确定哪个rank拥有这个参数的最新版本 (即负责更新它的rank)
                src_rank = i % self.world_size
                # 从源rank广播更新后的参数给所有其他rank
                dist.broadcast(p.data, src=src_rank)

    # 其他需要重写的方法
    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def add_param_group(self, param_group: Dict[str, Any]):
        # 作业要求实现此方法
        # 这里的逻辑与__init__中的分片逻辑类似
        sharded_params = []
        original_params = list(param_group['params'])
        for i, p in enumerate(original_params):
            if i % self.world_size == self.rank:
                sharded_params.append(p)
        
        # 创建一个新的参数组给内部优化器
        new_group_for_optimizer = {k: v for k, v in param_group.items() if k != 'params'}
        new_group_for_optimizer['params'] = sharded_params
        self.optimizer.add_param_group(new_group_for_optimizer)
        # 在ShardedOptimizer层面也记录这个组
        new_group_for_self = new_group_for_optimizer.copy()
        new_group_for_self['original_params'] = original_params
        self.param_groups.append(new_group_for_self)


# ==============================================================================
# 3. 分布式环境设置与工作进程函数
# ==============================================================================
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def worker_fn(rank, world_size):
    """每个GPU进程执行的工作函数。"""
    print(f"开始运行 Rank {rank}...")
    setup(rank, world_size)

    # 在每个GPU上初始化一个完整的模型副本
    model = ToyModel().to(rank)

    # --- 使用我们实现的分片优化器 ---
    # 它包装了一个标准的AdamW优化器
    optimizer = ShardedOptimizer(model.parameters(), AdamW, lr=0.01)
    
    print(f"Rank {rank}: ShardedOptimizer 已初始化。")

    # 模拟训练循环
    for i in range(10):
        # DDP的梯度同步 (all_reduce)
        # 注意：在ZeRO-1中，梯度仍然是需要在所有GPU上求平均的
        inputs = torch.randn(20, 10).to(rank)
        labels = torch.randint(0, 5, (20,)).to(rank)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        
        # 在ZeRO-1中，梯度是完整的，需要先进行all-reduce
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        
        # --- 调用分片优化器的step ---
        optimizer.step()

        if rank == 0:
            print(f"Rank {rank}, Iter {i}: Loss = {loss.item():.4f}")
        
        time.sleep(0.2)

    cleanup()
    print(f"Rank {rank} 已完成。")

# ==============================================================================
# 4. 主程序入口
# ==============================================================================
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("此脚本需要至少2块GPU来运行。")
    else:
        mp.spawn(worker_fn, args=(world_size,), nprocs=world_size, join=True)
