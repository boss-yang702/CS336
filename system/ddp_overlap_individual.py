import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List
import time

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
# 2. 计算与通信重叠的DDP实现
# 这是本次作业的核心实现。
# ==============================================================================
class DDPOverlapIndividualParams:
    """
    一个DDP容器，它使用后向钩子来实现计算与通信的重叠。
    它会为模型中的每一个参数注册一个钩子。
    """
    def __init__(self, module: nn.Module):
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # 存储异步通信操作的句柄
        self.handles: List[tuple[dist.Work, torch.nn.Parameter]] = []

        # (第零步) 广播模型权重，确保所有模型的起点完全一致
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        # --- 核心逻辑: 注册后向钩子 ---
        # 遍历模型中的每一个可训练参数
        for p in self.module.parameters():
            if p.requires_grad:
                # 为参数 p 注册一个“后向钩子”。
                # 当p的梯度被计算并累加完毕后，下面的_hook函数会被自动调用。
                p.register_post_accumulate_grad_hook(self._hook(p))

    def _hook(self, p: torch.nn.Parameter):
        """
        这个函数返回一个闭包，该闭包将被用作后向钩子。
        这是为了在钩子内部能够访问到正确的参数p。
        """
        def hook_fn(*args, **kwargs):
            # 当钩子被触发时，我们立即为这个参数的梯度
            # 发起一个异步的 all_reduce 操作。
            # 程序不会在这里等待，而是会立刻返回，继续计算其他梯度。
            handle = dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            # 将返回的通信句柄存起来，以便后续等待
            self.handles.append((handle, p))
        return hook_fn

    def forward(self, *args, **kwargs):
        """
        前向传播很简单，直接调用被包装的模型的forward方法。
        """
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """
        这是一个关键的同步点。
        在优化器执行step()之前，必须调用此方法。
        """
        # 遍历所有之前存储的通信句柄
        for handle, p in self.handles:
            # handle.wait() 会阻塞程序，直到这个特定的异步通信完成。
            handle.wait()
            # --- 新增：在梯度同步完成后，对其进行平均 ---
            if p.grad is not None:
                p.grad.data /= self.world_size
        # 清空句柄列表，为下一次迭代做准备
        self.handles.clear()

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

def worker_fn(rank, world_size):
    """每个GPU进程执行的工作函数。"""
    print(f"开始运行 Rank {rank}...")
    setup(rank, world_size)

    # 在每个GPU上初始化模型和优化器
    model = ToyModel().to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # --- 使用我们实现的DDP类来包装模型 ---
    ddp_model = DDPOverlapIndividualParams(model)
    print(f"Rank {rank}: DDP模型已初始化。")

    # 模拟训练循环
    for i in range(20):
        # 创建数据
        inputs = torch.randn(20, 10).to(rank)
        labels = torch.randint(0, 5, (20,)).to(rank)

        # 正常的前向传播
        optimizer.zero_grad()
        outputs = ddp_model.forward(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # 正常的后向传播
        # 在这个函数执行期间，我们注册的钩子会被自动触发
        loss.backward()
        
        # --- 在优化器更新之前，必须等待所有梯度同步完成 ---
        ddp_model.finish_gradient_synchronization()
        
        # 正常的优化器步骤
        optimizer.step()

        if rank == 0:
            print(f"Rank {rank}, Iter {i}: Loss = {loss.item():.4f}")
        
        # 简单的延时，以便观察输出
        time.sleep(0.1)

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
        print(f"将在所有 {world_size} 块GPU上运行...")
        mp.spawn(worker_fn, args=(world_size,), nprocs=world_size, join=True)
