import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit

# --- 占位符：请用您在作业一中实现的模型和优化器替换 ---
# 为了让脚本能独立运行，这里我们使用一个简单的玩具模型和优化器
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

# --- 分布式设置与工作进程 ---

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """销毁进程组"""
    dist.destroy_process_group()

def naive_ddp_worker(rank, world_size):
    """每个GPU进程执行的工作函数"""
    print(f"开始运行 Rank {rank}.")
    setup(rank, world_size)

    # 1. 在每个GPU上初始化模型和优化器
    model = ToyModel().to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 2. (第零步) 广播模型权重，确保所有模型的起点完全一致
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    print(f"Rank {rank}: 模型已同步。")

    # 模拟训练循环
    for i in range(10): # 运行10次迭代
        # 3. (第一步) 创建数据分片
        # 在实际应用中，您会使用分布式采样器来加载数据
        # 这里我们为每个GPU生成独立的随机数据
        inputs = torch.randn(20, 10).to(rank)
        labels = torch.randint(0, 5, (20,)).to(rank)

        # 4. (第二步) 独立计算：前向和后向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        
        # --- 这是“朴素”DDP的核心 ---
        # 5. (第三步) 梯度同步：在后向传播结束后，逐个对梯度进行all-reduce
        for param in model.parameters():
            if param.grad is not None:
                # 将所有GPU上的梯度相加，然后除以GPU数量得到平均值
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        # --------------------------------

        # 6. (第四步) 同步更新
        optimizer.step()

        if rank == 0:
            print(f"Rank {rank}, Iter {i}: Loss = {loss.item()}")

    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(naive_ddp_worker, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("需要至少2块GPU来运行此DDP示例。")