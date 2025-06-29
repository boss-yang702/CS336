#!/bin/bash

# ==============================================================================
# CS336 Assignment 1 - TinyStories 训练启动脚本
#
# 这个脚本使用作业PDF 7.2节中指定的参数来运行您的训练代码。
#
# 使用方法:
# 1. 将此内容保存为一个名为 `run_training.sh` 的文件。
# 2. 确保您的训练代码文件（例如 `train.py`）与此脚本位于同一目录。
# 3. 在终端中给予此脚本执行权限: `chmod +x run_training.sh`
# 4. 运行脚本: `./run_training.sh`
# ==============================================================================

echo "开始使用TinyStories配置进行训练..."

python -m basics.train.py \
    --dataset_name "tinystory" \
    --vocab_size 10000 \
    --context_length 256 \
    --batch_size 16 \
    --d_model 512 \
    --num_layers 4 \
    --num_heads 16 \
    --d_ff 1344 \
    --total_iters 20000 \
    --max_learning_rate 5e-4 \
    --cosine_cycle_iters 20000 \
    --weight_decay 0.001 \
    --wandb_logging True \
    --wandb_project "cs336-assignment1" \
    --wandb_run_name "tinystories-baseline" \
    --eval_interval 200 \
    --log_interval 20
    
echo "训练脚本执行完毕。"