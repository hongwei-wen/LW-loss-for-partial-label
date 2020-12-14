#!/bin/bash

### 给你这个作业起个名字，方便识别不同的作业
#SBATCH --job-name=no_beta

### 指定该作业需要多少个节点
#SBATCH --nodes=1

### 指定该作业需要多少个CPU
#SBATCH --ntasks=5

### 指定该作业在哪个队列上执行
### 目前可用的GPU队列有 titan/tesla
#SBATCH --partition=tesla

### 申请一块GPU卡
#SBATCH --gres=gpu:1 

### 执行你的作业
source activate pytorch-1.3.0
python main.py --tradeoff=1 --lr=5e-2 --weight_decay=1e-3 --partial_type=binomial --partial_rate=0.5 --batchsize=256 --n_epoch=501 --dataset=mnist --model=mlp --decay_step=50 --decay_rate=0.5 --seed=505

