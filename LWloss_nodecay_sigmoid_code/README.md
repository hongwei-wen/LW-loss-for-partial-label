# PRODEN

This is the code for the paper: LEVERAGED WEIGHTED LOSS FOR PARTIAL LABEL LEARNING.
Paper link is https://openreview.net/pdf?id=DHkGKg2fJay

## Setups

All code was developed and tested on a single machine equiped with a NVIDIA Tesla V100 GPU. The environment is as bellow:
- Python 3.6.8
- Numpy 1.16.4
- Cuda 10.1.168

## Quick Start

Here is an example:
```
python main.py --tradeoff=1 --lr=5e-2 --weight_decay=1e-3 --partial_type=binomial --partial_rate=0.5 --batchsize=256 --n_epoch=501 --dataset=mnist --model=mlp --decay_step=50 --decay_rate=0.5 --seed=505

```
## Parameters settings 

Information about tuned parameters is in file './LW loss parameters setting.md'.
We only use the MLP and ConvNet in papers, but Linear and Resnet models are also implemented. 

## Results

The test results will be saved in './results/mnist_mlp_binomial_0.5_0.05_0.001_ours0905_1.0_505_0.5_50_1_500.csv' directory. 