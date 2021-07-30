# LW loss

This is the code for the paper: LEVERAGED WEIGHTED LOSS FOR PARTIAL LABEL LEARNING.
Paper link is https://arxiv.org/abs/2106.05731

## Setups

All code was developed and tested on a single machine equiped with a NVIDIA Tesla V100 GPU. The environment is as bellow:
- Python 3.6.8
- Numpy 1.16.4
- Cuda 10.1.168

## Quick Start

Here is an example:
```
python Task0119_lws01_01.py

```
## Parameters settings 

Information about tuned parameters is in file './LW loss parameters setting.md'.
We only use the MLP and ConvNet in papers, but Linear and Resnet models are also implemented. 

## Results

The test results will be saved in './results/mnist_mlp_binomial_0.5_0.05_0.001_ours0905_1.0_505_0.5_50_1_500.csv' directory. 