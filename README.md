# LW loss

This is the code for the paper: LEVERAGED WEIGHTED LOSS FOR PARTIAL LABEL LEARNING.
Paper link is https://arxiv.org/abs/2106.05731

## Update

We have updated the code for LW loss with cross entropy (LWC), and the best parameters for both losses on the CIFAR-10 and MNIST datasets.

## Setups

All code was developed and tested on a single machine equiped with a NVIDIA Tesla V100 GPU. The environment is as bellow:
- Python 3.6.8
- Numpy 1.16.4
- Cuda 10.1.168

## Quick Start

Here is a quick start on the CIFAR-10 and MNIST datasets.

For LW loss with sigmoid on CIFAR-10:

```
python main-cv_sgd_best.py -ds cifar10 -pr 0.1 -mo cnn -lo lws -lw 1 -lr 0.05 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds cifar10 -pr 0.3 -mo cnn -lo lws -lw 1 -lr 0.05 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds cifar10 -pr 0.5 -mo cnn -lo lws -lw 1 -lr 0.05 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

```
For LW loss with sigmoid on MNIST:

```
python main-cv_sgd_best.py -ds mnist -pr 0.1 -mo mlp -lo lws -lw 2 -lr 0.05 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds mnist -pr 0.3 -mo mlp -lo lws -lw 1 -lr 0.1 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds mnist -pr 0.5 -mo mlp -lo lws -lw 1 -lr 0.1 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

```

For LW loss with cross entropy on CIFAR-10:

```
python main-cv_sgd_best.py -ds cifar10 -pr 0.1 -mo cnn -lo lwc -lw 1 -lr 0.01 -wd 0.01 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds cifar10 -pr 0.3 -mo cnn -lo lwc -lw 2 -lr 0.01 -wd 0.01 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds cifar10 -pr 0.5 -mo cnn -lo lwc -lw 1 -lr 0.01 -wd 0.01 -ldr 0.5 -lds 50 -bs 256 -ep 250

```

For LW loss with cross entropy on MNIST:

```
python main-cv_sgd_best.py -ds mnist -pr 0.1 -mo mlp -lo lwc -lw 2 -lr 0.1 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds mnist -pr 0.3 -mo mlp -lo lwc -lw 1 -lr 0.1 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds mnist -pr 0.5 -mo mlp -lo lwc -lw 2 -lr 0.1 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

```

More parameter grids can be found in file './main.py'.

We only use the MLP and ConvNet in papers, but Linear and Resnet models are also implemented. 

