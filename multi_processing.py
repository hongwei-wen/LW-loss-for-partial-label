
import os
from itertools import product
from multiprocessing import Pool

q_seq = [0.3]
tradeoff_seq = [1.0]
seed_seq = [505,404,303,202,101]
lr_seq = [0.05]
def func(tradeoff, seed, q, lr):
    os.system("python main.py --tradeoff={} --lr={} --weight_decay=1e-3 \
            --partial_type=binomial --result_dir=./results/cifar10 --partial_rate={}\
            --batchsize=256 --n_epoch=501 --dataset=cifar10\
            --model=cnn --decay_step=50 --decay_rate=0.5 --seed={}".format(tradeoff,lr, q, seed))

pool = Pool(processes=2)
for tradeoff, seed, q, lr in product(tradeoff_seq, seed_seq, q_seq, lr_seq):
    pool.apply_async(func, (tradeoff, seed, q, lr))
pool.close()
pool.join()

