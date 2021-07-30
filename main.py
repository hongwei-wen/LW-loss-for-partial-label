import os
from itertools import product

ds = "mnist"
pr_type = "01"
mo = "mlp"

lw0 = 1
seed = 101
gpu = 1

lr_seq = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
wd_seq = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
lw_seq = [1, 2]

for lw, lr, wd in list(product(lw_seq, lr_seq, wd_seq))[:10]:
    os.system("python main-cv_sgd_best.py -ds {} -pr {} -mo {} -lo lws -lw0 {} -lw {} -lr {} -wd {} -ldr 0.5 -lds 50 -bs 256 -ep 250 -seed {} -gpu {}".format(ds, pr_type, mo, lw0, lw, lr, wd, seed, gpu))



