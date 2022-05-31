python main-cv_sgd_best.py -ds cifar10 -pr 0.1 -mo cnn -lo lws -lw 1 -lr 0.05 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds cifar10 -pr 0.3 -mo cnn -lo lws -lw 1 -lr 0.05 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds cifar10 -pr 0.5 -mo cnn -lo lws -lw 1 -lr 0.05 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds mnist -pr 0.1 -mo mlp -lo lws -lw 2 -lr 0.05 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds mnist -pr 0.3 -mo mlp -lo lws -lw 1 -lr 0.1 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds mnist -pr 0.5 -mo mlp -lo lws -lw 1 -lr 0.1 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds cifar10 -pr 0.1 -mo cnn -lo lwc -lw 1 -lr 0.01 -wd 0.01 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds cifar10 -pr 0.3 -mo cnn -lo lwc -lw 2 -lr 0.01 -wd 0.01 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds cifar10 -pr 0.5 -mo cnn -lo lwc -lw 1 -lr 0.01 -wd 0.01 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds mnist -pr 0.1 -mo mlp -lo lwc -lw 2 -lr 0.1 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds mnist -pr 0.3 -mo mlp -lo lwc -lw 1 -lr 0.1 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

python main-cv_sgd_best.py -ds mnist -pr 0.5 -mo mlp -lo lwc -lw 2 -lr 0.1 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 250

