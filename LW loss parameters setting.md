Parameter Settings:

batchsize = 256
Epoch = 500

We do not apply "beta decay method", thus we set 
decay_coef = 1
epoch_coef = 500
If you want to decay your beta as the epoch increases(e.g. from beta=2 to beta=1), you can change the two parameters. 

# MLP                                              
Fashion:   lr = 5e-2    weight_decay = 1e-3 
           q=0.1:beta=1   decay_rate=0.5   decay_step=50
           q=0.3:beta=1   decay_rate=0.5   decay_step=50 
           q=0.5:beta=2   decay_rate=0.5   decay_step=50 
           q=0.7:beta=4   decay_rate=0.5   decay_step=30 

MNIST:     lr=5e-2  weight_decay=1e-3   beta=2  decay_rate=0.5   decay_step=50

KMNIST:    beta=2  decay_rate=0.5  decay_step=50 
           q=0.1,0.3,0.5: lr=5e-2    weight_decay=1e-3
           q=0.7:         lr=1e-1    weight_decay=1e-3   

# ConvNet
CIFAR-10:  q=0.1,0.3,0.5:  lr = 5e-2  weight_decay = 1e-3  decay_rate=0.5  decay_step=50
           q=0.7: no results
