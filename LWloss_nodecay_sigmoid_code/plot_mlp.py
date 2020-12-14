#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 19:37:57 2020

@author: wenhongwei
"""

import os
import glob
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
method_seq = ["ours0905"]
dataset_seq = ["mnist", "kmnist", "fashionmnist"]#, "cifar10"]
rate = 0.5

for dataset in dataset_seq:
    if dataset in ["mnist", "kmnist", "fashion-mnist"]:
        model_seq = ["mlp"]
    elif dataset in ["cifar10"]:
        model_seq = ["cnn"]
    results = {}
    for model in model_seq:
        for method in method_seq:
            if method == "PRODEN":
                test_accs = np.zeros(shape=(501, 1))                
                for idx, seed in enumerate([505]):
                    log_path = "./results/plot_mlp/{}_{}_binomial_*_{}_{}.csv".format(dataset, model, method, seed)
                    log_path = glob.glob(log_path)[0]
                    print(log_path)
                    log = pd.read_csv(log_path)
                    test_accs[:, idx] = log["test_acc"].values         
                results[method] = test_accs.mean(axis=1)
            if method == "ours0905":
                test_accs = np.zeros(shape=(501, 1))
                for tradeoff in [0.0,0.5,1.0,2.0,4.0]:
                    for idx, seed in enumerate([505]):
                        log_path = "./results/plot_mlp/{}_{}_binomial_*_{}_{}_{}*.csv".format(dataset, model, method, tradeoff, seed)
                        log_path = glob.glob(log_path)[0]
                        print(log_path)
                        log = pd.read_csv(log_path)
                        test_accs[:, idx] = log["test_acc"].values          
                    results[method+str(int(tradeoff))] = test_accs.mean(axis=1)
        
        for method in method_seq:    
            x_axis = np.arange(501)
            ylim_max, ylim_min = -np.inf, np.inf
            for method in method_seq:
                if method == "PRODEN":
                    ylim_max = max(results[method].max(), ylim_max)
                    ylim_min = min(results[method].min(), ylim_min)
                    plt.plot(x_axis, results[method], label=method)
                if method == "ours0905":
                    for tradeoff in [0.0,0.5,1.0,2.0,4.0]:
                        ylim_max = max(results[method+str(int(tradeoff))].max(), ylim_max)
                        ylim_min = min(results[method+str(int(tradeoff))].min(), ylim_min)
                        plt.plot(x_axis, results[method+str(int(tradeoff))], label="tradeoff={}".format(tradeoff))
            plt.title("{}-{}-{}".format(dataset, model, rate))
            plt.legend()
                
            plt.ylim((1.0*ylim_min+2.0*ylim_max)/3.0, ylim_max+5)
            plt.savefig("./figs/{}-{}-{}.svg".format(dataset, model, rate))
            plt.savefig("./figs/{}-{}-{}.png".format(dataset, model, rate), dpi=200)
            plt.clf()