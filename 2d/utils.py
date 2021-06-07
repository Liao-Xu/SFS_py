#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_2d(res, N, K, alpha, sigma, theta, res_dir = "results"):
    if not os.path.exists(res_dir):
            os.makedirs(res_dir)
    np.save(os.path.join(res_dir, 'N_'+str(N)+'_K_'+str(K)+'_data.npy'), res)
    
    fig, ax = plt.subplots(1, 1, figsize = (10, 10), dpi=200)
    h = sns.jointplot(x=res[0,:],y=res[1,:], kind='kde')
    h.ax_joint.set_xlabel('')
    h.ax_joint.set_ylabel('')
    plt.savefig(os.path.join(res_dir,'N_'+str(N)+'_K_'+str(K)+'_kde.png'))
    plt.show()