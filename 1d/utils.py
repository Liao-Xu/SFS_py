#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import float_range
import os

def p_density(x, mu, var, weight):
    p1 = 1/(np.sqrt(2*np.pi*var[0]))*np.exp(-(x-mu[0])**2/(2*var[0]))
    p2 = 1/(np.sqrt(2*np.pi*var[1]))*np.exp(-(x-mu[1])**2/(2*var[1]))
    p3 = weight[0]*p1+(1-weight[0])*p2
    return p3

def plot_1d(res, N, K, alpha, sigma, theta, res_dir = "results"):
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    np.save(os.path.join(res_dir, 'N_'+str(N)+'_K_'+str(K)+'_data.npy'), res)
    x0 = np.array(list(float_range.range(alpha[0]-2,alpha[1]+4,0.01)))
    xnorm = p_density(x0, alpha, sigma, theta)
    
    fig, ax = plt.subplots(1, 1, figsize = (12, 8), dpi=300)
    ax.fill(x0, xnorm, alpha=0.25, color = "grey", label="Target")
    sns.distplot(res, hist = False, kde = True,
                     kde_kws = {'shade': False, 'linewidth': 3, 'alpha': 1,'bw':0.1, "linestyle": "-"}, label = "SFS", color = "tomato")
    ax.set_ylabel('Density', fontsize=22)    
    ax.set_xlabel('')
    # plt.rcParams.update({'font.size': 15})
    plt.legend(fontsize=22, loc='upper right')
    plt.savefig(os.path.join(res_dir,'N_'+str(N)+'_K_'+str(K)+'_kde.png'))
    plt.show()