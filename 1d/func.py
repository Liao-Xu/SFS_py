#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 18:26:06 2021

@author: smoothl
"""
import numpy as np

N = 5000
K = 100
kap = 2
alpha = np.array([-6, 6])
sigma = np.array([0.25, 0.25])
theta = np.array([0.25, 0.75])



def SFS_1d(N, K, alpha, sigma, theta, parallel=False):
    def sample_1d(K, alpha, sigma, theta):
        def b(y, t, alpha, sigma, theta):
            b_de = np.zeros(2)
            b_nu = np.zeros(2)
            for i in range(2):
                alp = alpha[i]
                sig = sigma[i]
                
                sig_t = 1-t+t*sig
                b_de[i] = np.exp(-(t*alp**2+y**2-y**2*sig-2*y*alp)/(2*sig_t))/(sig_t**0.5)
                b_nu[i] = b_de[i]*(alp+y*(sig-1))/sig_t
            return np.matmul(b_nu, theta)/np.matmul(theta, b_de)
        
        y = np.zeros(K)
        s = 1/K
        for k in range(K-1):
            eps = np.random.normal(0, 1)
            y[k+1] = y[k] + s*b(y[k], k*s, alpha, sigma, theta) + np.sqrt(s)*eps
        return y[K-1]
    
    if parallel==False:
        res = np.zeros(N)
        for i in range(N):
            res[i] = sample_1d(K, alpha, sigma, theta)
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=6)(delayed(sample_1d)(K, alpha, sigma, theta) for i in range(N))
    return res

