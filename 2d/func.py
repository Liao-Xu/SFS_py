#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

def SFS_2d(N, p, kap, K, alpha, sigma, theta, parallel=False):
    stats = [[] for i in range(3)]
    for i in range(kap):
        alp = alpha[i]
        sig = sigma[i]
    
        stats[0].append(np.linalg.inv(sig))                    # sigma^(-1)
        stats[1].append(np.matmul(stats[0][i], alp))           # sigma^(-1)*alpha
        stats[2].append(np.matmul(alp.T, stats[1][i]))         # alpha^T*sigma^(-1)*alpha
        
    def sample_2d(p, kap, K, alpha, sigma, theta, stats):
        def b(y, p, kap, t, alpha, sigma, theta, stats):
            Ip = np.eye(p)
            b_de = np.zeros(kap)
            b_nu = np.zeros((p, kap))
            for i in range(kap):
                alp = alpha[i]
                sig = sigma[i]
            
                inv_sig = stats[0][i]                              # sigma^(-1)
                is_a = stats[1][i]                                 # sigma^(-1)*alpha
                at_is_a = stats[2][i]                              # alpha^T*sigma^(-1)*alpha
                t_is = (1-t)*inv_sig                               # (1-t)*sigma
                t_is_y = np.matmul(t_is, alp) + y                  # (1-t)*sigma*alpha + y
                com_I = np.linalg.inv(t*Ip + t_is)                 # ((1-t)*sigma+t*I)^(-1)
                com_sig = np.linalg.det(t*sig + (1-t)*Ip)**0.5     # (t*sigma+(1-t)*I)^(-0.5)
            
                g = np.exp((np.sum((np.matmul(com_I**0.5,(np.matmul(t_is, alp)+y)))**2)-np.sum(y**2))/(2-2*t) - 0.5*at_is_a)
                b_de[i] = g/com_sig
                b_nu[:,i] = (is_a + np.matmul(np.matmul((Ip-inv_sig), com_I), t_is_y))*b_de[i]
            return np.dot(b_nu, theta)/np.sum(np.dot(theta, b_de))
        
        y = np.zeros((p, K))
        s = 1/K
        for k in range(K-1):
            eps = np.random.multivariate_normal([0]*p, np.eye(p))
            y[:,(k+1)] = y[:,k] + s*b(y[:,k], p, kap, k*s, alpha, sigma, theta, stats) +  np.sqrt(s)*eps
        return y[:, K-1]
    
    if parallel==False:
        res = np.zeros((p, N))
        for i in range(N):
            res[:, i] = sample_2d(p, kap, K, alpha, sigma, theta, stats)
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=6)(delayed(sample_2d)(p, kap, K, alpha, sigma, theta, stats) for i in range(N))
        res = np.array(res).T
    return res