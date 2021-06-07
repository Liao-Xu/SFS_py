#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a demo for SFS on 2d simulation
"""

import numpy as np
import time

from func import SFS_2d
from utils import plot_2d

## Parameters setting
N = 5000
K = 100
kap = 8
p = 2

# unit = [-3,-1,1,3]
# alpha = [(x, y) for x in unit for y in unit] # 4 by 4 setting

## Mean and covariance initialize
rad = 8
alpha = [np.array((rad * np.cos(i*2*np.pi/(kap)), rad * np.sin(i*2*np.pi/(kap)))) for i in range(kap)]
sigma = [0.03*np.eye(2)] * kap
theta = np.array([1/kap] * kap)

## Start sampling
start = time.time()

res = np.zeros(N)
res = SFS_2d(N, p, kap, K, alpha, sigma, theta, parallel=6)

end = time.time()
print("All done at %.2f seconds" % (end - start))
    

plot_2d(res, N, K, alpha, sigma, theta, res_dir="results")