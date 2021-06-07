#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import time
from func import SFS_1d
from utils import plot_1d


N = 5000
K = 100
kap = 2
alpha = np.array([-4, 4])
sigma = np.array([0.25, 0.25])
theta = np.array([0.75, 0.25])

start = time.time()

res = np.zeros(N)

res = SFS_1d(N, K, alpha, sigma, theta, parallel=6)

end = time.time()
print("All done at %.2f seconds" % (end - start))

plot_1d(res, N, K, alpha, sigma, theta, res_dir="results")