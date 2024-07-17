#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def proportions_error(data, true_theta):
    """
    Perform k-means clustering on the provided data and calculate the error 
    between the estimated and true proportions of clusters.

    Parameters:
    - data (np.array): The data to cluster, should be an array of shape (p, N).
    - true_theta (np.array): The true proportions of each cluster.
    - num_clusters (int): The expected number of clusters (modes).

    Returns:
    - estimated_theta (np.array): The estimated proportions of each cluster.
    - error (float): The absolute error between the estimated and true proportions.
    """
    # Perform k-means clustering
    num_clusters = len(true_theta)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0).fit(data.T)
    labels = kmeans.labels_
    
    # Calculate estimated proportions
    estimated_theta = np.array([np.mean(labels == k) for k in range(num_clusters)])
    
    # Calculate error
    error = np.abs(estimated_theta - true_theta)
    
    return estimated_theta, error
    
def kmeans_plot(data, true_proportions):
    # Perform k-means clustering
    n_clusters = len(true_proportions)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    kmeans.fit(data.T)

    # Calculate cluster labels
    labels = kmeans.labels_

    # Calculate estimated proportions
    estimated_proportions = np.zeros(n_clusters)
    for label in labels:
        estimated_proportions[label] += 1
    estimated_proportions /= len(labels)

    # Plotting the proportions
    fig, ax = plt.subplots(figsize=(n_clusters, 2), dpi=300)
    index = np.arange(n_clusters)
    bar_width = 0.35

    rects1 = ax.bar(index, true_proportions, bar_width, label='Ground Truth')
    rects2 = ax.bar(index + bar_width, estimated_proportions, bar_width, label='Estimated by K-Means')

    ax.set_xlabel('Cluster')
    ax.set_ylabel('Proportions')
    ax.set_title('Proportions by Cluster: Ground Truth vs. K-Means Estimate')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([f'Cluster {i+1}' for i in range(n_clusters)])
    ax.legend()

    plt.show()
    
def generate_alpha(p, unit, rad):
    grid_values = [u * rad for u in unit]
    alpha = [np.array(vector) for vector in product(grid_values, repeat=p)]
    return alpha


def generate_circle_alpha(kap, p):
    """
    Generate points evenly distributed around a circle in p-dimensional space.
    
    :param kap: Number of modes to generate.
    :param p: Number of dimensions.
    :return: np.array of shape (n_points, p).
    """
    angles = np.linspace(0, 2 * np.pi, kap, endpoint=False)
    points = np.zeros((kap, p))
    for i in range(p // 2):  # Handle pairs of dimensions
        points[:, 2*i] = np.cos(angles)
        points[:, 2*i + 1] = np.sin(angles)
    if p % 2 == 1:  # Handle the last dimension if p is odd
        points[:, -1] = np.cos(angles)
    return points

