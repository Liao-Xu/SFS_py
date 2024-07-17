# Schrödinger-Föllmer Sampler (SFS) Demo

This repository demonstrates the Schrödinger-Föllmer Sampler (SFS) for high-dimensional Gaussian mixtures, optimized for high-performance computing.

## Overview

The SFS is implemented to sample from Gaussian mixture models across various dimensions, showcasing its scalability and effectiveness in high-dimensional spaces. This implementation leverages CUDA-enabled GPUs, PyTorch for efficient tensor operations, and batch computation to significantly boost performance.

## Key Features

- **CUDA Acceleration**: Utilizes GPU computing power when available.
- **PyTorch Integration**: Employs PyTorch for efficient tensor operations and automatic differentiation.
- **Batch Computation**: Implements batch processing to enhance throughput and reduce memory overhead.

## Key Files

- `demo_gauss_cuda.ipynb`: Main Jupyter notebook demonstrating SFS performance across dimensions.
- `func.py`: Contains core SFS implementation (`SFS_gauss`) with CUDA and PyTorch optimizations.
- `utils.py`: Utility functions for data generation and analysis.

## Main Functions

- `SFS_gauss`: Implements the Schrödinger-Föllmer Sampler for Gaussian mixtures, optimized for GPU computation.
- `generate_circle_alpha`: Generates evenly distributed points on a circle in high-dimensional space.
- `proportions_error`: Calculates the error between estimated and true cluster proportions.
- `kmeans_plot`: Visualizes clustering results compared to ground truth.

## Usage

Run the demo Jupyter notebook to see SFS performance across dimensions from 10 to 1000:

1. Ensure you have Jupyter Notebook or JupyterLab installed.
2. Open a terminal and navigate to the repository directory.
3. Launch Jupyter:
