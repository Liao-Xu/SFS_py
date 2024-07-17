# Schrödinger-Föllmer Sampler for Gaussian Mixtures

This repository contains the Python implementation of the Schrödinger-Föllmer Sampler (SFS) for Gaussian Mixtures, including demonstrations in 1D, 2D, and high-dimensional settings.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Sampling on the 1D setting

To implement the sampling on 1D Gaussian Mixtures, run this command:

```train_toys_1d
python demo_1d.py
```

## Sampling on the 2D setting

To implement the sampling on 2D Gaussian Mixtures, run this command:

```train_toys_2d
python demo_2d.py
```

## High-Dimensional Sampling with CUDA Optimization

For high-dimensional Gaussian Mixtures sampling optimized with CUDA, we provide a Jupyter notebook:

- `demo_gauss_cuda.ipynb`: Demonstrates SFS performance across dimensions from 10 to 1000.

### Key Features of High-Dimensional Implementation

- **CUDA Acceleration**: Utilizes GPU computing power when available.
- **PyTorch Integration**: Employs PyTorch for efficient tensor operations.
- **Batch Computation**: Implements batch processing to enhance throughput and reduce memory overhead.

### Running the High-Dimensional Demo

1. Ensure you have Jupyter Notebook or JupyterLab installed.
2. Run the Jupyter notebook: `demo_gauss_cuda.ipynb`.
   
The notebook automatically detects and uses a CUDA-enabled GPU if available, falling back to CPU if not.

## Additional Files

- `high_dimension/func.py`: Contains core SFS implementation with CUDA and PyTorch optimizations.
- `high_dimension/utils.py`: Utility functions for data generation and analysis.

This demo showcases SFS's ability to efficiently sample from Gaussian mixtures in various dimensions, from 10 to 1000 dimension, leveraging batch computational techniques for optimal performance in high-dimensional spaces.

## Citation

If you use this implementation in your research, please cite our paper:

```bibtex
@article{huang2021schrodinger,
  title={Schr{\"o}dinger-F{\"o}llmer sampler: sampling without ergodicity},
  author={Huang, Jian and Jiao, Yuling and Kang, Lican and Liao, Xu and Liu, Jin and Liu, Yanyan},
  journal={arXiv preprint arXiv:2106.10880},
  volume={1},
  year={2021}
}
