# Quantum Generative Adversarial Network (qGAN)

A novel quantum generative adversarial network implementation using PennyLane for industrial bioengineering process synthetic time series data generation. This repository contains the code for a Wasserstein GAN with Gradient Penalty utilizing quantum circuit as the generator component.

## Overview

This implementation features:
- **Quantum Generator**: Parameterized quantum circuit with IQP encoding and strongly entangled layers
- **Classical Discriminator**: Wasserstein GAN with gradient penalty for stable training
- **Time Series Focus**: Optimized for industrial bioprocess time series data
- **PennyLane Integration**: Built on PennyLane quantum computing framework with PyTorch backend

## Citation

Based on the work presented in Orlandi et al. (2023): [Enhancing Financial Time Series Prediction with Quantum-Enhanced Synthetic Data Generation: A Case Study on the S&P 500 Using a Quantum Wasserstein Generative Adversarial Network Approach with a Gradient Penalty](https://www.mdpi.com/2079-9292/13/11/2158)

Github: (https://github.com/EBarbierato/Enhancing-Financial-Time-Series-Prediction-)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/shawngibford/qGAN.git
cd qGAN
```

### 2. Create virtual environment
```bash
python -m venv qgan_env
source qgan_env/bin/activate  # On Windows: qgan_env\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook (Recommended)
```bash
jupyter notebook qgan_pennylane.ipynb
```


## Key Features

- **Quantum Circuit Architecture**: 5-qubit generator with 45 trainable parameters
- **Hybrid Training**: Classical-quantum optimization loop
- **Data Preprocessing**: Lambert W transformation and normalization
- **Performance Metrics**: Earth Mover Distance, statistical comparisons, visual analysis
- **Reproducible Results**: Fixed random seeds and comprehensive logging

## Requirements

- Python ≥ 3.8
- PennyLane ≥ 0.32.0
- PyTorch ≥ 2.0.0
- NumPy, Pandas, Matplotlib, SciPy

See `requirements.txt` for complete dependencies.

## Data

The included `data.csv` contains time series data for training and evaluation. The model is designed for univariate time series with preprocessing for optimal quantum circuit performance.

## License

See `LICENSE` for licensing information.

## Contact

