# PennyLane Quantum GAN for Industrial Bioprocess Synthetic Data Generation

## Overview

This repository implements a Quantum Generative Adversarial Network (QGAN) using PennyLane and PyTorch for generating synthetic time series data for industrial bioprocesses. The implementation leverages quantum computing principles to model complex temporal patterns in bioprocess data, enabling data augmentation and privacy-preserving synthetic data generation for industrial applications.

## Features

- **Quantum Generator**: Parameterized quantum circuit (PQC) with alternating rotation and entanglement layers
- **Classical Discriminator**: Deep neural network for distinguishing real from synthetic data
- **WGAN-GP Training**: Wasserstein GAN with gradient penalty for stable training
- **Comprehensive Analysis**: Statistical metrics including EMD, ACF, volatility clustering, and leverage effects
- **Visualization Suite**: Extensive plotting capabilities for data comparison and model evaluation
- **Time Series Preprocessing**: Lambert W transformation and normalization for improved training stability

## Scientific Context

This implementation addresses the critical need for synthetic data generation in industrial bioprocesses, where:
- Real process data is often limited and proprietary
- Data privacy and intellectual property concerns restrict data sharing
- Machine learning models require large datasets for robust training
- Process optimization benefits from diverse scenario modeling

The quantum approach offers potential advantages in modeling complex, non-linear temporal dependencies inherent in biological systems.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Pennylane_QGAN.git
   cd Pennylane_QGAN
   ```

2. **Create a virtual environment**
   ```bash
   # Using venv (recommended)
   python -m venv qgan_env
   
   # Activate the environment
   # On macOS/Linux:
   source qgan_env/bin/activate
   
   # On Windows:
   qgan_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import pennylane as qml; import torch; print('Installation successful!')"
   ```

## Usage

### Basic Usage

1. **Prepare your data**
   - Place your time series data in CSV format in the `qGAN/` directory
   - Update the data path in `_pl.py` (line 36): 
     ```python
     data = pd.read_csv('path/to/your/data.csv', header=None, names=['value'])
     ```

2. **Configure hyperparameters**
   - Modify training parameters in `qgan_pennylane.py` (lines 1011-1017):
     ```python
     EPOCHS = 10          # Number of training epochs
     BATCH_SIZE = 20      # Batch size for training
     NUM_QUBITS = 5       # Number of qubits in quantum circuit
     NUM_LAYERS = 3       # Number of quantum circuit layers
     ```

3. **Run the training**
   ```bash
   cd qGAN
   python qgan_pennylane.py
   ```

### Output Files

The script generates several output files:
- `OD_log_delta.csv`: Original log returns
- `fake_OD_log_delta.csv`: Generated log returns
- `real.csv`: Real data with timestamps
- `fake.csv`: Generated data with timestamps
- Various visualization plots (automatically displayed)

### Key Components

#### Quantum Generator
- **Architecture**: Parameterized quantum circuit with RX, RY rotation gates and CNOT entanglement
- **Measurements**: Both Pauli-X and Pauli-Z expectation values (2N outputs for N qubits)
- **Noise Encoding**: Direct uniform sampling from [0, 2π] for quantum gate parameters

#### Classical Discriminator
- **Architecture**: Multi-layer perceptron with batch normalization and dropout
- **Loss Function**: Wasserstein distance with gradient penalty
- **Optimization**: Adam optimizer with separate learning rates

#### Training Metrics
- **Earth Mover's Distance (EMD)**: Measures distributional similarity
- **Autocorrelation Function (ACF) RMSE**: Captures temporal dependencies
- **Volatility RMSE**: Models variance clustering
- **Leverage Effect RMSE**: Asymmetric volatility response

## Model Architecture Details

### Quantum Circuit Design
```
|0⟩ ──RY(θ₁)──●────────RX(φ₁)──RX(ψ₁)──RY(θ₂)──●──── ... ──⟨X⟩⟨Z⟩
|0⟩ ──RY(θ₁)──X──●─────RX(φ₁)──RX(ψ₁)──RY(θ₂)──X──── ... ──⟨X⟩⟨Z⟩
|0⟩ ──RY(θ₁)─────X──●──RX(φ₁)──RX(ψ₁)──RY(θ₂)─────── ... ──⟨X⟩⟨Z⟩
...
```

### Parameter Count
- **Quantum Generator**: N × (3 × layers + 2) parameters
- **Classical Discriminator**: ~390K parameters
- **Total Trainable Parameters**: Depends on configuration

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
   ```bash
   # Force CPU usage if GPU issues occur
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **Memory Issues**
   - Reduce `BATCH_SIZE` in the configuration
   - Reduce `NUM_QUBITS` or `NUM_LAYERS`

3. **Convergence Issues**
   - Increase `EPOCHS`
   - Adjust learning rates in optimizer configuration
   - Check data preprocessing and scaling

### Dependencies Issues

If you encounter package conflicts:
```bash
# Create a fresh environment
deactivate
rm -rf qgan_env
python -m venv qgan_env
source qgan_env/bin/activate
pip install -r requirements.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pennylane_qgan_bioprocess,
  title={PennyLane Quantum GAN for Industrial Bioprocess Synthetic Data Generation},
  author={Shawn Gibford},
  year={2024},
  url={https://github.com/shawngibford/qGAN}
}
```

## Acknowledgments

- PennyLane team for the quantum computing framework
- PyTorch team for the deep learning framework
- The quantum machine learning community for foundational research

## Contact

For questions and support, please open an issue on GitHub or contact [shawgi@dtu.dk].
