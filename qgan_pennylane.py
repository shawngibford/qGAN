# Install required packages
# %pip install pennylane torch torchvision pandas numpy matplotlib seaborn scipy statsmodels fastdtw dtaidistance


# Library imports
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, wasserstein_distance, probplot, entropy
from scipy.spatial.distance import euclidean
from scipy.special import lambertw
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
import time
import numpy as np 
from fastdtw import fastdtw
from dtaidistance import dtw_visualisation as dtwvis
import random
from dtaidistance import dtw
# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# PennyLane imports
import pennylane as qml
# Set random seeds for reproducibility ### check source code for seed usage
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Load and preprocess the data
data = pd.read_csv('/Users/shawngibford/dev/Pennylane_QGAN/qGAN/data.csv', header=None, names=['value'])
# Convert string values to float, replacing any non-numeric values with NaN
data['value'] = pd.to_numeric(data['value'], errors='coerce')
# Fill missing values with rolling mean
data['value'] = data['value'].fillna(data['value'].rolling(window=10, min_periods=10).mean())
# Drop any remaining NaN values
data = data.dropna()
# Convert to tensor
OD = torch.tensor(data['value'].values, dtype=torch.float32)
print('Data shape (total measurements):', OD.shape)
# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(OD.numpy())
plt.title(' Optical Density Time Series')
plt.xlabel('Time Steps')
plt.ylabel('Optical Density (OD)')
plt.grid(True)
plt.show()
# Display some basic statistics
print('\nBasic data statistics:')
print(data['value'].describe())


# Create time index for plotting
time_index = np.arange(len(OD))
# Direct returns over time
OD_delta = OD[1:] - OD[:-1]
# Logarithmic returns over time
OD_log_delta = torch.log(OD[1:]) - torch.log(OD[:-1])
# Convert to numpy for plotting
OD_delta_np = OD_delta.numpy()
OD_log_delta_np = OD_log_delta.numpy()
# Plot the graphs side-by-side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
# Plot direct returns
axes[0].plot(time_index[1:], OD_delta_np)
axes[0].set_title('Lucy Direct Returns')
axes[0].set_xlabel('Time Steps')
axes[0].set_ylabel('Direct Returns')
axes[0].grid(True)
# Plot log returns
axes[1].plot(time_index[1:], OD_log_delta_np)
axes[1].set_title('Lucy Log Returns')
axes[1].set_xlabel('Time Steps')
axes[1].set_ylabel('Log Returns')
axes[1].grid(True)
plt.tight_layout()
plt.show()
# Print basic statistics of returns
print("\nDirect Returns Statistics:")
print(f"Mean: {np.mean(OD_delta_np):.4f}")
print(f"Std: {np.std(OD_delta_np):.4f}")
print(f"Min: {np.min(OD_delta_np):.4f}")
print(f"Max: {np.max(OD_delta_np):.4f}")
print(f"Skewness: {stats.skew(OD_delta_np):.4f}")
print(f"Kurtosis: {stats.kurtosis(OD_delta_np):.4f}")
print("\nLog Returns Statistics:")
print(f"Mean: {np.mean(OD_log_delta_np):.4f}")
print(f"Std: {np.std(OD_log_delta_np):.4f}")
print(f"Min: {np.min(OD_log_delta_np):.4f}")
print(f"Max: {np.max(OD_log_delta_np):.4f}")
print(f"Skewness: {stats.skew(OD_log_delta_np):.4f}")
print(f"Kurtosis: {stats.kurtosis(OD_log_delta_np):.4f}")

# Statistical analysis of the data
data = OD.numpy()
# Create subplots for different visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# Histogram with KDE
sns.histplot(data=data, kde=True, ax=axes[0,0])
axes[0,0].set_title('Distribution of OD Values')
# Q-Q plot
stats.probplot(data, dist="norm", plot=axes[0,1])
axes[0,1].set_title('Q-Q Plot')
# Autocorrelation plot
sm.graphics.tsa.plot_acf(data, lags=40, ax=axes[1,0])
axes[1,0].set_title('Autocorrelation Plot')
# Box plot
sns.boxplot(data=data, ax=axes[1,1])
axes[1,1].set_title('Box Plot')
plt.tight_layout()
plt.show()
# Print basic statistics
print('\nBasic Statistics:')
print(f'Mean: {np.mean(data):.4f}')
print(f'Std: {np.std(data):.4f}')
print(f'Min: {np.min(data):.4f}')
print(f'Max: {np.max(data):.4f}')
print(f'Skewness: {stats.skew(data):.4f}')
print(f'Kurtosis: {stats.kurtosis(data):.4f}')

# Analyze the distribution of log returns
# Convert PyTorch tensor to numpy array for plotting
OD_log_delta_np = OD_log_delta.numpy()
# plot the graphs side-by-side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,4))
# density of log-returns
bin_edges = np.linspace(-0.05, 0.05, num=50)  # define the bin edges
bin_width = bin_edges[1] - bin_edges[0]
bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)
axes[0].hist(OD_log_delta_np, bins=bin_edges, density=True, width=0.001, label='Log-returns density')
axes[0].grid()
# normal distribution with same mean and standard deviation as log-returns
mu = np.mean(OD_log_delta_np)
sigma = np.std(OD_log_delta_np)
# Generate a set of points x
x = np.linspace(-0.05, 0.05, 100)
# Generate the Gaussian PDF for the points x with same mean and standard deviation as the log-returns
pdf = norm.pdf(x, mu, sigma)
# plot the Gaussian PDF
axes[0].plot(x, pdf, 'r', label='Gaussian distribution')
axes[0].legend()
axes[0].set_title('Log Returns Distribution vs Normal Distribution')
axes[0].set_xlabel('Log Returns')
axes[0].set_ylabel('Density')
# plot in logarithmic scale
axes[1].hist(OD_log_delta_np, bins=bin_edges, density=True, width=0.001, log=True)
axes[1].grid()
axes[1].set_title('Log Returns Distribution (Log Scale)')
axes[1].set_xlabel('Log Returns')
axes[1].set_ylabel('Log Density')
# plot the Gaussian PDF in logarithmic scale
axes[1].semilogy(x, pdf, 'r')
plt.tight_layout()
plt.show()
# Print some statistics about the log returns
print("\nLog Returns Statistics:")
print(f"Mean: {mu:.4f}")
print(f"Standard Deviation: {sigma:.4f}")
print(f"Skewness: {stats.skew(OD_log_delta_np):.4f}")
print(f"Kurtosis: {stats.kurtosis(OD_log_delta_np):.4f}")

# Analyze autocorrelation in log returns
# Convert PyTorch tensor to numpy array for ACF plotting
OD_log_delta_np = OD_log_delta.numpy()
# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
# Plot ACF of log returns
tsaplots.plot_acf(OD_log_delta_np, lags=18, zero=False, ax=ax1)
ax1.set_xlabel('Lags')
ax1.set_title('ACF OD Log-Returns')
ax1.set_ylabel(r'$\rho$')
ax1.grid(True)
# Plot ACF of absolute log returns
tsaplots.plot_acf(np.abs(OD_log_delta_np), lags=18, zero=False, ax=ax2)
ax2.set_xlabel('Lags')
ax2.set_title('ACF OD Absolute Log-Returns')
ax2.set_ylabel(r'$\rho_{abs}$')
ax2.grid(True)
plt.tight_layout()
plt.show()
# Print some additional statistics about the autocorrelation
print("\nAutocorrelation Statistics:")
# Calculate first-order autocorrelation for both series
acf_log = sm.tsa.acf(OD_log_delta_np, nlags=1)[1]
acf_abs = sm.tsa.acf(np.abs(OD_log_delta_np), nlags=1)[1]
print(f"First-order autocorrelation of log returns: {acf_log:.4f}")
print(f"First-order autocorrelation of absolute log returns: {acf_abs:.4f}")

# Generate the Q-Q plot
plt.figure(figsize=(6, 4))
probplot(OD_log_delta, dist='norm', plot=plt)
plt.title('Q-Q Plot - Lucy OD Log Returns')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True)
plt.show()

# Define utility functions for data preprocessing
def normalize(data):
    """Normalize the data to have zero mean and unit variance."""
    mu = torch.mean(data)
    sigma = torch.std(data)
    return (data - mu) / sigma  # Return only normalized data
def denormalize(norm_data, mu_original, std_original):
    """Denormalize the data back to original scale."""
    return norm_data * std_original + mu_original

# normalize the log-returns
norm_OD_log_delta = normalize(OD_log_delta)
# display the mean and standard deviation of the original log-returns
print(f'Original Lucy OD log-returns mean = {torch.mean(OD_log_delta)}, std = {torch.std(OD_log_delta)}')
# display the mean and standard deviation of the normalized log-returns
print(f'Normalized Lucy OD log-returns mean = {torch.mean(norm_OD_log_delta)}, std = {torch.std(norm_OD_log_delta)}')

print('Original Data Min-Max')
print(torch.min(OD_log_delta).item(), torch.max(OD_log_delta).item())
print('Normalized Data Min-Max')
print(torch.min(norm_OD_log_delta).item(), torch.max(norm_OD_log_delta).item())
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7,4))
# density of log-returns
bin_edges = np.linspace(-6, 6, num=100)  # define the bin edges
bin_width = bin_edges[1] - bin_edges[0]
bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)
# Convert PyTorch tensor to numpy for matplotlib
axes.hist(norm_OD_log_delta.numpy(), bins=bin_edges, density=True, width=0.1, label='Normalized density')
axes.grid()
# normal distribution with same mean and standard deviation as log-returns
mu = torch.mean(norm_OD_log_delta).item()  # Convert to Python scalar
sigma = torch.std(norm_OD_log_delta).item()  # Convert to Python scalar
# Generate a set of points x
x = np.linspace(-6, 6, 100)
# Generate the Gaussian PDF for the points x with same mean and standard deviation as the log-returns
pdf = norm.pdf(x, mu, sigma)
# plot the Gaussian PDF
axes.plot(x, pdf, 'r', label='Gaussian')
axes.legend()
plt.show()

def inverse_lambert_w_transform(data, delta):
    """
    Apply inverse Lambert W transform to the input data using the specified delta value.
    Parameters:
    - data: Input data tensor
    - delta: Delta value for the transform (tail parameter)
    Returns:
    - Transformed data tensor
    """
    # Convert to float64 for precision
    data = data.double()
    
    sign = torch.sign(data)
    
    # Convert to numpy for lambertw, then back to tensor
    data_squared = data ** 2
    lambert_input = (delta * data_squared).cpu().numpy()
    lambert_result = lambertw(lambert_input).real
    
    # Convert back to tensor and apply transform
    lambert_tensor = torch.tensor(lambert_result, dtype=torch.float64, device=data.device)
    transformed_data = sign * torch.sqrt(lambert_tensor / delta)
    return transformed_data
def lambert_w_transform(transformed_data, delta, clip_low=-12.0, clip_high=11.0):
    """
    Transform the Gaussianized data back to its original state.
    Parameters:
    - transformed_data: Input data tensor which was transformed using inverse Lambert W
    - delta: Delta value for the transform (tail parameter)
    - clip_low: Lower clipping bound
    - clip_high: Upper clipping bound
    Returns:
    - Original Data tensor
    """
    # Convert to float64 for precision
    transformed_data = transformed_data.double()
    
    # Apply the reverse transform
    exp_term = torch.exp((delta / 2) * transformed_data ** 2)
    reversed_data = transformed_data * exp_term
    
    # Clip the values
    return torch.clamp(reversed_data, clip_low, clip_high)

# apply inverse Lambert W transform to the normalized log-returns
delta = 1
transformed_norm_OD_log_delta = inverse_lambert_w_transform(norm_OD_log_delta, delta)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7,4))
# density of normalized log-returns
bin_edges = np.linspace(-3, 3, num=50)  # define the bin edges
bin_width = bin_edges[1] - bin_edges[0]
bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)
# Convert PyTorch tensor to numpy for matplotlib
axes.hist(transformed_norm_OD_log_delta.numpy(), bins=bin_edges, density=True, width=0.1, label='Transformed log-returns')
axes.grid()
# normal distribution with same mean and standard deviation as log-returns
mu = torch.mean(transformed_norm_OD_log_delta).item()  # Convert to Python scalar
sigma = torch.std(transformed_norm_OD_log_delta).item()  # Convert to Python scalar
# Generate a set of points x
x = np.linspace(-3, 3, 100)
# Generate the Gaussian PDF for the points x with same mean and standard deviation as the normalized log-returns
pdf = norm.pdf(x, mu, sigma)
# plot the Gaussian PDF
axes.plot(x, pdf, 'r', label='Gaussian')
axes.legend()
plt.show()

# Generate the Q-Q plot
plt.figure(figsize=(6, 4))
probplot(transformed_norm_OD_log_delta, dist='norm', plot=plt)
plt.title('Q-Q Plot - Transformed Lucy OD "Log Returns"')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True)
plt.show()

print('Transformed Data Min-Max')
print(torch.min(transformed_norm_OD_log_delta).item(), torch.max(transformed_norm_OD_log_delta).item())


min_val = torch.min(transformed_norm_OD_log_delta)
max_val = torch.max(transformed_norm_OD_log_delta)
scaled_data = -1.0 + 2.0 * (transformed_norm_OD_log_delta - min_val) / (max_val - min_val)
print(f'Scaled Normalized Transformed log-returns mean = {torch.mean(scaled_data)}, std = {torch.std(scaled_data)}')
print('Scaled Normalized Transformed log-returns min-max: ', torch.min(scaled_data).item(), torch.max(scaled_data).item())


def rescale(scaled_data, original_data):
    """
    Scale back from [-1,1] to the previous range
    
    Parameters:
    - scaled_data: Data scaled to [-1, 1] range
    - original_data: Original data to get min/max values from
    
    Returns:
    - Rescaled data in original range
    """
    min_val = torch.min(original_data)
    max_val = torch.max(original_data)
    previous_data = 0.5 * (scaled_data + 1.0) * (max_val - min_val) + min_val
    
    return previous_data
def rolling_window(data, m, s):
    """
    Create rolling windows from data
    
    Parameters:
    - data: Input tensor
    - m: Window size
    - s: Step size (stride)
    
    Returns:
    - Tensor of rolling windows
    """
    # Calculate number of windows
    num_windows = (len(data) - m) // s + 1
    
    # Create indices for each window
    windows = []
    for i in range(0, len(data) - m + 1, s):
        windows.append(data[i:i+m])
    
    # Stack all windows into a tensor
    return torch.stack(windows)

# Enable best available device (CUDA > MPS > CPU)
device = torch.device(
            "cuda:0" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
print(device)

# Part 1 - __init__
class qGAN(nn.Module):
    def __init__(self, num_epochs, batch_size, window_length, n_critic, gp, num_layers, num_qubits):
        super().__init__()  # Modern Python 3+ syntax
        
        # classical hyperparameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.window_length = window_length
        self.n_critic = n_critic
        self.gp = gp
        
        # quantum hyperparameters
        self.num_layers = num_layers
        self.num_qubits = num_qubits 
        # Quantum device (for PennyLane)
        self.quantum_dev = qml.device("default.qubit", wires=self.num_qubits)
        
        # PyTorch device (for classical computation)
        self.torch_device = torch.device(
            "cuda:0" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
        
        # quantum circuit settings
        self.qubits = list(range(num_qubits))  # PennyLane uses wire indices instead of GridQubit
        
        # create the set of Pauli strings to measure -> {X1, Z1, X2, Z2, etc}
        # X1 means we measure the first qubit only with X, Z1 the first qubit only with Z and so on...
        # Return measurements based on self.measurements (X and Z measurements)
        self.measurements = []
        for i in range(self.num_qubits):
            self.measurements.append(qml.expval(qml.PauliX(i)))  # X measurement
            self.measurements.append(qml.expval(qml.PauliZ(i)))  # Z measurement
        # number of parameters of the PQC and re-uploading layers
        self.num_params = self.count_params()
        # define the trainable parameters of the PQC main and re-uploading layers (trainable)
        # Use larger initialization for better expressivity
        self.params_pqc = torch.nn.Parameter(
            torch.randn(self.num_params, requires_grad=True, dtype=torch.float32) * 0.5
        )
        # define the classical critic network (CNN)
        self.critic = self.define_critic_model(window_length)
        # define the quantum generator network (PQC)
        self.generator = self.define_generator_model()
        # Create QNode with proper PyTorch integration for gradient flow        
       
        # monitoring purposes
        # average critic and generator losses for each epoch
        self.critic_loss_avg = []
        self.generator_loss_avg = []
        # Earth's mover distance (EMD) for each epoch
        self.emd_avg = []
        # stylized facts RMSEs for each epoch
        self.acf_avg = []
        self.vol_avg = []
        self.lev_avg = []
# Part 2 - gen and critic models
    ####################################################################################
    #
    # count the parameters of the quantum circuit
    #
    ####################################################################################
    def count_params(self):
        # Simple parameter counting for StronglyEntanglingLayers:
        # - StronglyEntanglingLayers: num_layers * num_qubits * 3 (each layer has 3 rotations per qubit)
        # - Additional rotations for remaining parameters: up to num_qubits * 3
        strongly_entangling_params = self.num_layers * self.num_qubits * 3
        additional_rotations = self.num_qubits * 3  # Extra rotation layer for flexibility
        return strongly_entangling_params + additional_rotations
    ####################################################################################
    #
    # the classical critic model as a convolutional network
    #
    ####################################################################################
    def define_critic_model(self, window_length):
        """Define the classical critic model"""
        model = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=64, kernel_size=10, stride=1, padding=5),
        nn.LeakyReLU(negative_slope=0.1),
        
        nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, stride=1, padding=5),
        nn.LeakyReLU(negative_slope=0.1),
        
        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=10, stride=1, padding=5),
        nn.LeakyReLU(negative_slope=0.1),
        
        # Add adaptive pooling to get fixed size
        nn.AdaptiveAvgPool1d(output_size=1),  # This gives 128 * 1 = 128 features ### verify if this needs to be here
        nn.Flatten(),
        
        nn.Linear(in_features=128, out_features=32),  # 128 -> 32
        nn.LeakyReLU(negative_slope=0.1),
        nn.Dropout(p=0.2),
        
        nn.Linear(in_features=32, out_features=1)
        )
    
        model = model.double()
        return model
    ####################################################################################
    #
    # the encoding layer: resolve the parameters by uniform noise values,
    # used to prepare the initial state for the generator circuit
    #
    ####################################################################################
    def encoding_layer(self, noise_params):
        """
        IQP (Instantaneous Quantum Polynomial) Encoding for Time Series Data
        Based on Havlicek et al. (2018) - captures feature correlations and temporal relationships
        """
        # Step 1: Hadamard gates to create superposition
        for qubit in range(self.num_qubits):
            qml.Hadamard(wires=qubit)
        
        # Step 2: Single-qubit RZ rotations encoding individual features
        for i in range(min(len(noise_params), self.num_qubits)):
            qml.RZ(noise_params[i], wires=i)
        
        # Step 3: Two-qubit ZZ entanglers encoding feature correlations
        # This captures temporal relationships between adjacent time steps
        for i in range(min(len(noise_params)-1, self.num_qubits-1)):
            if i+1 < len(noise_params):
                # Encode correlation between adjacent time steps
                correlation_angle = noise_params[i] * noise_params[i+1]
                qml.MultiRZ(correlation_angle, wires=[i, i+1])
        
        # Step 4: Additional correlations for richer encoding (if we have enough qubits)
        if len(noise_params) >= 3 and self.num_qubits >= 3:
            # Encode second-order temporal correlations
            for i in range(min(len(noise_params)-2, self.num_qubits-2)):
                if i+2 < len(noise_params):
                    # Correlation between time steps separated by one
                    skip_correlation = noise_params[i] * noise_params[i+2] * 0.5
                    qml.MultiRZ(skip_correlation, wires=[i, i+2])
    
    def parameter_reupload(self, noise_params, reupload_weights):
        """
        Parameter re-uploading: re-encode the data with learned transformations
        This significantly improves expressivity and helps avoid barren plateaus
        """
        # Transform the noise with learned parameters
        weight_idx = 0
        
        # Apply learned rotations to re-encode the data
        for i in range(min(len(noise_params), self.num_qubits)):
            if weight_idx < len(reupload_weights):
                # Re-encode with learned transformation
                transformed_noise = noise_params[i] + reupload_weights[weight_idx]
                qml.RY(transformed_noise, wires=i)
                weight_idx += 1
                
                if weight_idx < len(reupload_weights):
                    qml.RZ(noise_params[i] * reupload_weights[weight_idx], wires=i)
                    weight_idx += 1
        
        # Add some entanglement after re-uploading
        for i in range(min(self.num_qubits-1, (len(reupload_weights) - weight_idx) // 2)):
            if weight_idx + 1 < len(reupload_weights):
                qml.CRY(reupload_weights[weight_idx], wires=[i, (i+1) % self.num_qubits])
                weight_idx += 1
    ####################################################################################
    #
    # the quantum generator as a PQC with All-to-all topology for the entangling layer
    #
    ####################################################################################
    def define_generator_circuit(self, noise_params, params_pqc):
        # Apply IQP encoding layer (includes Hadamards + feature encoding)
        self.encoding_layer(noise_params)
    
        # index for the parameter tensor of the PQC main and re-uploading layers
        idx = 0
        
        # Note: Hadamard gates are now included in the IQP encoding layer
    
        # Use StronglyEntanglingLayers for much better expressivity
        # Reshape parameters to fit StronglyEntanglingLayers format: (n_layers, n_qubits, 3)
        
        # Calculate how many complete strongly entangling layers we can make
        params_per_layer = self.num_qubits * 3  # 3 rotations per qubit per layer
        available_layers = len(params_pqc) // params_per_layer
        
        if available_layers > 0:
            # Reshape parameters for StronglyEntanglingLayers
            strongly_entangling_params = params_pqc[:available_layers * params_per_layer]
            strongly_entangling_weights = strongly_entangling_params.reshape(
                available_layers, self.num_qubits, 3
            )
            
            # Apply StronglyEntanglingLayers with custom connectivity
            qml.StronglyEntanglingLayers(
                weights=strongly_entangling_weights,
                wires=range(self.num_qubits),
                ranges=None,  # Use default range pattern for good connectivity
                imprimitive=qml.CNOT  # Use CNOT as the entangling gate
            )
            
            # Use remaining parameters for additional expressivity if available
            remaining_params = params_pqc[available_layers * params_per_layer:]
            idx = 0
            
            # Add extra rotation layer with remaining parameters
            for qubit in range(min(len(remaining_params) // 3, self.num_qubits)):
                if idx + 2 < len(remaining_params):
                    qml.RX(phi=remaining_params[idx], wires=qubit)
                    idx += 1
                    qml.RY(phi=remaining_params[idx], wires=qubit)
                    idx += 1
                    qml.RZ(phi=remaining_params[idx], wires=qubit)
                    idx += 1
        else:
            # Fallback: if not enough parameters, use simple rotations
            for qubit in range(min(len(params_pqc) // 3, self.num_qubits)):
                if idx + 2 < len(params_pqc):
                    qml.RX(phi=params_pqc[idx], wires=qubit)
                    idx += 1
                    qml.RY(phi=params_pqc[idx], wires=qubit)
                    idx += 1
                    qml.RZ(phi=params_pqc[idx], wires=qubit)
                    idx += 1
        # Note: Final rotation layer is now handled within StronglyEntanglingLayers
        # This provides better parameter efficiency and expressivity
    
        # Simple measurement strategy - just X and Z measurements
        # Return measurements as tuple for proper PennyLane autograd integration
        measurements = []
        for i in range(self.num_qubits):
            measurements.append(qml.expval(qml.PauliX(i)))  # X measurement
            measurements.append(qml.expval(qml.PauliZ(i)))  # Z measurement
        
        # Return as unpacked tuple (comma-separated) for proper gradient flow
        return (*measurements,)
    ####################################################################################
    #
    # the quantum generator model
    #
    ####################################################################################
    def define_generator_model(self):
        
    
        # define the pennylane quantum layer (trainable)
        generator = qml.QNode(
            self.define_generator_circuit, 
            self.quantum_dev, 
            interface='torch', 
            diff_method='parameter-shift'  # Explicit gradient method for better stability and gradient flow
        )
    
        # generator output (will be computed when called)
        generator_output = generator
    
        # pytorch model equivalent
        model = generator
    
        return model
    #############################################################################
    #
    # compile model with given optimizers for critic and generator networks
    #
    #############################################################################
    def compile_QGAN(self, c_optimizer, g_optimizer):
        # PyTorch doesn't have a compile method like Keras
        # Just store the optimizers
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
    def train_qgan(self, gan_data, original_data, preprocessed_data, num_elements):
        """
        Parameters:
        - gan_data is the preprocessed dataset with windows for qGAN training
        - original_data is the original S&P 500 log-returns for evaluation of RMSEs (monitoring purposes)
        - preprocessed_data is the preprocessed log-returns without the last normalization step and without windows
         (for reversing the process of generated samples using the mean and std and evaluating the RMSEs)
        """
        # Convert DataLoader to list for random sampling (PyTorch approach)
        gan_data_list = []
        for batch in gan_data:
            for sample in batch[0]:  # batch[0] contains the data
                gan_data_list.append(sample)
                
        # training loop
        for epoch in range(self.num_epochs):
            print(f'Processing epoch {epoch+1}/{self.num_epochs}')
            self._train_one_epoch(gan_data_list, original_data, preprocessed_data, epoch)
    def _train_one_epoch(self, gan_data_list, original_data, preprocessed_data, epoch: int):
        """Runs critic and generator steps and logs metrics for a single epoch."""
        ################################################################
        #
        # Train the critic for n_critic iterations
        # Process 'batch_size' samples in each iteration
        #
        ################################################################
        # critic loss for 'n_critic' iterations
        critic_t_sum = 0
        for t in range(self.n_critic):
            # zero gradients
            self.c_optimizer.zero_grad()
            # critic loss for 'batch_size' samples using proper GAN batch training
            # Generate batches of real and fake samples
            real_batch = []
            fake_batch = []
            
            for i in range(self.batch_size):
                # Sample real data
                random_idx = torch.randint(0, len(gan_data_list), (1,))
                real_sample = gan_data_list[random_idx.item()]
                real_sample = torch.reshape(real_sample, (1, self.window_length))
                real_sample = real_sample.unsqueeze(1).double()  # [1, 1, window_length]
                real_batch.append(real_sample)
                
                # Generate fake sample from random noise
                noise_values = np.random.uniform(0, 2 * np.pi, size=self.num_qubits)
                generator_input = torch.tensor(noise_values, dtype=torch.float32)
                generated_sample = self.generator(generator_input, self.params_pqc)
                
                # Handle different output types from quantum generator
                if isinstance(generated_sample, (list, tuple)):
                    generated_sample = torch.stack(list(generated_sample))
                elif not isinstance(generated_sample, torch.Tensor):
                    generated_sample = torch.tensor(generated_sample)
                
                generated_sample = generated_sample.to(torch.float64)
                # Scale quantum output [-1,1] to approximate real data range
                # Real data is roughly in [-0.08, 0.09], so scale by ~0.1
                generated_sample = generated_sample * 0.1
                generated_sample_input = generated_sample.unsqueeze(0).unsqueeze(1)  # [1, 1, window_length]
                fake_batch.append(generated_sample_input)

            # Combine batches for proper GAN training
            real_batch_tensor = torch.cat(real_batch, dim=0)  # [batch_size, 1, window_length]
            fake_batch_tensor = torch.cat(fake_batch, dim=0)  # [batch_size, 1, window_length]

            # Calculate critic scores for real and fake batches
            real_scores = self.critic(real_batch_tensor)
            fake_scores = self.critic(fake_batch_tensor)

            # Compute Wasserstein loss: E[D(fake)] - E[D(real)]
            real_score_mean = torch.mean(real_scores)
            fake_score_mean = torch.mean(fake_scores)

            # Compute gradient penalty using interpolation between real and fake batches
            alpha = torch.rand(self.batch_size, 1, 1).to(real_batch_tensor.device)
            interpolated = alpha * real_batch_tensor + (1 - alpha) * fake_batch_tensor
            interpolated.requires_grad_(True)

            interpolated_scores = self.critic(interpolated)
            gradients = torch.autograd.grad(
                outputs=interpolated_scores,
                inputs=interpolated,
                grad_outputs=torch.ones_like(interpolated_scores),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            gradient_penalty = torch.mean((gradients.norm(2, dim=[1, 2]) - 1) ** 2)

            # Final critic loss following WGAN-GP formulation
            critic_loss = fake_score_mean - real_score_mean + self.gp * gradient_penalty
            critic_sum = critic_loss
            # compute the gradients of critic and apply them
            critic_sum.backward()
            self.c_optimizer.step()
            # accumulate the critic loss for this 't' iteration
            critic_t_sum += critic_sum
        # average critic loss for this epoch of WGAN training
        self.critic_loss_avg.append(critic_t_sum / self.n_critic)
        ################################################################
        #
        # Train generator for one iteration
        #
        ################################################################
        # sample a batch of input states using noise parameters (not encoding layer)
        input_circuits_batch = []
        for _ in range(self.batch_size):
            noise_values = np.random.uniform(0, 2 * np.pi, size=self.num_qubits)
            input_circuits_batch.append(noise_values)
        # convert to torch tensor batch
        generator_inputs = torch.stack([torch.tensor(noise, dtype=torch.float32) for noise in input_circuits_batch])
        # zero gradients
        self.g_optimizer.zero_grad()
        # generate fake samples using the generator (FIXED for gradient flow)
        generated_samples = []
        for i in range(generator_inputs.shape[0]):
            # Each circuit call must maintain connection to self.params_pqc for gradient flow
            gen_out = self.generator(generator_inputs[i], self.params_pqc)
            if isinstance(gen_out, (list, tuple)):
                gen_out = torch.stack(gen_out)
            # Scale quantum output [-1,1] to approximate real data range
            gen_out = gen_out.to(torch.float64) * 0.1
            generated_samples.append(gen_out)
        generated_samples = torch.stack(generated_samples)
        # Add channel dimension for Conv1D
        generated_samples_input = generated_samples.unsqueeze(1)  # [batch, 1, features]
        # calculate the critic scores for fake samples
        fake_scores = self.critic(generated_samples_input)
        # calculate the generator loss
        generator_loss = -torch.mean(fake_scores)

        # compute the gradients of generator and apply them
        generator_loss.backward()
        self.g_optimizer.step()
        # average generator loss for this epoch
        self.generator_loss_avg.append(generator_loss)
        ########################################################################################################
        #
        # Calculate the stylized facts RMSEs and the EMD for real and fake data
        #
        # Fake data has shape (num_samples x window_length), with num_samples = original_length / window_length
        # in order to get a time series close to the length of the original
        #
        ########################################################################################################
        # generate noise
        num_samples = len(original_data) // self.window_length
        input_circuits_batch = []
        for _ in range(num_samples):
            noise_values = np.random.uniform(0, 2 * np.pi, size=self.num_qubits)
            input_circuits_batch.append(noise_values)
        # convert to torch tensor batch
        generator_inputs = torch.stack([torch.tensor(noise, dtype=torch.float32) for noise in input_circuits_batch])
        # generate fake samples using the generator
        batch_generated = []
        for generator_input in generator_inputs:
            gen_out = self.generator(generator_input, self.params_pqc)
            if isinstance(gen_out, (list, tuple)):
                gen_out = torch.stack(list(gen_out))
            # Scale quantum output [-1,1] to approximate real data range
            gen_out = gen_out.to(torch.float64) * 0.1
            batch_generated.append(gen_out)
        batch_generated = torch.stack(batch_generated)
        # concatenate all time series data into one
        generated_data = torch.reshape(batch_generated, shape=(num_samples * self.window_length,))
        generated_data = generated_data.double()
        # rescale
        generated_data = rescale(generated_data, preprocessed_data)
        # reverse the preprocessing on generated sample
        original_norm = lambert_w_transform(generated_data, delta)
        # Skip denormalization - use Lambert W output directly  
        fake_original = original_norm
        # calculate the temporal metrics for monitoring the training process
        corr_rmse, volatility_rmse, lev_rmse, emd = self.stylized_facts(original_data, fake_original)
        # store the EMD and RMSEs of stylized facts
        self.acf_avg.append(corr_rmse)
        self.vol_avg.append(volatility_rmse)
        self.lev_avg.append(lev_rmse)
        self.emd_avg.append(emd)
        # print progress every 100 epochs
        if epoch % 100 == 0 or epoch + 1 == 3000:
            print(f'\nEpoch {epoch+1} completed')
        # Safe access to loss values with bounds checking
        if len(self.critic_loss_avg) > epoch:
            critic_loss_val = self.critic_loss_avg[epoch]
            if hasattr(critic_loss_val, 'item'):
                critic_loss_val = critic_loss_val.item()
            print(f'Critic loss (average): {critic_loss_val}')
        else:
            print(f'Critic loss (average): Not available (index {epoch}, list length {len(self.critic_loss_avg)})')
        if len(self.generator_loss_avg) > epoch:
            generator_loss_val = self.generator_loss_avg[epoch]
            if hasattr(generator_loss_val, 'item'):
                generator_loss_val = generator_loss_val.item()
            print(f'Generator loss (average): {generator_loss_val}')
        else:
            print(f'Generator loss (average): Not available (index {epoch}, list length {len(self.generator_loss_avg)})')
        # Safe access to other metrics
        if len(self.emd_avg) > epoch:
            print(f'\nEMD (average): {self.emd_avg[epoch]}')
            print(f'ACF RMSE (average): {self.acf_avg[epoch]}')
            print(f'VOLATILITY RMSE (average): {self.vol_avg[epoch]}')
            print(f'LEVERAGE RMSE (average): {self.lev_avg[epoch]}\n')
        else:
            print(f'\nMetrics not available for epoch {epoch}')
        # Only print min/max if variables exist
        try:
            print('Min-Max values of original log-returns: ', torch.min(original_data).item(), torch.max(original_data).item())
            print('Min-Max values of generated log-returns (for all batches): ', torch.min(fake_original).item(), torch.max(fake_original).item())
            print('Min-Max values after Lambert: ', torch.min(original_norm).item(), torch.max(original_norm).item())
        except NameError:
            print('Min-Max values not available (variables not defined)')
            print()
    ###########################################################
    #
    # Sample a random number epsilon ~ U[0,1]
    # Create a convex combination of real and generated sample
    # Compute the gradient penalty for the critic network
    #
    ###########################################################
    def compute_gradient_penalty(self, real_sample, generated_sample):
        epsilon = torch.rand(1, dtype=torch.float64)
        interpolated_sample = epsilon * real_sample + (1 - epsilon) * generated_sample
        interpolated_sample.requires_grad_(True)
        scores = self.critic(interpolated_sample)
        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated_sample,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True
        )[0]
    
        gradients_norm = torch.norm(gradients)
        gradient_penalty = (gradients_norm - 1)**2
        return gradient_penalty
    def stylized_facts(self, original_data, fake_original):
        """
        - Calculate the RMSEs of the stylized facts between the original S&P 500 log-returns and
          generated time series
        - Evaluate the EMD between real and generated samples
        """
        # Ensure NumPy arrays for statsmodels and NumPy operations
        if isinstance(fake_original, torch.Tensor):
            fake_np = fake_original.detach().cpu().numpy()
        else:
            fake_np = np.asarray(fake_original)
        if isinstance(original_data, torch.Tensor):
            orig_np = original_data.detach().cpu().numpy()
        else:
            orig_np = np.asarray(original_data)
        ################################################
        #
        # stylized facts for fake samples
        #
        ################################################
        # compute acf for maximum lags = 18
        acf_values = sm.tsa.acf(fake_np, nlags=18)
        # exclude zero lag
        acf_values_generated = torch.tensor(acf_values[1:])
        # compute absolute acf (volatility clustering) for maximum lags = 18
        acf_abs_values = sm.tsa.acf(np.abs(fake_np), nlags=18)
        # exclude zero lag
        acf_abs_values_generated = torch.tensor(acf_abs_values[1:])
        # compute leverage effect for maximum lags = 18
        lev = []
        for lag in range(1, 19):
            # slice the tensors to get the appropriate lagged sequences
            r_t = fake_np[:-lag]
            squared_lag_r = np.square(np.abs(fake_np[lag:]))
            # calculate the leverage effect
            # calculate the correlation coefficient
            correlation_matrix = np.corrcoef(r_t, squared_lag_r)
            lev.append(correlation_matrix[0, 1])
        leverage_generated = torch.tensor(lev)
        ################################################
        #
        # stylized facts for real samples
        #
        ################################################
        # compute acf for maximum lags = 18
        acf_values = sm.tsa.acf(orig_np, nlags=18)
        # exclude zero lag
        acf_values_original = torch.tensor(acf_values[1:])
        # compute absolute acf (volatility clustering) for maximum lags = 18
        acf_abs_values = sm.tsa.acf(np.abs(orig_np), nlags=18)
        # exclude zero lag
        acf_abs_values_original = torch.tensor(acf_abs_values[1:])
        # compute leverage effect for maximum lags = 18
        lev = []
        for lag in range(1, 19):
            # slice the tensors to get the appropriate lagged sequences
            r_t = orig_np[:-lag]
            squared_lag_r = np.square(np.abs(orig_np[lag:]))
            # calculate the leverage effect
            correlation_matrix = np.corrcoef(r_t, squared_lag_r)
            lev.append(correlation_matrix[0, 1])
        leverage_original = torch.tensor(lev)
        # calculate average RMSEs of stylized facts
        # autocorrelations
        rmse_acf = torch.sqrt(torch.mean((acf_values_original-acf_values_generated)**2))
        # volatility clustering
        rmse_vol = torch.sqrt(torch.mean((acf_abs_values_original-acf_abs_values_generated)**2))
        # leverage effect
        rmse_lev = torch.sqrt(torch.mean((leverage_original-leverage_generated)**2))
        ####################################################################################
        #
        # compute the Earth's mover distance (EMD)
        #
        ####################################################################################
        bin_edges = np.linspace(-0.05, 0.05, num=50)  # define the bin edges
        bin_width = bin_edges[1] - bin_edges[0]
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)
        # compute the empirical distribution of original data
        empirical_real, _ = np.histogram(orig_np, bins=bin_edges, density=True)
        empirical_real /= np.sum(empirical_real)
        # compute the empirical distribution of generated data
        empirical_fake, _ = np.histogram(fake_np, bins=bin_edges, density=True)
        empirical_fake /= np.sum(empirical_fake)
        # evaluate the EMD using SciPy
        emd = wasserstein_distance(empirical_real, empirical_fake)
        return rmse_acf, rmse_vol, rmse_lev, emd

##################################################################
#
# Hyperparameters
#
##################################################################
WINDOW_LENGTH = 10  # this must be equal to the number of Pauli strings to measure
NUM_QUBITS = 5  # number of qubits
NUM_LAYERS = 3  # number of layers for the PQC
# training hyperparameters
EPOCHS = 10  # Increase from 10 to allow proper learning
BATCH_SIZE = 20
n_critic = 2  # Critic iterations per generator update - back to standard
LAMBDA = 0.1  # gradient penalty strength - much smaller to allow adversarial signal
# Learning rates for optimizers - boost generator learning
LR_CRITIC = 1e-4  # Critic learning rate
LR_GENERATOR = 5e-4  # Generator learning rate - balanced with critic
# instantiate the QGAN model objec0
qgan = qGAN(EPOCHS, BATCH_SIZE, WINDOW_LENGTH, n_critic, LAMBDA, NUM_LAYERS, NUM_QUBITS)
# set the optimizers
c_optimizer = torch.optim.Adam(qgan.critic.parameters(), lr=LR_CRITIC, betas=(0.0, 0.9))
g_optimizer = torch.optim.Adam([qgan.params_pqc], lr=LR_GENERATOR, betas=(0.0, 0.9))  # Use the quantum parameters
qgan.compile_QGAN(c_optimizer, g_optimizer)

##################################################################################
#
# Data pre-processing
#
##################################################################################
# apply rolling window in transformed (scaled) log-returns with stride s=2
gan_data_tf = rolling_window(scaled_data, WINDOW_LENGTH, 2)
# create PyTorch datasets (handle both numpy array and tensor inputs)
if isinstance(gan_data_tf, np.ndarray):
    data_tensor = torch.from_numpy(gan_data_tf).float()
else:
    data_tensor = gan_data_tf.float()
gan_data = torch.utils.data.TensorDataset(data_tensor)
gan_data = torch.utils.data.DataLoader(gan_data, batch_size=1, shuffle=True)
# get the number of elements in the dataset
num_elements = len(gan_data_tf)
# Create dummy parameters for visualization
dummy_noise_params = torch.tensor([0.5] * NUM_QUBITS, dtype=torch.float32)
dummy_circuit_params = torch.tensor([0.1] * qgan.num_params, dtype=torch.float32)
# Use the correct attribute name 
print("Circuit visualization:")
print(qml.draw(qgan.generator)(dummy_noise_params, dummy_circuit_params))
# For matplotlib visualization:
fig, ax = qml.draw_mpl(qgan.generator)(dummy_noise_params, dummy_circuit_params)
fig.show()

# train the QGAN
print('Training started...')
print('Number of samples to process per epoch: ', num_elements)
print()
start_time_train = time.time()
model = qgan.train_qgan(gan_data, OD_log_delta, transformed_norm_OD_log_delta, num_elements)
exec_time_train = time.time() - start_time_train
print(f'\nQGAN training completed. Training time: --- {exec_time_train/3600:.02f} hours ---')

# Check training status - FIXED VERSION
print("QGAN Training Status:")
print(f"Epochs completed: {len(qgan.critic_loss_avg)}")
print(f"Generator parameters shape: {qgan.params_pqc.shape}")
print(f"Parameter range: [{qgan.params_pqc.min().item():.4f}, {qgan.params_pqc.max().item():.4f}]")
if len(qgan.critic_loss_avg) == 0:
    print("❌ MODEL NOT TRAINED - Run training cell first!")
else:
    print("✅ Model has been trained")
    # Convert tensors to scalars using .item() before formatting
    critic_loss = qgan.critic_loss_avg[-1]
    generator_loss = qgan.generator_loss_avg[-1]
    
    # Handle both tensor and scalar cases
    if hasattr(critic_loss, 'item'):
        critic_loss = critic_loss.item()
    if hasattr(generator_loss, 'item'):
        generator_loss = generator_loss.item()
    
    print(f"Final losses - Critic: {critic_loss:.4f}, Generator: {generator_loss:.4f}")

# Convert PyTorch data to match TensorFlow structure exactly
def convert_losses_pytorch_to_tf_format(critic_losses, generator_losses):
    """Convert PyTorch losses to match TensorFlow format"""
    
    # Handle critic losses - equivalent to tf.squeeze(qgan.critic_loss_avg, axis=(1,2)).numpy()
    if isinstance(critic_losses, list):
        critic_array = []
        for loss in critic_losses:
            if isinstance(loss, torch.Tensor):
                # Squeeze out extra dimensions and convert to scalar
                squeezed = loss.squeeze().detach().cpu().numpy()
                if squeezed.ndim == 0:  # scalar
                    critic_array.append(squeezed.item())
                else:
                    critic_array.append(squeezed)
            else:
                critic_array.append(loss)
        critic_loss = np.array(critic_array)
    else:
        critic_loss = np.array(critic_losses).squeeze()
    
    # Handle generator losses - equivalent to np.array(qgan.generator_loss_avg)
    if isinstance(generator_losses, list):
        gen_array = []
        for loss in generator_losses:
            if isinstance(loss, torch.Tensor):
                gen_array.append(loss.detach().cpu().numpy().item())
            else:
                gen_array.append(loss)
        generator_loss = np.array(gen_array)
    else:
        generator_loss = np.array(generator_losses)
    
    return critic_loss, generator_loss
# Convert the data
critic_loss, generator_loss = convert_losses_pytorch_to_tf_format(
    qgan.critic_loss_avg, qgan.generator_loss_avg
)
print(f"Converted data shapes:")
print(f"critic_loss: {critic_loss.shape}, values: {critic_loss}")
print(f"generator_loss: {generator_loss.shape}, values: {generator_loss}")
# Check if we have enough data for the original plotting logic
if len(critic_loss) < 50:
    print(f"WARNING: Only {len(critic_loss)} epochs of data. Original code expects more for moving average.")
    print("This suggests training stopped early or only ran for a few epochs.")
    
    # For small datasets, use a smaller window or no moving average
    if len(critic_loss) > 1:
        window = min(5, len(critic_loss))  # Use smaller window
        print(f"Using reduced window size: {window}")
    else:
        print("Cannot create moving average with only 1 data point.")
        # Show the single value
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(['Critic Loss', 'Generator Loss'], [critic_loss[0], generator_loss[0]])
        ax.set_title('Single Epoch Results')
        plt.show()
        exit()
else:
    window = 50
# plot the moving average
generator_ma = np.convolve(generator_loss, np.ones(window)/window, mode='valid')
critic_ma = np.convolve(critic_loss, np.ones(window)/window, mode='valid')
# plot the graphs side-by-side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,4))
# plot the critic loss moving average as a line
axes[0].plot(range(window-1, len(critic_loss)), critic_ma, label='Average Critic Loss', color='blue')
# plot the critic loss
axes[0].plot(critic_loss, color='black', alpha=0.2)
# plot the generator loss moving average as a line
axes[0].plot(range(window-1, len(generator_loss)), generator_ma, label='Average Generator Loss', color='orange')
# plot the generator loss
axes[0].plot(generator_loss, color='black', alpha=0.2)
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid()
# Convert other metrics
emd_avg = np.array([x.item() if isinstance(x, torch.Tensor) else x for x in qgan.emd_avg])
emd_ma = np.convolve(emd_avg, np.ones(window)/window, mode='valid')
axes[1].plot(range(window-1, len(emd_avg)), emd_ma, label='EMD', color='red')
axes[1].plot(emd_avg, color='red', linewidth=0.5, alpha=0.5)
axes[1].set_ylabel('EMD')
axes[1].legend()
axes[1].grid()
acf_avg = np.array([x.item() if isinstance(x, torch.Tensor) else x for x in qgan.acf_avg])
vol_avg = np.array([x.item() if isinstance(x, torch.Tensor) else x for x in qgan.vol_avg])
lev_avg = np.array([x.item() if isinstance(x, torch.Tensor) else x for x in qgan.lev_avg])
acf_ma = np.convolve(acf_avg, np.ones(window)/window, mode='valid')
vol_ma = np.convolve(vol_avg, np.ones(window)/window, mode='valid')
lev_ma = np.convolve(lev_avg, np.ones(window)/window, mode='valid')
# Creating a twin axes for the second graph
axes2 = axes[1].twinx()
axes2.plot(range(window-1, len(acf_avg)), acf_ma, label='ACF', color='green')
axes2.plot(acf_avg, color='green', linewidth=0.5, alpha=0.4)
axes2.plot(range(window-1, len(vol_avg)), vol_ma, label='Volatility Clustering', color='black')
axes2.plot(vol_avg, color='black', linewidth=0.5, alpha=0.3)
axes2.plot(range(window-1, len(lev_avg)), lev_ma, label='Leverage Effect', color='orange')
axes2.set_ylabel('Temporal Metrics')
axes2.legend()
axes2.grid()
# Adjusting the spacing between subplots
plt.tight_layout()
plt.show()

print(f"window: {window}, len(critic_loss): {len(critic_loss)}")

def debug_and_fix_generation():
    """
    Complete debugging and fixing of QGAN generation pipeline
    Run this cell to diagnose and fix generation issues
    """
    print("🔍 QGAN GENERATION PIPELINE DEBUGGER")
    print("=" * 60)
    
    # Step 1: Check if all required variables exist
    print("\n1️⃣ VARIABLE EXISTENCE CHECK")
    print("-" * 30)
    
    required_vars = ['qgan', 'OD_log_delta', 'transformed_norm_OD_log_delta', 'WINDOW_LENGTH', 'NUM_QUBITS']
    missing_vars = []
    
    for var_name in required_vars:
        try:
            var_value = eval(var_name)
            print(f"✅ {var_name}: Found")
        except NameError:
            print(f"❌ {var_name}: MISSING!")
            missing_vars.append(var_name)
    
    if missing_vars:
        print(f"\n❌ Missing variables: {missing_vars}")
        print("Please ensure all required variables are defined before running generation code.")
        return None
    
    # Step 2: Check training status
    print("\n2️⃣ TRAINING STATUS")
    print("-" * 30)
    
    epochs = len(qgan.critic_loss_avg)
    if epochs == 0:
        print("❌ MODEL NOT TRAINED! Run training cell first.")
        return None
    
    param_min = qgan.params_pqc.min().item()
    param_max = qgan.params_pqc.max().item()
    print(f"✅ Training epochs: {epochs}")
    print(f"✅ Parameter range: [{param_min:.4f}, {param_max:.4f}]")
    
    # Step 3: Test quantum generator
    print("\n3️⃣ QUANTUM GENERATOR TEST")
    print("-" * 30)
    
    try:
        # Test single generation
        test_noise = np.random.uniform(0, 2 * np.pi, size=NUM_QUBITS)
        test_input = torch.tensor(test_noise, dtype=torch.float32)
        
        with torch.no_grad():
            test_output = qgan.generator(test_input, qgan.params_pqc)
        
        if isinstance(test_output, (list, tuple)):
            test_output = torch.stack(test_output)
        
        out_min, out_max = test_output.min().item(), test_output.max().item()
        print(f"✅ Generator output shape: {test_output.shape}")
        print(f"✅ Generator output range: [{out_min:.6f}, {out_max:.6f}]")
        
        if torch.isnan(test_output).any():
            print("❌ Generator produces NaN values!")
            return None
        
        if out_min == out_max:
            print("❌ Generator produces constant values!")
            return None
            
    except Exception as e:
        print(f"❌ Generator test failed: {e}")
        return None
    
    # Step 4: Full generation with debugging
    print("\n4️⃣ FULL GENERATION PIPELINE")
    print("-" * 30)
    
    try:
        # Generate samples with debugging
        num_samples = min(len(OD_log_delta) // WINDOW_LENGTH, 20)  # Limit for debugging
        print(f"Generating {num_samples} samples...")
        
        # Generate noise batch
        input_circuits_batch = []
        for _ in range(num_samples):
            noise_values = np.random.uniform(0, 2 * np.pi, size=NUM_QUBITS)
            input_circuits_batch.append(noise_values)
        
        generator_inputs = torch.stack([torch.tensor(noise, dtype=torch.float32) for noise in input_circuits_batch])
        
        # Generate samples
        batch_generated = []
        for generator_input in generator_inputs:
            with torch.no_grad():  # Important: disable gradients for generation
                generated_sample = qgan.generator(generator_input, qgan.params_pqc)
                if isinstance(generated_sample, (list, tuple)):
                    generated_sample = torch.stack(list(generated_sample))
                # Scale quantum output [-1,1] to approximate real data range
                generated_sample = generated_sample.to(torch.float64) * 0.1
                batch_generated.append(generated_sample)
        
        batch_generated = torch.stack(batch_generated)
        print(f"✅ Generated batch shape: {batch_generated.shape}")
        
        # Reshape
        generated_data = torch.reshape(batch_generated, shape=(num_samples * WINDOW_LENGTH,))
        gen_min, gen_max = generated_data.min().item(), generated_data.max().item()
        print(f"✅ Reshaped data: {generated_data.shape}, range [{gen_min:.6f}, {gen_max:.6f}]")
        
        # Rescale step
        generated_data_rescaled = rescale(generated_data, transformed_norm_OD_log_delta)
        resc_min, resc_max = generated_data_rescaled.min().item(), generated_data_rescaled.max().item()
        print(f"✅ After rescale: range [{resc_min:.6f}, {resc_max:.6f}]")
        
        # Lambert W transform
        original_norm = lambert_w_transform(generated_data_rescaled, 1)
        lamb_min, lamb_max = original_norm.min().item(), original_norm.max().item()
        print(f"✅ After Lambert W: range [{lamb_min:.6f}, {lamb_max:.6f}]")
        
        # Denormalize
        # Skip denormalization - use Lambert W output directly
        fake_original = original_norm
        final_min, final_max = fake_original.min().item(), fake_original.max().item()
        print(f"✅ Final data: range [{final_min:.6f}, {final_max:.6f}]")
        
        # Check for issues
        if torch.isnan(fake_original).any():
            print("❌ Final data contains NaN values!")
            return None
        
        if final_min == final_max:
            print("❌ Final data is constant!")
            return None
        
        print("✅ Generation pipeline completed successfully!")
        
        return fake_original
        
    except Exception as e:
        print(f"❌ Generation pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None
def create_fixed_plots(fake_original):
    """
    Create plots with proper error handling and debugging
    """
    if fake_original is None:
        print("❌ No data to plot!")
        return
    
    print("\n5️⃣ CREATING PLOTS")
    print("-" * 30)
    
    try:
        # Convert to numpy
        OD_log_delta_np = OD_log_delta.detach().cpu().numpy()
        fake_OD_log_delta_np = fake_original.detach().cpu().numpy()
        
        print(f"✅ Real data: {len(OD_log_delta_np)} points, range [{OD_log_delta_np.min():.6f}, {OD_log_delta_np.max():.6f}]")
        print(f"✅ Fake data: {len(fake_OD_log_delta_np)} points, range [{fake_OD_log_delta_np.min():.6f}, {fake_OD_log_delta_np.max():.6f}]")
        
        # Create figure
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        
        # Plot original data
        axes[0].plot(OD_log_delta_np, 'b-', linewidth=0.8, alpha=0.8)
        axes[0].set_xlabel('Time Steps')
        axes[0].set_title('Original Log-Returns')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([OD_log_delta_np.min() * 1.1, OD_log_delta_np.max() * 1.1])
        
        # Plot generated data
        axes[1].plot(fake_OD_log_delta_np, 'r-', linewidth=0.8, alpha=0.8)
        axes[1].set_xlabel('Time Steps')
        axes[1].set_title('Generated Log-Returns')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([fake_OD_log_delta_np.min() * 1.1, fake_OD_log_delta_np.max() * 1.1])
        
        plt.tight_layout()
        plt.show()
        
        # Create comparison histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(OD_log_delta_np, bins=50, alpha=0.7, label='Original', density=True, color='blue')
        ax.hist(fake_OD_log_delta_np, bins=50, alpha=0.7, label='Generated', density=True, color='red')
        ax.set_xlabel('Log Returns')
        ax.set_ylabel('Density')
        ax.set_title('Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
        
        print("✅ Plots created successfully!")
        
        # Print statistics
        print(f"\nStatistics Comparison:")
        print(f"Original  - Mean: {np.mean(OD_log_delta_np):.6f}, Std: {np.std(OD_log_delta_np):.6f}")
        print(f"Generated - Mean: {np.mean(fake_OD_log_delta_np):.6f}, Std: {np.std(fake_OD_log_delta_np):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False
# ============================================================================
# MAIN EXECUTION
# ============================================================================
print("🚀 Starting QGAN Generation Pipeline Debug and Fix...")
# Run the debugging
fake_original = debug_and_fix_generation()
# Create plots if generation succeeded
if fake_original is not None:
    success = create_fixed_plots(fake_original)
    
    if success:
        print("\n🎉 SUCCESS! Generation pipeline is now working!")
        print("Your QGAN is successfully generating synthetic time series data.")
    else:
        print("\n⚠️ Generation worked but plotting failed. Check plot code.")
else:
    print("\n❌ Generation pipeline has issues. Check the debug output above.")
    print("\nCommon fixes:")
    print("1. Ensure model is trained (run training cell)")
    print("2. Check that all variables are defined")
    print("3. Verify preprocessing functions are available")
    print("4. Check for NaN/Inf values in data")
print("\n" + "=" * 60)
print("🔍 DEBUGGING COMPLETE")
print("=" * 60)


# generate noise
num_samples = len(OD_log_delta) // WINDOW_LENGTH
print(f"num_samples: {num_samples}")
print(f"WINDOW_LENGTH: {WINDOW_LENGTH}")
input_circuits_batch = []
for _ in range(num_samples):
    noise_values = np.random.uniform(0, 2 * np.pi, size=NUM_QUBITS)
    input_circuits_batch.append(noise_values)  # Store noise values, not encoding layer result ### -check this shit out? we need encoding layer
# convert to torch tensor batch
generator_inputs = torch.stack([torch.tensor(noise, dtype=torch.float32) for noise in input_circuits_batch])
print(f"generator_inputs shape: {generator_inputs.shape}")
# generate fake samples using the generator
batch_generated = []
print(f"Generating {len(generator_inputs)} samples...")
for i in range(generator_inputs.shape[0]):
    with torch.no_grad():  # Disable gradients for generation
        generated_sample = qgan.generator(generator_inputs[i], qgan.params_pqc)
        if isinstance(generated_sample, (list, tuple)):
            generated_sample = torch.stack(list(generated_sample))
        # Scale quantum output [-1,1] to approximate real data range
        generated_sample = generated_sample.to(torch.float64) * 0.1
        batch_generated.append(generated_sample)
    
    # Debug print for first few samples
    if i < 3:
        sample_min, sample_max = generated_sample.min().item(), generated_sample.max().item()
        print(f"Sample {i}: range [{sample_min:.6f}, {sample_max:.6f}]")
batch_generated = torch.stack(batch_generated)
print(f"batch_generated shape: {batch_generated.shape}")
# concatenate all time series data into one
generated_data = torch.reshape(batch_generated, shape=(num_samples * WINDOW_LENGTH,))
generated_data = generated_data.double()
print(f"generated_data shape: {generated_data.shape}")
print(f"generated_data range: [{generated_data.min().item():.6f}, {generated_data.max().item():.6f}]")
# rescale
generated_data = rescale(generated_data, transformed_norm_OD_log_delta)
print(f"After rescale: [{generated_data.min().item():.6f}, {generated_data.max().item():.6f}]")
# reverse the preprocessing on generated sample
original_norm = lambert_w_transform(generated_data, 1)
print(f"After Lambert W: [{original_norm.min().item():.6f}, {original_norm.max().item():.6f}]")
# Skip denormalization - use Lambert W output directly since it's already in good scale
fake_original = original_norm
print(f"Final fake_original: [{fake_original.min().item():.6f}, {fake_original.max().item():.6f}]")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
# Debug: Check variable existence and content
print("=== DEBUGGING GENERATION ISSUE ===")
# Check if variables exist
try:
    print(f"OD_log_delta shape: {OD_log_delta.shape}")
    print(f"OD_log_delta range: [{OD_log_delta.min().item():.6f}, {OD_log_delta.max().item():.6f}]")
    print(f"OD_log_delta first 5 values: {OD_log_delta[:5].detach().cpu().numpy()}")
except NameError:
    print("❌ OD_log_delta not defined!")
try:
    print(f"fake_original shape: {fake_original.shape}")
    print(f"fake_original range: [{fake_original.min().item():.6f}, {fake_original.max().item():.6f}]")
    print(f"fake_original first 5 values: {fake_original[:5].detach().cpu().numpy()}")
except NameError:
    print("❌ fake_original not defined! Need to run generation code first.")
# Debug: Check generation pipeline step by step
print("\n=== GENERATION PIPELINE DEBUG ===")
# Check if generation variables exist
variables_to_check = ['num_samples', 'generated_data', 'original_norm', 'fake_original']
for var_name in variables_to_check:
    try:
        var_value = locals()[var_name]
        if hasattr(var_value, 'shape'):
            print(f"✅ {var_name}: shape {var_value.shape}, range [{var_value.min():.6f}, {var_value.max():.6f}]")
        else:
            print(f"✅ {var_name}: {var_value}")
    except KeyError:
        print(f"❌ {var_name}: NOT DEFINED")
# Create plot with explicit debugging
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
# Convert to numpy explicitly
OD_log_delta_np = OD_log_delta.detach().cpu().numpy()
fake_OD_log_delta_np = fake_original.detach().cpu().numpy()
print(f"Plotting data - Real: {len(OD_log_delta_np)} points, Fake: {len(fake_OD_log_delta_np)} points")
# Plot original
axes[0].plot(OD_log_delta_np, 'b-', linewidth=1)
axes[0].set_xlabel('Days')
axes[0].set_title('Original Log-Returns')
axes[0].grid(True)
axes[0].set_ylim([-0.1, 0.1])
# Plot generated
axes[1].plot(fake_OD_log_delta_np, 'r-', linewidth=1)
axes[1].set_xlabel('Days')
axes[1].set_title('Generated Log-Returns') 
axes[1].grid(True)
axes[1].set_ylim([-0.1, 0.1])
plt.tight_layout()
plt.show()
# Additional check
print(f"Real data stats: mean={np.mean(OD_log_delta_np):.6f}, std={np.std(OD_log_delta_np):.6f}")
print(f"Fake data stats: mean={np.mean(fake_OD_log_delta_np):.6f}, std={np.std(fake_OD_log_delta_np):.6f}")

###################################################################################################################
#
# plot original log-returns on the left and the generated on the right
#
###################################################################################################################
# Convert PyTorch tensors to numpy for plotting
OD_log_delta_np = OD_log_delta.detach().cpu().numpy() if isinstance(OD_log_delta, torch.Tensor) else np.asarray(OD_log_delta)
fake_OD_log_delta_np = fake_original.detach().cpu().numpy() if isinstance(fake_original, torch.Tensor) else np.asarray(fake_original)
# Debug the converted numpy arrays
print(f"OD_log_delta_np shape: {OD_log_delta_np.shape}, range: [{OD_log_delta_np.min():.6f}, {OD_log_delta_np.max():.6f}]")
print(f"fake_OD_log_delta_np shape: {fake_OD_log_delta_np.shape}, range: [{fake_OD_log_delta_np.min():.6f}, {fake_OD_log_delta_np.max():.6f}]")
# Create DataFrames for later visualization code
# Generate date range for the data (assuming daily data)
import pandas as pd
from datetime import datetime, timedelta
start_date = datetime(2020, 1, 1)
real_dates = [start_date + timedelta(days=i) for i in range(len(OD_log_delta_np))]
fake_dates = [start_date + timedelta(days=i) for i in range(len(fake_OD_log_delta_np))]
real_data = pd.DataFrame({
    'DATE': real_dates,
    'Log_Return': OD_log_delta_np.flatten()
})
fake_data = pd.DataFrame({
    'DATE': fake_dates,
    'Log_Return': fake_OD_log_delta_np.flatten()
})
# Create figure before plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
# Plot using index on x-axis with better debugging
print(f"Plotting {len(OD_log_delta_np)} original points and {len(fake_OD_log_delta_np)} generated points")
axes[0].plot(OD_log_delta_np, 'b-', linewidth=0.8, alpha=0.8)
axes[0].set_xlabel('Days')
axes[0].set_title('Original Log-Returns')
axes[0].grid(True)
axes[0].set_ylim([-0.1, 0.1])
axes[1].plot(fake_OD_log_delta_np, 'r-', linewidth=0.8, alpha=0.8)
axes[1].set_xlabel('Days')
axes[1].set_title('Generated Log-Returns') 
axes[1].grid(True)
axes[1].set_ylim([-0.1, 0.1])
plt.tight_layout()
plt.show()
# Save generated series to CSV
df = pd.DataFrame(fake_OD_log_delta_np, columns=['generated_log_return'])
csv_filename = 'fake_original_OD_log_delta.csv'
df.to_csv(csv_filename, index=False)
print(f"Saved {len(fake_OD_log_delta_np)} generated samples to {csv_filename}")

# Create figure FIRST before using axes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
###################################################################################################################
#
# plot histogram of generated along with original log-returns on the left and the Q-Q plot on the right
#
###################################################################################################################
bin_edges = np.linspace(-0.05, 0.05, num=50)  # define the bin edges
bin_width = bin_edges[1] - bin_edges[0]
bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)
orig_hist_np = OD_log_delta.detach().cpu().numpy() if isinstance(OD_log_delta, torch.Tensor) else np.asarray(OD_log_delta)
# Debug: Check if data exists before plotting
print(f"Original data for histogram: shape={orig_hist_np.shape}, range=[{orig_hist_np.min():.6f}, {orig_hist_np.max():.6f}]")
print(f"Generated data for histogram: shape={fake_OD_log_delta_np.shape}, range=[{fake_OD_log_delta_np.min():.6f}, {fake_OD_log_delta_np.max():.6f}]")
axes[0].hist(fake_OD_log_delta_np, bins=bin_edges, density=True, label='Generated', alpha=0.7, color='red')
axes[0].hist(orig_hist_np, bins=bin_edges, density=True, label='Original', alpha=0.7, color='blue')
axes[0].set_title('Original vs Generated Density')
axes[0].grid()
axes[0].legend()
probplot(fake_OD_log_delta_np, dist='norm', plot=axes[1])
axes[1].set_xlabel('Theoretical Quantiles')
axes[1].set_title('Q-Q Plot of Generated Data')
axes[1].grid()
plt.tight_layout()
plt.show()

# Create figure FIRST before using axes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
###################################################################################################################
#
# plot autocorrelations of original log-returns on the left and the generated on the right
#
###################################################################################################################
# Debug: Check if data exists
print(f"Original data type: {type(OD_log_delta)}, shape: {OD_log_delta.shape if hasattr(OD_log_delta, 'shape') else 'no shape'}")
print(f"Generated data type: {type(fake_OD_log_delta_np)}, shape: {fake_OD_log_delta_np.shape}")
tsaplots.plot_acf(OD_log_delta, ax=axes[0], lags=18, zero=False)
axes[0].set_xlabel('Lags')
axes[0].set_title('ACF Log-Returns (Original)')
axes[0].grid()
tsaplots.plot_acf(fake_OD_log_delta_np, ax=axes[1], lags=18, zero=False)
axes[1].set_xlabel('Lags')
axes[1].set_title('ACF Log-Returns (Generated)')
axes[1].grid()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
###################################################################################################################
#
# plot volatility clustering of original log-returns on the left and the generated on the right
#
###################################################################################################################
tsaplots.plot_acf(torch.abs(OD_log_delta).detach().cpu().numpy(), ax=axes[0], lags=18, zero=False)
axes[0].set_xlabel('Lags')
axes[0].set_title('ACF Absolute Log-Returns')
axes[0].grid()
tsaplots.plot_acf(torch.abs(fake_original).detach().cpu().numpy(), ax=axes[1], lags=18, zero=False)
axes[1].set_xlabel('Lags')
axes[1].set_title('ACF Absolute Log-Returns (Generated)')
axes[1].grid()


# Create figure FIRST before using axes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
###################################################################################################################
#
# plot leverage effect of original log-returns on the left and the generated on the right
#
###################################################################################################################
# Debug: Check if data exists
print(f"OD_log_delta type: {type(OD_log_delta)}, shape: {OD_log_delta.shape if hasattr(OD_log_delta, 'shape') else 'no shape'}")
print(f"fake_original type: {type(fake_original)}, shape: {fake_original.shape if hasattr(fake_original, 'shape') else 'no shape'}")
# compute leverage effect for maximum lags = 18
leverage_original = []
for lag in range(1, 19):
    # slice the tensors to get the appropriate lagged sequences
    r_t = OD_log_delta[:-lag].detach().cpu().numpy()
    squared_lag_r = torch.square(torch.abs(OD_log_delta[lag:])).detach().cpu().numpy()
    
    # calculate the leverage effect
    # calculate the correlation coefficient
    correlation_matrix = np.corrcoef(r_t, squared_lag_r)
    leverage_original.append(correlation_matrix[0, 1])
leverage_generated = []
for lag in range(1, 19):
    # slice the tensors to get the appropriate lagged sequences
    r_t = fake_original[:-lag].detach().cpu().numpy()
    squared_lag_r = torch.square(torch.abs(fake_original[lag:])).detach().cpu().numpy()
    
    # calculate the leverage effect
    # calculate the correlation coefficient
    correlation_matrix = np.corrcoef(r_t, squared_lag_r)
    leverage_generated.append(correlation_matrix[0, 1])
# Debug the leverage calculations
print(f"Leverage original: {len(leverage_original)} values, range: [{min(leverage_original):.6f}, {max(leverage_original):.6f}]")
print(f"Leverage generated: {len(leverage_generated)} values, range: [{min(leverage_generated):.6f}, {max(leverage_generated):.6f}]")
# Plot the leverage effects
axes[0].plot(leverage_original, 'b-', linewidth=1.5)
axes[0].set_xlabel('Lags')
axes[0].set_title('Original Leverage Effect')
axes[0].grid(True)
axes[1].plot(leverage_generated, 'r-', linewidth=1.5)
axes[1].set_xlabel('Lags')
axes[1].set_title('Generated Leverage Effect')
axes[1].grid(True)
plt.tight_layout()
plt.show()

# Save the DataFrame to a CSV file
csv_filename = 'real.csv'
real_data.to_csv(csv_filename, index=False)
# Save the DataFrame to a CSV file
csv_filename = 'fake.csv'
fake_data.to_csv(csv_filename, index=False)

# Ensure both time series are of the same length for comparison
min_length = min(len(real_data), len(fake_data))
real_ts = real_data['Log_Return'][:min_length]
fake_ts = fake_data['Log_Return'][:min_length]
# Generate QQ plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
# QQ plot for the original Lucy log returns
stats.probplot(real_ts, dist="norm", plot=ax1)
ax1.set_title('QQ Plot of Original Lucy Log Returns', fontsize= 18)
ax1.grid(True)
# QQ plot for the generated Lucy log returns
stats.probplot(fake_ts, dist="norm", plot=ax2)
ax2.get_lines()[0].set_color('orange')
ax2.set_title('QQ Plot of Generated Lucy Log Returns', fontsize= 18)
ax2.grid(True)
# Set the same scale for y axis
axes = plt.gca()
ylim1 = ax1.get_ylim()
ylim2 = ax2.get_ylim()
global_ylim = (min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
ax1.set_ylim(global_ylim)
ax2.set_ylim(global_ylim)
# Show the plots
plt.tight_layout()
plt.show()

# Calculate the probability distribution of the real and fake time series
real_ts_values, real_ts_base = np.histogram(real_ts, bins=100, density=True)
fake_ts_values, fake_ts_base = np.histogram(fake_ts, bins=100, density=True)
# Compute the cumulative distribution functions for both time series
real_ts_cdf = np.cumsum(real_ts_values * np.diff(real_ts_base))
fake_ts_cdf = np.cumsum(fake_ts_values * np.diff(fake_ts_base))
# Plotting the probability distributions of the real and fake time series
plt.figure(figsize=(12, 6))
plt.plot(real_ts_base[:-1], real_ts_cdf, label='CDF of Original Lucy Log Returns', color='blue')
plt.plot(fake_ts_base[:-1], fake_ts_cdf, label='CDF of Generated Lucy Log Returns', color='orange')
plt.xlabel('Log Return Value',fontsize=14)
plt.ylabel('Cumulative Distribution Function',fontsize=14)
plt.title('CDF of Original vs. Generated Lucy Log Returns',fontsize=18)
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Plotting the time series with the same y-axis scale
# Convert 'Date' columns to datetime for proper plotting
real_data['DATE'] = pd.to_datetime(real_data['DATE'])
fake_data['DATE'] = pd.to_datetime(fake_data['DATE'])
# Determine the global min and max log return values for the y-axis scale
min_log_return = min(real_data['Log_Return'].min(), fake_data['Log_Return'].min())
max_log_return = max(real_data['Log_Return'].max(), fake_data['Log_Return'].max())
# Plot both time series with the same y-axis scale
plt.figure(figsize=(20, 5))
# Plotting the original 
plt.subplot(1, 2, 1)
plt.plot(real_data['DATE'], real_data['Log_Return'], color='blue')
plt.title('Original Lucy Log Returns',fontsize=18)
plt.xlabel('Date',fontsize=14)
plt.ylabel('Log Return',fontsize=14)
plt.ylim(min_log_return, max_log_return)
plt.grid(True)
# Plotting the generated 
plt.subplot(1, 2, 2)
plt.plot(fake_data['DATE'], fake_data['Log_Return'], color='orange')
plt.title('Generated Lucy Log Returns',fontsize=18)
plt.xlabel('Date',fontsize=14)
plt.ylim(min_log_return, max_log_return)
plt.grid(True)
# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plots
plt.show()

# We will use seaborn's kdeplot to plot the PDFs of the real and fake data
plt.figure(figsize=(12, 6))
# Debug: Check if data exists
print(f"Original data: shape={OD_log_delta_np.shape}, range=[{OD_log_delta_np.min():.6f}, {OD_log_delta_np.max():.6f}]")
print(f"Generated data: shape={fake_OD_log_delta_np.shape}, range=[{fake_OD_log_delta_np.min():.6f}, {fake_OD_log_delta_np.max():.6f}]")
# Plotting the PDF of the original data
sns.kdeplot(data=OD_log_delta_np, color='blue', linewidth=4, label='Original Log Returns')
# Plotting the PDF of the generated data
sns.kdeplot(data=fake_OD_log_delta_np, color='orange', linewidth=4, label='Generated Log Returns')
# Plot formatting
plt.legend(prop={'size': 12}, title='PDF')
plt.title('PDF of Original vs. Generated Log Returns', fontsize=18)
plt.xlabel('Log Return', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(True)
# Show the plot
plt.show()

# Fix the variable names and calculate entropy properly
print("=== ENTROPY ANALYSIS ===")
# Use the correct variable names
original_data = OD_log_delta_np.flatten()
generated_data = fake_OD_log_delta_np.flatten()
print(f"Original data: {len(original_data)} points")
print(f"Generated data: {len(generated_data)} points")
# Calculate the Earth Mover's distance (Wasserstein distance)
emd_value = wasserstein_distance(original_data, generated_data)
print(f"\nEarth Mover's Distance: {emd_value:.6f}")
print("(Lower values indicate more similar distributions)")
# Calculate entropy of the two distributions
# Use the same bins for fair comparison
bins = np.linspace(min(np.min(original_data), np.min(generated_data)), 
                   max(np.max(original_data), np.max(generated_data)), 
                   100)
real_prob, _ = np.histogram(original_data, bins=bins, density=True)
fake_prob, _ = np.histogram(generated_data, bins=bins, density=True)
# Normalize to get proper probabilities
real_prob = real_prob / np.sum(real_prob)
fake_prob = fake_prob / np.sum(fake_prob)
# Add small epsilon to avoid log(0)
epsilon = 1e-10
real_prob = np.maximum(real_prob, epsilon)
fake_prob = np.maximum(fake_prob, epsilon)
# Calculate entropies
entropy_real = entropy(real_prob)
entropy_fake = entropy(fake_prob)
print(f"\nEntropy Analysis:")
print(f"Original data entropy: {entropy_real:.6f}")
print(f"Generated data entropy: {entropy_fake:.6f}")
print(f"Entropy difference: {abs(entropy_real - entropy_fake):.6f}")
print(f"Relative entropy difference: {abs(entropy_real - entropy_fake)/entropy_real*100:.2f}%")
# Calculate additional statistical measures
print(f"\n=== ADDITIONAL STATISTICAL COMPARISONS ===")
# Kolmogorov-Smirnov test
from scipy.stats import ks_2samp
ks_statistic, ks_pvalue = ks_2samp(original_data, generated_data)
print(f"Kolmogorov-Smirnov test:")
print(f"  Statistic: {ks_statistic:.6f}")
print(f"  P-value: {ks_pvalue:.6f}")
print(f"  Interpretation: {'Distributions are significantly different' if ks_pvalue < 0.05 else 'Distributions are similar'}")
# Jensen-Shannon divergence (symmetric version of KL divergence)
from scipy.spatial.distance import jensenshannon
js_distance = jensenshannon(real_prob, fake_prob)
print(f"\nJensen-Shannon Distance: {js_distance:.6f}")
print("(Range: 0-1, where 0 = identical distributions)")
# Basic statistical moments comparison
print(f"\n=== MOMENT COMPARISONS ===")
stats_comparison = {
    'Mean': (np.mean(original_data), np.mean(generated_data)),
    'Std': (np.std(original_data), np.std(generated_data)),
    'Skewness': (stats.skew(original_data), stats.skew(generated_data)),
    'Kurtosis': (stats.kurtosis(original_data), stats.kurtosis(generated_data))
}
for stat_name, (real_val, fake_val) in stats_comparison.items():
    diff = abs(real_val - fake_val)
    rel_diff = (diff / abs(real_val)) * 100 if real_val != 0 else float('inf')
    print(f"{stat_name:10}: Real={real_val:8.6f}, Generated={fake_val:8.6f}, Diff={diff:.6f} ({rel_diff:.2f}%)")
# Create comprehensive comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# 1. Probability distributions comparison
axes[0,0].hist(original_data, bins=50, density=True, alpha=0.7, label='Original', color='blue')
axes[0,0].hist(generated_data, bins=50, density=True, alpha=0.7, label='Generated', color='orange')
axes[0,0].set_title('Probability Distributions')
axes[0,0].set_xlabel('Log Returns')
axes[0,0].set_ylabel('Density')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)
# 2. Cumulative distributions
axes[0,1].hist(original_data, bins=100, density=True, cumulative=True, alpha=0.7, label='Original', color='blue')
axes[0,1].hist(generated_data, bins=100, density=True, cumulative=True, alpha=0.7, label='Generated', color='orange')
axes[0,1].set_title('Cumulative Distributions')
axes[0,1].set_xlabel('Log Returns')
axes[0,1].set_ylabel('Cumulative Density')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)
# 3. Q-Q plot comparison
from scipy.stats import probplot
probplot(original_data, dist="norm", plot=axes[0,2])
axes[0,2].get_lines()[0].set_markerfacecolor('blue')
axes[0,2].get_lines()[0].set_markeredgecolor('blue')
axes[0,2].get_lines()[0].set_label('Original')
probplot(generated_data, dist="norm", plot=axes[0,2])
axes[0,2].get_lines()[2].set_markerfacecolor('orange')
axes[0,2].get_lines()[2].set_markeredgecolor('orange')
axes[0,2].get_lines()[2].set_label('Generated')
axes[0,2].set_title('Q-Q Plot vs Normal Distribution')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)
# 4. Statistical metrics comparison
metrics = ['Mean', 'Std', 'Skewness', 'Kurtosis']
real_values = [stats_comparison[m][0] for m in metrics]
fake_values = [stats_comparison[m][1] for m in metrics]
x = np.arange(len(metrics))
width = 0.35
axes[1,0].bar(x - width/2, real_values, width, label='Original', color='blue', alpha=0.7)
axes[1,0].bar(x + width/2, fake_values, width, label='Generated', color='orange', alpha=0.7)
axes[1,0].set_title('Statistical Moments Comparison')
axes[1,0].set_xlabel('Metrics')
axes[1,0].set_ylabel('Values')
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(metrics)
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)
# 5. Distance metrics visualization
distance_metrics = ['EMD', 'Jensen-Shannon', 'Entropy Diff']
distance_values = [emd_value, js_distance, abs(entropy_real - entropy_fake)]
colors = ['red', 'purple', 'green']
bars = axes[1,1].bar(distance_metrics, distance_values, color=colors, alpha=0.7)
axes[1,1].set_title('Distance Metrics (Lower = More Similar)')
axes[1,1].set_ylabel('Distance Value')
axes[1,1].grid(True, alpha=0.3)
# Add value labels on bars
for bar, value in zip(bars, distance_values):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{value:.4f}', ha='center', va='bottom')
# 6. Entropy comparison visualization
entropy_data = [entropy_real, entropy_fake]
entropy_labels = ['Original', 'Generated']
colors = ['blue', 'orange']
bars = axes[1,2].bar(entropy_labels, entropy_data, color=colors, alpha=0.7)
axes[1,2].set_title('Entropy Comparison')
axes[1,2].set_ylabel('Entropy Value')
axes[1,2].grid(True, alpha=0.3)
# Add value labels
for bar, value in zip(bars, entropy_data):
    axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.4f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()
# Summary interpretation
print(f"\n=== SUMMARY INTERPRETATION ===")
print(f"1. Earth Mover's Distance ({emd_value:.6f}):")
if emd_value < 0.01:
    print("   EXCELLENT: Distributions are very similar")
elif emd_value < 0.05:
    print("   GOOD: Distributions are reasonably similar")
elif emd_value < 0.1:
    print("   FAIR: Distributions have noticeable differences")
else:
    print("   POOR: Distributions are quite different")
print(f"\n2. Jensen-Shannon Distance ({js_distance:.6f}):")
if js_distance < 0.1:
    print("   EXCELLENT: Very similar probability distributions")
elif js_distance < 0.3:
    print("   GOOD: Reasonably similar distributions")
elif js_distance < 0.5:
    print("   FAIR: Moderate differences")
else:
    print("   POOR: Significantly different distributions")
print(f"\n3. Entropy Difference ({abs(entropy_real - entropy_fake):.6f}):")
entropy_rel_diff = abs(entropy_real - entropy_fake)/entropy_real*100
if entropy_rel_diff < 5:
    print("   EXCELLENT: Very similar information content")
elif entropy_rel_diff < 15:
    print("   GOOD: Similar complexity/randomness")
elif entropy_rel_diff < 30:
    print("   FAIR: Noticeable difference in randomness")
else:
    print("   POOR: Very different levels of randomness")
print(f"\n4. Overall Assessment:")
good_metrics = sum([
    emd_value < 0.05,
    js_distance < 0.3,
    entropy_rel_diff < 15,
    ks_pvalue > 0.05
])
if good_metrics >= 3:
    print("   EXCELLENT: Generated data closely matches original distribution")
elif good_metrics >= 2:
    print("   GOOD: Generated data is reasonably similar to original")
elif good_metrics >= 1:
    print("   FAIR: Generated data has some similarities but needs improvement")
else:
    print("   POOR: Generated data differs significantly from original")

# Creating the ACF plots 
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))
# Plot autocorrelations of original log-returns on the left
tsaplots.plot_acf(real_ts, ax=axes[0], lags=30, zero=False)
axes[0].set_xlabel('Lags', fontsize = 14)
axes[0].set_title('ACF for Original Log $\delta$', fontsize = 16)
axes[0].grid()
# Plot autocorrelations of generated log-returns on the right
tsaplots.plot_acf(fake_ts, ax=axes[1], lags=30, zero=False, color='orange')
axes[1].set_xlabel('Lags', fontsize = 14)
axes[1].set_title('ACF for Generated $\delta$', fontsize = 16)
axes[1].grid()
plt.show()

# To reproduce a similar scatter plot for the time series, we will plot the real data against the generated data.
# Since we only have one dimension (the log returns), we'll create a lagged version of the data to allow for a 2D scatter plot.
# Create a lagged version of the time series
real_ts_lagged = real_ts.shift(1).dropna()
fake_ts_lagged = fake_ts.shift(1).dropna()
# Take the same number of points from each to have a fair comparison
min_length = min(len(real_ts_lagged), len(fake_ts_lagged))
real_ts_plot = real_ts_lagged.iloc[:min_length]
fake_ts_plot = fake_ts_lagged.iloc[:min_length]
real_ts_current = real_ts.iloc[1:min_length+1]
fake_ts_current = fake_ts.iloc[1:min_length+1]
# Plot
plt.figure(figsize=(14, 6))
plt.scatter(real_ts_current, real_ts_plot, color='blue', alpha=0.5, label='Real Data', marker='.')
plt.scatter(fake_ts_current, fake_ts_plot, color='orange', alpha=0.5, label='Synthetic Data', marker='.')
plt.xlabel('$x_1$', fontsize = 14)
plt.ylabel('$x_2$', fontsize = 14)
plt.title('Real vs. Synthetic Data', fontsize = 18)
plt.legend()
plt.grid()
plt.show()

# Plotting the histograms of the two time series one over the other
plt.figure(figsize=(12, 6))
bin_edges = np.linspace(-0.05, 0.05, num=50)  # define the bin edges
bin_width = bin_edges[1] - bin_edges[0]
bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)
# Plotting the histogram for the original S&P 500 Log Returns
plt.hist(real_ts, bins=bin_edges, alpha=0.7, label='Original OD', color='blue')
# Plotting the histogram for the generated S&P 500 Log Returns
plt.hist(fake_ts, bins=bin_edges, alpha=0.7, label='Generated Brasilian Stock Index', color='orange')
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.title('Histogram of Original vs. Generated $\delta$')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

series1 = real_data['Log_Return'].to_numpy().reshape(-1,1)
series2 = fake_data['Log_Return'].to_numpy().reshape(-1,1)
dtw_distance, warping_path = fastdtw(series1,series2, dist=euclidean)
dtw_distance

for idx in range(len(series2)):
    if random.random() < 0.05:
        series2[idx] += (random.random() - 0.5) / 2
d, paths = dtw.warping_paths(series1, series2, window=500, psi=2)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(series1, series2, paths, best_path)
# Re-attempt to plot the warping path correctly
# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the first series on the x-axis
ax.plot(series1, 'bo-', label='Real Data', alpha = 0.25)
# Plot the second series on the y-axis
ax.plot(series2, 'gs-', label='Fake Data', alpha = 0.25)
# Draw lines between the matched points in the warping path
for (i, j) in warping_path:
    ax.plot([i, j], [series1[i], series2[j]], 'r-', alpha=0.1)
# Labeling the plot
ax.set_xlabel('Time')
ax.set_ylabel('Log Return')
ax.set_title('DTW Warping Path between Real and Fake Data')
ax.legend()
plt.show()

d

##################################################################
#
# Convert Generated Log Delta Back to Optical Density
#
##################################################################
print("\n" + "="*60)
print("CONVERTING GENERATED LOG DELTA TO OPTICAL DENSITY")
print("="*60)
# Convert generated log delta back to optical density for comparison
# fake_OD_log_delta_np contains the generated log returns (log delta)
# We need to reconstruct the optical density from these log returns
# Get the original optical density data for reference
original_OD_np = OD.detach().cpu().numpy()  # Original optical density values
# Convert generated log delta to numpy if it's a tensor
if isinstance(fake_OD_log_delta_np, torch.Tensor):
    fake_log_delta_np = fake_OD_log_delta_np.detach().cpu().numpy()
else:
    fake_log_delta_np = fake_OD_log_delta_np
print(f"Generated log delta shape: {fake_log_delta_np.shape}")
print(f"Generated log delta range: [{fake_log_delta_np.min():.6f}, {fake_log_delta_np.max():.6f}]")
# Reconstruct optical density from log delta
# Log delta represents: log(OD[t]) - log(OD[t-1])
# So: log(OD[t]) = log(OD[t-1]) + log_delta[t-1]
# And: OD[t] = exp(log(OD[t]))
# Initialize the reconstructed log(OD) array
fake_log_OD = np.zeros(len(fake_log_delta_np) + 1)
# Set the initial value to match the original data's first point
initial_log_OD = np.log(original_OD_np[0])
fake_log_OD[0] = initial_log_OD
# Reconstruct log(OD) using cumulative sum
fake_log_OD[1:] = initial_log_OD + np.cumsum(fake_log_delta_np)
# Convert back to optical density
fake_OD = np.exp(fake_log_OD)
print(f"Reconstructed OD shape: {fake_OD.shape}")
print(f"Reconstructed OD range: [{fake_OD.min():.6f}, {fake_OD.max():.6f}]")
print(f"Original OD range: [{original_OD_np.min():.6f}, {original_OD_np.max():.6f}]")
# Calculate statistics for comparison
print(f"\nStatistical Comparison:")
print(f"Original OD - Mean: {original_OD_np.mean():.6f}, Std: {original_OD_np.std():.6f}")
print(f"Generated OD - Mean: {fake_OD.mean():.6f}, Std: {fake_OD.std():.6f}")
# Calculate correlation between original and generated OD (using overlapping portion)
min_length = min(len(original_OD_np), len(fake_OD))
correlation = np.corrcoef(original_OD_np[:min_length], fake_OD[:min_length])[0, 1]
print(f"Correlation coefficient: {correlation:.6f}")
# Create comprehensive visualization
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
# Plot 1: Log Delta Comparison
axes[0].plot(fake_log_delta_np[:500], label='Generated Log Delta', color='orange', alpha=0.8)
axes[0].plot(OD_log_delta.detach().cpu().numpy()[:500], label='Original Log Delta', color='blue', alpha=0.8)
axes[0].set_title('Log Delta Comparison (First 500 Points)', fontsize=14)
axes[0].set_xlabel('Time Index')
axes[0].set_ylabel('Log Delta Value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
# Plot 2: Optical Density Comparison (Linear Scale)
axes[1].plot(fake_OD[:500], label='Reconstructed OD', color='orange', alpha=0.8)
axes[1].plot(original_OD_np[:500], label='Original OD', color='blue', alpha=0.8)
axes[1].set_title('Optical Density Comparison - Linear Scale (First 500 Points)', fontsize=14)
axes[1].set_xlabel('Time Index')
axes[1].set_ylabel('Optical Density')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
# Plot 3: Optical Density Comparison (Log Scale for better visualization)
axes[2].semilogy(fake_OD[:500], label='Reconstructed OD', color='orange', alpha=0.8)
axes[2].semilogy(original_OD_np[:500], label='Original OD', color='blue', alpha=0.8)
axes[2].set_title('Optical Density Comparison - Log Scale (First 500 Points)', fontsize=14)
axes[2].set_xlabel('Time Index')
axes[2].set_ylabel('Optical Density (Log Scale)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/shawngibford/dev/Pennylane_QGAN/qGAN/optical_density_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
# Save the reconstructed optical density data to CSV
import pandas as pd
# Fix array length mismatch: fake_OD has one more element than fake_log_delta_np
# because we added the initial OD value. We need to align the arrays properly.
log_delta_length = len(fake_log_delta_np)
original_log_delta_np = OD_log_delta.detach().cpu().numpy()
# Use the length of the generated log delta as the reference
comparison_length = min(log_delta_length, len(original_log_delta_np), len(original_OD_np)-1, len(fake_OD)-1)
print(f"Array lengths for DataFrame:")
print(f"  Generated log delta: {len(fake_log_delta_np)}")
print(f"  Original log delta: {len(original_log_delta_np)}")
print(f"  Original OD: {len(original_OD_np)}")
print(f"  Reconstructed OD: {len(fake_OD)}")
print(f"  Using comparison length: {comparison_length}")
# Create DataFrame with aligned arrays
comparison_df = pd.DataFrame({
    'Time_Index': range(comparison_length),
    'Original_OD': original_OD_np[1:comparison_length+1],  # Skip first element to align with log delta
    'Reconstructed_OD': fake_OD[1:comparison_length+1],    # Skip first element to align with log delta
    'Original_Log_Delta': original_log_delta_np[:comparison_length],
    'Generated_Log_Delta': fake_log_delta_np[:comparison_length]
})
# Save to CSV
comparison_df.to_csv('/Users/shawngibford/dev/Pennylane_QGAN/qGAN/generated_optical_density.csv', index=False)
print(f"\nData saved to: generated_optical_density.csv")
print(f"Plots saved to: optical_density_comparison.png")
print("\n" + "="*60)
print("OPTICAL DENSITY CONVERSION COMPLETE")
print("="*60)
##################################################################
#
# Scatter Plot: Original vs Generated Log Delta Comparison
#
##################################################################
print("\n" + "="*50)
print("CREATING LOG DELTA SCATTER PLOT COMPARISON")
print("="*50)
# Create scatter plot comparing original vs generated log delta values
plt.figure(figsize=(10, 8))
# Use the aligned data from the comparison DataFrame for consistency
original_log_delta_scatter = comparison_df['Original_Log_Delta'].values
generated_log_delta_scatter = comparison_df['Generated_Log_Delta'].values
# Create scatter plot
plt.scatter(original_log_delta_scatter, generated_log_delta_scatter, 
           alpha=0.6, s=20, color='blue', edgecolors='navy', linewidth=0.5)
# Add perfect correlation line (y = x) for reference
min_val = min(original_log_delta_scatter.min(), generated_log_delta_scatter.min())
max_val = max(original_log_delta_scatter.max(), generated_log_delta_scatter.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
         label='Perfect Correlation (y=x)', alpha=0.8)
# Calculate and display correlation coefficient
correlation_log_delta = np.corrcoef(original_log_delta_scatter, generated_log_delta_scatter)[0, 1]
# Add labels and formatting
plt.xlabel('Original Log Delta (Log Returns)', fontsize=14)
plt.ylabel('Generated Log Delta (Log Returns)', fontsize=14)
plt.title(f'Original vs Generated Log Delta Comparison\nCorrelation: {correlation_log_delta:.4f}', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
# Make the plot square for better visual comparison
plt.axis('equal')
plt.tight_layout()
# Save the scatter plot
plt.savefig('/Users/shawngibford/dev/Pennylane_QGAN/qGAN/log_delta_scatter_comparison.png', 
           dpi=300, bbox_inches='tight')
plt.show()
# Print statistics
print(f"\nLog Delta Scatter Plot Statistics:")
print(f"Correlation coefficient: {correlation_log_delta:.6f}")
print(f"Original log delta range: [{original_log_delta_scatter.min():.6f}, {original_log_delta_scatter.max():.6f}]")
print(f"Generated log delta range: [{generated_log_delta_scatter.min():.6f}, {generated_log_delta_scatter.max():.6f}]")
print(f"Data points plotted: {len(original_log_delta_scatter)}")
print(f"\nScatter plot saved to: log_delta_scatter_comparison.png")
print("\n" + "="*50)
print("LOG DELTA SCATTER PLOT COMPLETE")
print("="*50)

