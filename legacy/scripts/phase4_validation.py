#!/usr/bin/env python3
"""
Phase 4 Validation Run (200 epochs)
Validates that HPO hyperparameters transfer to corrected [0, 4pi] noise range.

Standalone script that replicates the notebook preprocessing pipeline,
creates a fresh qGAN instance, trains for 200 epochs, and saves
comprehensive baseline metrics to results/phase4_validation.json.

Usage:
    cd /Users/shawngibford/dev/phd/qGAN
    source qgan_env/bin/activate
    python scripts/phase4_validation.py
"""

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
from scipy.special import lambertw
from scipy.stats import kurtosis, wasserstein_distance

# PennyLane imports
import pennylane as qml

# Reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# ─── Configuration ───────────────────────────────────────────────────────────

VAL_EPOCHS = 200
VAL_LR_CRITIC = 1.8046e-05
VAL_LR_GEN = 6.9173e-05
VAL_LAMBDA_GP = 2.16
VAL_N_CRITIC = 9

NUM_QUBITS = 5
NUM_LAYERS = 4
WINDOW_LENGTH = 2 * NUM_QUBITS  # 10
BATCH_SIZE = 12
EVAL_EVERY = 10

# HPO baseline for comparison
HPO_BASELINE_EMD = 0.001137
THRESHOLD_2X = 0.002274

# v1.0 fallback defaults
FALLBACK_LAMBDA_GP = 10
FALLBACK_N_CRITIC = 5
FALLBACK_LR_CRITIC = 3e-5
FALLBACK_LR_GEN = 8e-5

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# ─── Data Loading (replicates cells 5, 7, 9, 15, 17, 18, 21, 30) ─────────

print("Loading data...")
full_data = pd.read_csv("./data.csv")

raw_data = full_data[["OD"]].copy()
raw_data.columns = ["value"]
raw_data["value"] = pd.to_numeric(raw_data["value"], errors="coerce")
raw_data["value"] = raw_data["value"].fillna(
    raw_data["value"].rolling(window=10, min_periods=10).mean()
)
raw_data = raw_data.dropna()
OD = torch.tensor(raw_data["value"].values, dtype=torch.float32)

PAR_LIGHT = torch.tensor(full_data["PAR_LIGHT"].values, dtype=torch.float32)


def normalize(data):
    mu = torch.mean(data)
    sigma = torch.std(data)
    return (data - mu) / sigma, mu, sigma


def compute_log_delta(od_values, dither=0.0, rng=None):
    od_np = od_values.numpy() if isinstance(od_values, torch.Tensor) else od_values.copy()
    if dither > 0:
        if rng is None:
            rng = np.random.default_rng()
        od_np = od_np + rng.uniform(-dither, dither, size=len(od_np))
    log_od = np.log(od_np)
    return torch.tensor(log_od[1:] - log_od[:-1], dtype=torch.float32)


def inverse_lambert_w_transform(data, delta):
    data = data.double()
    sign = torch.sign(data)
    data_squared = data ** 2
    lambert_input = (delta * data_squared).cpu().numpy()
    lambert_result = lambertw(lambert_input).real
    lambert_tensor = torch.tensor(lambert_result, dtype=torch.float64, device=data.device)
    transformed_data = sign * torch.sqrt(lambert_tensor / delta)
    return transformed_data


def lambert_w_transform(transformed_data, delta, clip_low=-12.0, clip_high=11.0):
    transformed_data = transformed_data.double()
    exp_term = torch.exp((delta / 2) * transformed_data ** 2)
    reversed_data = transformed_data * exp_term
    return torch.clamp(reversed_data, clip_low, clip_high)


def rescale(scaled_data, original_data):
    min_val = torch.min(original_data)
    max_val = torch.max(original_data)
    previous_data = 0.5 * (scaled_data + 1.0) * (max_val - min_val) + min_val
    return previous_data


def rolling_window(data, m, s):
    windows = []
    for i in range(0, len(data) - m + 1, s):
        windows.append(data[i : i + m])
    return torch.stack(windows)


# Compute log-delta
DITHER = 0.005
DITHER_SEED = 42
log_delta = compute_log_delta(OD, dither=DITHER, rng=np.random.default_rng(DITHER_SEED))

# PAR_LIGHT aligned with log-returns
par_light_aligned = PAR_LIGHT[1:]
PAR_LIGHT_MAX = 12.5
par_light_norm = par_light_aligned / PAR_LIGHT_MAX

# Normalize
norm_log_delta, mu, sigma = normalize(log_delta)

# Find optimal Lambert W delta
from scipy.optimize import minimize_scalar


def kurtosis_for_delta(d, data):
    sign = np.sign(data)
    lr = lambertw(d * data ** 2).real
    lr = np.maximum(lr, 0)
    transformed = sign * np.sqrt(lr / d)
    return abs(kurtosis(transformed, fisher=True))


normed_np = norm_log_delta.numpy()
result = minimize_scalar(
    kurtosis_for_delta, bounds=(0.01, 2.0), args=(normed_np,), method="bounded"
)
delta = result.x
print(f"Optimal Lambert W delta: {delta:.4f}")

# Apply inverse Lambert W transform
transformed_norm_log_delta = inverse_lambert_w_transform(norm_log_delta, delta)

# Scale to [-1, 1]
min_val = torch.min(transformed_norm_log_delta)
max_val = torch.max(transformed_norm_log_delta)
scaled_data = -1.0 + 2.0 * (transformed_norm_log_delta - min_val) / (max_val - min_val)

# Scale PAR_LIGHT to [-1, 1]
par_light_scaled = -1.0 + 2.0 * par_light_norm

# Rolling windows
windowed_data = rolling_window(scaled_data, WINDOW_LENGTH, 2)
windowed_par = rolling_window(par_light_scaled, WINDOW_LENGTH, 2)

data_tensor = windowed_data.float()
par_tensor = windowed_par.float()

dataset = torch.utils.data.TensorDataset(data_tensor, par_tensor)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
)
num_elements = len(windowed_data)

print(f"Windowed data shape: {data_tensor.shape}")
print(f"Number of windows: {num_elements}")
print(f"mu={mu.item():.6f}, sigma={sigma.item():.6f}")


# ─── Model Definition (replicates cell 26 qGAN class) ───────────────────────


class qGAN(nn.Module):
    def __init__(
        self, num_epochs, batch_size, window_length, n_critic, gp, num_layers, num_qubits, delta=1
    ):
        super().__init__()

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.window_length = window_length
        self.n_critic = n_critic
        self.gp = gp
        self.delta = delta
        self.eval_every = EVAL_EVERY

        self.num_layers = num_layers
        self.num_qubits = num_qubits

        self.quantum_dev = qml.device("default.qubit", wires=self.num_qubits)
        self.torch_device = torch.device("cpu")

        self.qubits = list(range(num_qubits))

        self.measurements = []
        for i in range(self.num_qubits):
            self.measurements.append(qml.expval(qml.PauliX(i)))
            self.measurements.append(qml.expval(qml.PauliZ(i)))

        self.num_params = self.count_params()

        self.params_pqc = torch.nn.Parameter(
            torch.randn(self.num_params, requires_grad=True, dtype=torch.float32) * 0.5
        )

        self.critic = self.define_critic_model(window_length)
        self.generator = self.define_generator_model()

        self.critic_loss_avg = []
        self.generator_loss_avg = []
        self.emd_avg = []
        self.acf_avg = []
        self.vol_avg = []
        self.lev_avg = []

    def count_params(self):
        iqp_params = self.num_qubits
        rotation_params_per_layer = self.num_qubits * 3
        params_per_layer = rotation_params_per_layer
        main_params = self.num_layers * params_per_layer
        final_params = self.num_qubits * 2
        return iqp_params + main_params + final_params

    def define_critic_model(self, window_length):
        model = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=10, stride=1, padding=5),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, stride=1, padding=5),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=10, stride=1, padding=5),
            nn.LeakyReLU(negative_slope=0.1),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=32, out_features=1),
        )
        model = model.double()
        return model

    def encoding_layer(self, noise_params):
        for i in range(min(len(noise_params), self.num_qubits)):
            qml.RZ(noise_params[i], wires=i)

    def par_light_encoding(self, par_light_params):
        for i in range(self.num_qubits):
            qml.RY(par_light_params[i] * np.pi, wires=i)

    def define_generator_circuit(self, noise_params, par_light_params, params_pqc):
        idx = 0

        for qubit in range(self.num_qubits):
            qml.Hadamard(wires=qubit)

        for qubit in range(self.num_qubits):
            if idx < len(params_pqc):
                qml.RZ(phi=params_pqc[idx], wires=qubit)
                idx += 1

        self.encoding_layer(noise_params)
        self.par_light_encoding(par_light_params)

        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                if idx + 2 < len(params_pqc):
                    qml.Rot(
                        phi=params_pqc[idx],
                        theta=params_pqc[idx + 1],
                        omega=params_pqc[idx + 2],
                        wires=qubit,
                    )
                    idx += 3

            if self.num_qubits > 1:
                range_param = (layer % (self.num_qubits - 1)) + 1
                for qubit in range(self.num_qubits):
                    target_qubit = (qubit + range_param) % self.num_qubits
                    qml.CNOT(wires=[qubit, target_qubit])

        for qubit in range(self.num_qubits):
            if idx + 1 < len(params_pqc):
                qml.RX(phi=params_pqc[idx], wires=qubit)
                idx += 1
                qml.RY(phi=params_pqc[idx], wires=qubit)
                idx += 1

        measurements = []
        for i in range(self.num_qubits):
            measurements.append(qml.expval(qml.PauliX(i)))
            measurements.append(qml.expval(qml.PauliZ(i)))

        return (*measurements,)

    def define_generator_model(self):
        generator = qml.QNode(
            self.define_generator_circuit,
            self.quantum_dev,
            interface="torch",
            diff_method="parameter-shift",
        )
        return generator

    def compile_QGAN(self, c_optimizer, g_optimizer):
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer

    def train_qgan(self, gan_data, original_data, preprocessed_data, num_elements, early_stopper=None):
        gan_data_list = []
        par_data_list = []

        for batch in gan_data:
            log_return_batch, par_light_batch = batch[0], batch[1]
            for sample in log_return_batch:
                gan_data_list.append(sample)
            for sample in par_light_batch:
                par_data_list.append(sample)

        for epoch in range(self.num_epochs):
            print(f"Processing epoch {epoch+1}/{self.num_epochs}")
            self._train_one_epoch(
                gan_data_list, par_data_list, original_data, preprocessed_data, epoch
            )

            eval_every = getattr(self, "eval_every", 10)
            if (
                early_stopper is not None
                and (epoch % eval_every == 0 or epoch + 1 == self.num_epochs)
                and len(self.emd_avg) > 0
            ):
                should_stop = early_stopper.check(epoch, self.emd_avg[-1], self, mu, sigma)
                if should_stop:
                    print(f"Training stopped early at epoch {epoch+1}")
                    break

    def _train_one_epoch(self, gan_data_list, par_data_list, original_data, preprocessed_data, epoch):
        # ── Critic training ──
        critic_t_sum = 0
        for t in range(self.n_critic):
            self.c_optimizer.zero_grad()

            real_batch = []
            fake_batch = []

            for i in range(self.batch_size):
                random_idx = torch.randint(0, len(gan_data_list), (1,)).item()
                real_log_return = gan_data_list[random_idx]
                real_log_return = torch.reshape(real_log_return, (1, self.window_length))

                par_window = par_data_list[random_idx]
                par_window = torch.reshape(par_window, (1, self.window_length))

                real_sample = torch.stack([real_log_return, par_window], dim=1).double()
                real_batch.append(real_sample)

                with torch.no_grad():
                    noise_values = np.random.uniform(0, 4 * np.pi, size=self.num_qubits)
                    generator_input = torch.tensor(noise_values, dtype=torch.float32)

                    par_for_circuit = par_window.reshape(self.num_qubits, 2).mean(dim=1).float()
                    par_for_circuit = (par_for_circuit + 1.0) / 2.0

                    generated_sample = self.generator(
                        generator_input, par_for_circuit, self.params_pqc
                    )

                    if isinstance(generated_sample, (list, tuple)):
                        generated_sample = torch.stack(list(generated_sample))
                    elif not isinstance(generated_sample, torch.Tensor):
                        generated_sample = torch.tensor(generated_sample)

                    generated_sample = generated_sample.to(torch.float64)
                    generated_sample = generated_sample * 0.1

                    gen_reshaped = generated_sample.unsqueeze(0)
                    generated_sample_input = torch.stack(
                        [gen_reshaped, par_window.double()], dim=1
                    )
                fake_batch.append(generated_sample_input)

            real_batch_tensor = torch.cat(real_batch, dim=0)
            fake_batch_tensor = torch.cat(fake_batch, dim=0)

            real_scores = self.critic(real_batch_tensor)
            fake_scores = self.critic(fake_batch_tensor)

            real_score_mean = torch.mean(real_scores)
            fake_score_mean = torch.mean(fake_scores)

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
                only_inputs=True,
            )[0]

            gradient_penalty = torch.mean((gradients.norm(2, dim=[1, 2]) - 1) ** 2)

            critic_loss = fake_score_mean - real_score_mean + self.gp * gradient_penalty
            critic_sum = critic_loss

            critic_sum.backward()
            self.c_optimizer.step()

            critic_t_sum += critic_sum

        self.critic_loss_avg.append(critic_t_sum / self.n_critic)

        # ── Generator training ──
        input_circuits_batch = []
        par_circuits_batch = []
        par_windows_batch = []
        for _ in range(self.batch_size):
            noise_values = np.random.uniform(0, 4 * np.pi, size=self.num_qubits)
            input_circuits_batch.append(noise_values)
            random_idx = torch.randint(0, len(par_data_list), (1,)).item()
            par_window = par_data_list[random_idx]
            par_window = torch.reshape(par_window, (self.window_length,))
            par_windows_batch.append(par_window)
            par_compressed = par_window.reshape(self.num_qubits, 2).mean(dim=1).float()
            par_compressed = (par_compressed + 1.0) / 2.0
            par_circuits_batch.append(par_compressed)

        generator_inputs = torch.stack(
            [torch.tensor(noise, dtype=torch.float32) for noise in input_circuits_batch]
        )

        self.g_optimizer.zero_grad()

        generated_samples = []
        for i in range(generator_inputs.shape[0]):
            gen_out = self.generator(generator_inputs[i], par_circuits_batch[i], self.params_pqc)
            if isinstance(gen_out, (list, tuple)):
                gen_out = torch.stack(gen_out)
            gen_out = gen_out.to(torch.float64) * 0.1
            generated_samples.append(gen_out)

        generated_samples = torch.stack(generated_samples)

        par_windows_tensor = torch.stack(par_windows_batch).double()
        generated_samples_input = torch.stack(
            [generated_samples, par_windows_tensor], dim=1
        )

        fake_scores = self.critic(generated_samples_input)

        generator_loss_wgan = -torch.mean(fake_scores)
        generator_loss = generator_loss_wgan

        generator_loss.backward()
        self.g_optimizer.step()

        self.generator_loss_avg.append(generator_loss)

        # ── Evaluation ──
        eval_every = getattr(self, "eval_every", 10)
        if epoch % eval_every == 0 or epoch + 1 == self.num_epochs:
            num_samples = len(original_data) // self.window_length
            input_circuits_batch = []

            for _ in range(num_samples):
                noise_values = np.random.uniform(0, 4 * np.pi, size=self.num_qubits)
                input_circuits_batch.append(noise_values)

            generator_inputs = torch.stack(
                [torch.tensor(noise, dtype=torch.float32) for noise in input_circuits_batch]
            )

            batch_generated = []
            for j, generator_input in enumerate(generator_inputs):
                random_idx = torch.randint(0, len(par_data_list), (1,)).item()
                par_window = par_data_list[random_idx]
                par_for_circuit = par_window.reshape(self.num_qubits, 2).mean(dim=1).float()
                par_for_circuit = (par_for_circuit + 1.0) / 2.0
                gen_out = self.generator(generator_input, par_for_circuit, self.params_pqc)
                if isinstance(gen_out, (list, tuple)):
                    gen_out = torch.stack(list(gen_out))
                gen_out = gen_out.to(torch.float64) * 0.1
                batch_generated.append(gen_out)

            batch_generated = torch.stack(batch_generated)

            generated_data = torch.reshape(
                batch_generated, shape=(num_samples * self.window_length,)
            )
            generated_data = generated_data.double()

            generated_data = rescale(generated_data, preprocessed_data)
            original_norm = lambert_w_transform(generated_data, delta)
            fake_original = original_norm

            corr_rmse, volatility_rmse, lev_rmse, emd = self.stylized_facts(
                original_data, fake_original
            )

            self.acf_avg.append(corr_rmse)
            self.vol_avg.append(volatility_rmse)
            self.lev_avg.append(lev_rmse)
            self.emd_avg.append(emd)

            print(f"\n[Eval] Epoch {epoch+1}")
            print(f"  EMD: {emd:.6f}")
            print(f"  ACF RMSE: {corr_rmse:.6f}")
            print(f"  VOL RMSE: {volatility_rmse:.6f}")
            print(f"  LEV RMSE: {lev_rmse:.6f}")
            print(
                f"  Generated range: [{torch.min(fake_original).item():.6f}, {torch.max(fake_original).item():.6f}]"
            )

        if epoch % 100 == 0 or epoch + 1 == self.num_epochs:
            print(f"\nEpoch {epoch+1} completed")
            if len(self.critic_loss_avg) > epoch:
                critic_loss_val = self.critic_loss_avg[epoch]
                if hasattr(critic_loss_val, "item"):
                    critic_loss_val = critic_loss_val.item()
                print(f"  Critic loss: {critic_loss_val}")
            if len(self.generator_loss_avg) > epoch:
                generator_loss_val = self.generator_loss_avg[epoch]
                if hasattr(generator_loss_val, "item"):
                    generator_loss_val = generator_loss_val.item()
                print(f"  Generator loss: {generator_loss_val}")
            print(
                f"  Original log-delta range: [{torch.min(original_data).item():.6f}, {torch.max(original_data).item():.6f}]"
            )

    def stylized_facts(self, original_data, fake_original):
        if isinstance(fake_original, torch.Tensor):
            fake_np = fake_original.detach().cpu().numpy()
        else:
            fake_np = np.asarray(fake_original)
        if isinstance(original_data, torch.Tensor):
            orig_np = original_data.detach().cpu().numpy()
        else:
            orig_np = np.asarray(original_data)

        acf_values = sm.tsa.acf(fake_np, nlags=18)
        acf_values_generated = torch.tensor(acf_values[1:])

        acf_abs_values = sm.tsa.acf(np.abs(fake_np), nlags=18)
        acf_abs_values_generated = torch.tensor(acf_abs_values[1:])

        lev = []
        for lag in range(1, 19):
            r_t = fake_np[:-lag]
            squared_lag_r = np.square(np.abs(fake_np[lag:]))
            correlation_matrix = np.corrcoef(r_t, squared_lag_r)
            lev.append(correlation_matrix[0, 1])
        leverage_generated = torch.tensor(lev)

        acf_values = sm.tsa.acf(orig_np, nlags=18)
        acf_values_original = torch.tensor(acf_values[1:])

        acf_abs_values = sm.tsa.acf(np.abs(orig_np), nlags=18)
        acf_abs_values_original = torch.tensor(acf_abs_values[1:])

        lev = []
        for lag in range(1, 19):
            r_t = orig_np[:-lag]
            squared_lag_r = np.square(np.abs(orig_np[lag:]))
            correlation_matrix = np.corrcoef(r_t, squared_lag_r)
            lev.append(correlation_matrix[0, 1])
        leverage_original = torch.tensor(lev)

        rmse_acf = torch.sqrt(torch.mean((acf_values_original - acf_values_generated) ** 2))
        rmse_vol = torch.sqrt(
            torch.mean((acf_abs_values_original - acf_abs_values_generated) ** 2)
        )
        rmse_lev = torch.sqrt(torch.mean((leverage_original - leverage_generated) ** 2))

        bin_edges = np.linspace(-0.05, 0.05, num=50)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)

        empirical_real, _ = np.histogram(orig_np, bins=bin_edges, density=True)
        empirical_real /= np.sum(empirical_real)

        empirical_fake, _ = np.histogram(fake_np, bins=bin_edges, density=True)
        empirical_fake /= np.sum(empirical_fake)

        emd = wasserstein_distance(empirical_real, empirical_fake)
        return rmse_acf, rmse_vol, rmse_lev, emd


# ─── EarlyStopping (replicates cell 31) ─────────────────────────────────────


class EarlyStopping:
    def __init__(self, patience=50, warmup_epochs=100, checkpoint_path="best_checkpoint.pt"):
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.checkpoint_path = checkpoint_path
        self.best_emd = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def check(self, epoch, emd, model, mu_val, sigma_val):
        if epoch < self.warmup_epochs:
            return False

        if emd < self.best_emd:
            self.best_emd = emd
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(epoch, emd, model, mu_val, sigma_val)
            print(f"  [ES] New best EMD: {emd:.6f} at epoch {epoch+1}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n[Early Stopping] Triggered at epoch {epoch+1}")
                print(f"  Best EMD: {self.best_emd:.6f} at epoch {self.best_epoch+1}")
                print(
                    f"  Patience exhausted: {self.counter}/{self.patience} eval cycles without improvement"
                )
                self._load_checkpoint(model)
                return True
        return False

    def _save_checkpoint(self, epoch, emd, model, mu_val, sigma_val):
        checkpoint = {
            "epoch": epoch,
            "emd": emd,
            "params_pqc": model.params_pqc.detach().clone(),
            "critic_state": model.critic.state_dict(),
            "c_optimizer": model.c_optimizer.state_dict(),
            "g_optimizer": model.g_optimizer.state_dict(),
            "mu": mu_val,
            "sigma": sigma_val,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def _load_checkpoint(self, model):
        checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        model.params_pqc.data = checkpoint["params_pqc"]
        model.critic.load_state_dict(checkpoint["critic_state"])
        model.c_optimizer.load_state_dict(checkpoint["c_optimizer"])
        model.g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        model.g_optimizer.param_groups[0]["params"] = [model.params_pqc]
        print(
            f'  [ES] Loaded best checkpoint from epoch {checkpoint["epoch"]+1} (EMD: {checkpoint["emd"]:.6f})'
        )


# ─── Training Execution ─────────────────────────────────────────────────────


def run_validation(lr_critic, lr_gen, lambda_gp, n_critic, label="HPO"):
    """Run a validation training session and return the model + early stopper."""
    print(f"\n{'='*60}")
    print(f"Phase 4 Validation: {VAL_EPOCHS} epochs, noise range [0, 4pi]")
    print(f"{label} params: lr_c={lr_critic:.2e}, lr_g={lr_gen:.2e}, lambda_gp={lambda_gp}, n_critic={n_critic}")
    print(f"{'='*60}\n")

    qgan_val = qGAN(
        num_epochs=VAL_EPOCHS,
        batch_size=BATCH_SIZE,
        window_length=WINDOW_LENGTH,
        n_critic=n_critic,
        gp=lambda_gp,
        num_layers=NUM_LAYERS,
        num_qubits=NUM_QUBITS,
        delta=delta,
    )

    c_opt = torch.optim.Adam(qgan_val.critic.parameters(), lr=lr_critic, betas=(0.0, 0.9))
    g_opt = torch.optim.Adam([qgan_val.params_pqc], lr=lr_gen, betas=(0.0, 0.9))
    qgan_val.compile_QGAN(c_opt, g_opt)

    early_stopper = EarlyStopping(
        patience=50,
        warmup_epochs=50,
        checkpoint_path="results/phase4_validation_checkpoint.pt",
    )

    start = time.time()
    qgan_val.train_qgan(
        dataloader, log_delta, transformed_norm_log_delta, num_elements, early_stopper=early_stopper
    )
    elapsed = time.time() - start

    return qgan_val, early_stopper, elapsed


def check_divergence(model):
    """Check if training diverged (NaN or critic loss > 1000)."""
    if len(model.critic_loss_avg) == 0:
        return True
    last_critic = model.critic_loss_avg[-1]
    if hasattr(last_critic, "item"):
        last_critic = last_critic.item()
    if math.isnan(last_critic) or abs(last_critic) > 1000:
        return True
    if len(model.emd_avg) > 0:
        last_emd = model.emd_avg[-1]
        if isinstance(last_emd, float) and math.isnan(last_emd):
            return True
    return False


# ─── Main Execution ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    fallback_used = False

    # Primary attempt with HPO parameters
    try:
        qgan_val, early_stopper_val, elapsed = run_validation(
            VAL_LR_CRITIC, VAL_LR_GEN, VAL_LAMBDA_GP, VAL_N_CRITIC, label="HPO"
        )

        if check_divergence(qgan_val):
            print("\nDivergence detected! Falling back to v1.0 defaults...")
            fallback_used = True
            qgan_val, early_stopper_val, elapsed = run_validation(
                FALLBACK_LR_CRITIC,
                FALLBACK_LR_GEN,
                FALLBACK_LAMBDA_GP,
                FALLBACK_N_CRITIC,
                label="v1.0-fallback",
            )
    except Exception as e:
        print(f"\nTraining failed: {e}")
        print("Auto-fallback to v1.0 defaults...")
        fallback_used = True
        qgan_val, early_stopper_val, elapsed = run_validation(
            FALLBACK_LR_CRITIC,
            FALLBACK_LR_GEN,
            FALLBACK_LAMBDA_GP,
            FALLBACK_N_CRITIC,
            label="v1.0-fallback",
        )

    # ─── Capture Metrics ─────────────────────────────────────────────────

    # EMD
    if (
        hasattr(early_stopper_val, "best_emd")
        and early_stopper_val.best_emd is not None
        and early_stopper_val.best_emd < float("inf")
    ):
        final_emd = early_stopper_val.best_emd
    else:
        final_emd = qgan_val.emd_avg[-1] if len(qgan_val.emd_avg) > 0 else float("nan")

    # Generate fake samples for moment/PSD analysis
    print("\nGenerating samples for baseline metrics...")
    num_samples = len(log_delta) // WINDOW_LENGTH
    fake_windows = []
    real_windows_for_psd = []

    with torch.no_grad():
        gan_data_list = []
        par_data_list = []
        for batch in dataloader:
            for s in batch[0]:
                gan_data_list.append(s)
            for s in batch[1]:
                par_data_list.append(s)

        for _ in range(num_samples):
            noise = np.random.uniform(0, 4 * np.pi, size=NUM_QUBITS)
            gen_input = torch.tensor(noise, dtype=torch.float32)
            rand_idx = torch.randint(0, len(par_data_list), (1,)).item()
            par_w = par_data_list[rand_idx]
            par_c = par_w.reshape(NUM_QUBITS, 2).mean(dim=1).float()
            par_c = (par_c + 1.0) / 2.0
            gen_out = qgan_val.generator(gen_input, par_c, qgan_val.params_pqc)
            if isinstance(gen_out, (list, tuple)):
                gen_out = torch.stack(list(gen_out))
            gen_out = gen_out.to(torch.float64) * 0.1
            fake_windows.append(gen_out)

    fake_flat = torch.cat(fake_windows).detach()
    # Denormalize fake samples for moment comparison
    fake_denorm = rescale(fake_flat, transformed_norm_log_delta)
    fake_denorm = lambert_w_transform(fake_denorm, delta)

    real_np = log_delta.numpy()
    fake_np = fake_denorm.detach().cpu().numpy()

    # Moment statistics
    from scipy.stats import kurtosis as scipy_kurtosis

    real_moments = {
        "mean": float(np.mean(real_np)),
        "std": float(np.std(real_np)),
        "kurtosis": float(scipy_kurtosis(real_np, fisher=True)),
    }
    fake_moments = {
        "mean": float(np.mean(fake_np)),
        "std": float(np.std(fake_np)),
        "kurtosis": float(scipy_kurtosis(fake_np, fisher=True)),
    }

    # PSD baseline -- store full arrays (only 6 bins for T=10, negligible storage)
    real_windows_tensor = torch.tensor(
        real_np[: (len(real_np) // WINDOW_LENGTH) * WINDOW_LENGTH].reshape(-1, WINDOW_LENGTH),
        dtype=torch.float64,
    )
    fake_windows_tensor = torch.stack(fake_windows).detach()
    # Denormalize fake windows for PSD
    fake_windows_denorm = []
    for w in fake_windows:
        w_denorm = rescale(w.detach(), transformed_norm_log_delta)
        w_denorm = lambert_w_transform(w_denorm, delta)
        fake_windows_denorm.append(w_denorm)
    fake_windows_denorm_tensor = torch.stack(fake_windows_denorm)

    # Compute PSD: average over windows
    real_psd = torch.abs(torch.fft.rfft(real_windows_tensor, dim=1)).pow(2).mean(dim=0)
    fake_psd = torch.abs(torch.fft.rfft(fake_windows_denorm_tensor, dim=1)).pow(2).mean(dim=0)

    # PSD summary stats
    total_power_ratio = float((fake_psd.sum() / real_psd.sum()).item())
    real_peak_freq = int(torch.argmax(real_psd[1:]).item()) + 1  # skip DC
    fake_peak_freq = int(torch.argmax(fake_psd[1:]).item()) + 1
    peak_freq_match = real_peak_freq == fake_peak_freq

    psd_metrics = {
        "real_psd": real_psd.tolist(),
        "fake_psd": fake_psd.tolist(),
        "total_power_ratio": total_power_ratio,
        "real_peak_freq_bin": real_peak_freq,
        "fake_peak_freq_bin": fake_peak_freq,
        "peak_freq_match": peak_freq_match,
        "num_freq_bins": len(real_psd.tolist()),
    }

    # Training dynamics
    final_critic_loss = qgan_val.critic_loss_avg[-1] if len(qgan_val.critic_loss_avg) > 0 else None
    final_gen_loss = qgan_val.generator_loss_avg[-1] if len(qgan_val.generator_loss_avg) > 0 else None

    if final_critic_loss is not None and hasattr(final_critic_loss, "item"):
        final_critic_loss = float(final_critic_loss.item())
    if final_gen_loss is not None and hasattr(final_gen_loss, "item"):
        final_gen_loss = float(final_gen_loss.item())

    epochs_completed = len(qgan_val.critic_loss_avg)

    # Determine outcome
    if fallback_used:
        outcome = "FALLBACK_USED"
    elif isinstance(final_emd, float) and final_emd <= THRESHOLD_2X:
        outcome = "PASS"
    else:
        outcome = "FLAG_FOR_HPO_RERUN"

    # Git hash
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
    except Exception:
        git_hash = "unknown"

    # ─── Build and Save JSON ─────────────────────────────────────────────

    validation_results = {
        "phase": "4-code-regression-fixes",
        "timestamp": datetime.now().isoformat(),
        "git_hash": git_hash,
        "config": {
            "noise_range": [0, "4*pi"],
            "lr_critic": VAL_LR_CRITIC if not fallback_used else FALLBACK_LR_CRITIC,
            "lr_generator": VAL_LR_GEN if not fallback_used else FALLBACK_LR_GEN,
            "lambda_gp": VAL_LAMBDA_GP if not fallback_used else FALLBACK_LAMBDA_GP,
            "n_critic": VAL_N_CRITIC if not fallback_used else FALLBACK_N_CRITIC,
            "lambda_acf": 0,
            "epochs": VAL_EPOCHS,
            "epochs_completed": epochs_completed,
            "batch_size": BATCH_SIZE,
            "num_qubits": NUM_QUBITS,
            "num_layers": NUM_LAYERS,
            "window_length": WINDOW_LENGTH,
            "eval_every": EVAL_EVERY,
            "early_stopping_patience": 50,
            "early_stopping_warmup": 50,
            "fallback_used": fallback_used,
        },
        "metrics": {
            "emd": float(final_emd),
            "moments": {
                "real": real_moments,
                "fake": fake_moments,
            },
            "psd": psd_metrics,
            "training_dynamics": {
                "final_critic_loss": final_critic_loss,
                "final_generator_loss": final_gen_loss,
                "elapsed_seconds": round(elapsed, 2),
                "epochs_completed": epochs_completed,
                "emd_history": [float(e) for e in qgan_val.emd_avg],
                "best_emd_epoch": early_stopper_val.best_epoch + 1,
            },
        },
        "hpo_baseline": {
            "best_emd": HPO_BASELINE_EMD,
            "threshold_2x": THRESHOLD_2X,
        },
        "outcome": outcome,
    }

    with open("results/phase4_validation.json", "w") as f:
        json.dump(validation_results, f, indent=2)

    # ─── Print Summary ───────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("PHASE 4 VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Outcome: {outcome}")
    print(f"Final EMD: {final_emd:.6f}")
    print(f"HPO Baseline EMD: {HPO_BASELINE_EMD:.6f}")
    print(f"Threshold (2x): {THRESHOLD_2X:.6f}")
    print(f"Fallback used: {fallback_used}")
    print(f"Epochs completed: {epochs_completed}/{VAL_EPOCHS}")
    print(f"Elapsed: {elapsed/60:.1f} minutes")
    print(f"\nMoments (real):  mean={real_moments['mean']:.6f}, std={real_moments['std']:.6f}, kurt={real_moments['kurtosis']:.4f}")
    print(f"Moments (fake):  mean={fake_moments['mean']:.6f}, std={fake_moments['std']:.6f}, kurt={fake_moments['kurtosis']:.4f}")
    print(f"PSD power ratio: {total_power_ratio:.4f}")
    print(f"Peak freq match: {peak_freq_match}")
    print(f"\nResults saved to: results/phase4_validation.json")
