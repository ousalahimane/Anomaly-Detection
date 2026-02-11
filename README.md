# Traffic Anomaly Detection using VAE–LSTM (PeMS Dataset)

This repository presents an **unsupervised deep learning framework** for **traffic anomaly detection** using a **hybrid VAE–LSTM architecture**, applied to the **PeMS (Performance Measurement System) Traffic Dataset**.

The approach combines:
- **Variational Autoencoders (VAE)** for spatial and multivariate feature modeling
- **Long Short-Term Memory networks (LSTM)** for temporal dependency modeling

The system is designed to detect abnormal traffic patterns such as sudden congestion, traffic breakdowns, or unusual fluctuations without requiring labeled anomaly data.

---

## Project Objectives

- Detect anomalies in traffic time series data
- Model both **local patterns** and **temporal dynamics**
- Provide interpretable anomaly scores and visualizations
- Enable large-scale multi-station analysis
- Serve as a foundation for future **predictive models** (Temporal GANs, Diffusion Models)

---

## Methodology Overview

### Variational Autoencoder (VAE)
- Learns a latent representation of normal traffic windows
- Detects anomalies using reconstruction error and ELBO loss
- Handles multivariate traffic signals efficiently

### LSTM Network
- Operates on VAE latent embeddings
- Learns normal temporal evolution
- Detects anomalies via prediction and reconstruction errors

### Combined VAE + LSTM Score
- Final anomaly score = VAE score + LSTM reconstruction error
- More robust to noise and temporal drift than single models

---

## Dataset

**PeMS Traffic Dataset**
- Multivariate time series (speed / flow / occupancy)
- Multiple stations across California highways
- Preprocessed into `.npz` files per station

Each `.npz` file contains:
- `X_norm`: normalized traffic signals
- `timestamps`: original timestamps
- `idx_split`: train / test split index
- `train_m`, `train_std`: normalization parameters
- `station_id`

---
## Preprocessing

We pre-processed the **PeMS traffic dataset**, which consists of multivariate traffic time series collected from multiple highway stations (e.g., speed, flow, and occupancy measurements).

For each station, the original continuous time series was split into:
- a **training set** containing only normal traffic behavior,
- and a **test set** used for anomaly detection and evaluation.

The split index is stored explicitly to ensure reproducibility and avoid data leakage.
Code available in PEMS-dataset-preprocessing.ipynb

### Training

1. **VAE training**
   - The VAE is trained on normal traffic windows only.
   - It learns a latent representation of normal traffic patterns.
   - The objective function is based on the ELBO (reconstruction loss + KL divergence).

2. **LSTM training**
   - The LSTM is trained on sequences of VAE latent embeddings.
   - It learns the normal temporal evolution of traffic patterns.
   - 
To train the model, simply run:

```bash
python3 train_vae.py --config PEMS_config.json
```
```bash
python3 train_lstm.py --config PEMS_config.json
```
