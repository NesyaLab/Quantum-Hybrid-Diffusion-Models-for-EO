# Quantum-Hybrid-Diffusion-Models-for-EO
<img width="1475" height="617" alt="image" src="https://github.com/user-attachments/assets/2da36116-aaf8-4528-815d-e3a807aefe77" />
**Figure 1. Class-conditioned *Quantum Diffusion Model* for EO image generation.**  
**Top:** The forward diffusion process \( q(x_t \mid x_{t-1}) \) (blue) progressively corrupts a clean RGB satellite patch \( x_0 \) into Gaussian noise \( x_T \).  
The learned reverse process \( p_{\theta}(x_{t-1} \mid x_t, y) \) (purple) reconstructs the image step-by-step while being guided by the land-cover label \( y \), enabling class-specific generation (e.g., *residential*, *forest*, *river*, *crop*).  
**Bottom:** The U-Net denoiser integrates both classical components (Convolutional, ResNet, and Attention blocks) and two quantum-enhanced residual variants.  
The *QuanResNet Block* (red) replaces the first convolution with a quanvolutional quantum filter applied to local spatial patches.  
The *QResNet Block* (purple), placed at the bottleneck, replaces both convolutions with a Variational Quantum Circuit (VQC) acting on \( 2 \times 2 \times 3 \) feature blocks and applied to a fraction \( \rho \) of channels.
## Overview

This repository contains an experimental framework for quantum hybrid diffusion models applied to Earth Observation (EO) data.

## Project Structure

* configs/ — Configuration files for different experimental setups (datasets, model size, noise schedule, quantum layers)
* main.py — Main script to run training and sampling
* sampling.py — Sampling routines and preprocessing utilities for EO data
* train.py — Training loop and optimization of the quantum hybrid diffusion model
* unet.py — U-Net implementation used as the backbone of the denoising network
* utils.py — Utility functions

## Running the Model

To launch an experiment, simply specify the desired configuration file:

```bash
python main.py --config configs/eurosat.py
```

## Quantum Hybrid Diffusion Pipeline

This project implements a denoising diffusion probabilistic model (DDPM) extended with variational quantum circuits. The overall pipeline is composed of the following stages:

* **Forward diffusion**: Gaussian noise is progressively added to the EO images following the standard DDPM formulation.

* **Denoising U-Net**: A Quanvolutional Conditioned U-Net (QCU-Net) is used as the backbone to estimate the noise component to be removed.
  The model adopts a hybrid quantum–classical architecture, where variational quantum circuits are integrated into the feature extraction stages through a novel quanvolutional approach within a conditioned diffusion framework.

* **Reverse diffusion sampling**: Starting from a random Gaussian input, the learned score function is iteratively applied to denoise the sample and reconstruct a class-conditioned EO image.

---

## Training Procedure Overview

The following pseudocode describes the training procedure of the proposed QCU-Net model.

```text
## Pseudocode

PROCEDURE Train(Dataset_EO):

    Initialize classical weights W
    Initialize quantum parameters Θ

    FOR each training iteration (x₀, label) DO:

        Sample t ~ Uniform({1, ..., T})
        Sample ε ~ Normal(0, 1)

        x_t ← √(α_t) · x₀ + √(1 − α_t) · ε

        ε_pred ← QCU-Net(x_t, label, t, Θ, W)

        loss ← MSE(ε, ε_pred)

        Update Θ and W using Adam optimizer

    END FOR

  
