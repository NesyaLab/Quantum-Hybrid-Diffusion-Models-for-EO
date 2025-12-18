# Quantum-Hybrid-Diffusion-Models-for-EO

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

This project implements a denoising diffusion probabilistic model (DDPM) extended with variational quantum circuits. The flow is as follows:

Forward diffusion: Noise is gradually added to the EO image following the standard DDPM process.

Denoising U-Net: A Quanvolutional Conditioned U-Net serves as the backbone for estimating the noise to be removed. This model is a hybrid quantum-classical architecture which applies operations to feature extraction stages via a novel quanvolutional approach within a conditioned diffusion framework.

Reverse diffusion sampling: Using the learned score function, the model iteratively denoises a random Gaussian input to reconstruct a class-specific EO image.

## Training Procedure Overview

The following pseudocode describes the training procedure of the proposed QCU-Net model.

PROCEDURE Train(Dataset_EO):
  Initialize Classical Weights W
  Initialize Quantum Theta

  For each iteration (x_0, label):
    t ~ Uniform({1,...T})
    ε ~  N(0,1)
    x_t= sqrt(α_t)*x_0 + sqrt(1- α_t)*ε
    ε_pred= QCU-Net(x_t, label, t, Theta, W)
    loss= MSE(ε, ε_pred)
    update Theta, W using Adam Optimizer

  
