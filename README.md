# LFFG: Joint Spatial-Frequency and Angular-Geometry Learning for Light Field Image Denoising

[![Framework](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository is the official implementation of the paper: **"Joint Spatial-Frequency and Angular-Geometry Learning for Light Field Image Denoising"**.

We propose the **Light Field Frequency-Geometry Network (LFFG)**, a lightweight "Physically-Embedded Global Modeling" framework. It achieves state-of-the-art performance by explicitly integrating physical priors into global modeling:
1.  **Spatial Domain:** Uses **Wavelet-guided Spatial-Frequency Attention (WSFA)** as a frequency inductive bias (CNN for high-freq noise, Transformer for low-freq structure).
2.  **Angular Domain:** Uses **Relative Angular Geometry Attention (RAGA)** with Manhattan distance as a geometric inductive bias.

> **Highlight:** LFFG achieves a **0.54 dB** PSNR gain over the SOTA method (HLFRN) on the STFLytro dataset while maintaining a highly compact parameter budget of only **0.73M**.

---

## 🏗️ Architecture

![Architecture](image_597f0f.jpg)
*Figure 1: Visual comparison and Performance vs. Complexity trade-off.*

The LFFG framework employs a cascaded design where features are sequentially processed. The **WSFA** module explicitly disentangles structural information from noise artifacts using Haar Wavelet Transform. The **RAGA** module captures geometric consistency across multiple views using a baseline-aware decay mask. Finally, an **ECDF** block fuses the features.

---

## ⚙️ Requirements

The code is implemented in PyTorch. Please install the dependencies via:

```bash
pip install torch torchvision
pip install tensorboard matplotlib scikit-image imageio scipy einops
