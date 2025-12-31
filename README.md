# Joint Spatial-Frequency and Angular-Geometry Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

This repository contains the PyTorch implementation of the paper: **"Joint Spatial-Frequency and Angular-Geometry Learning for Light Field Image Denoising"** (Submitted to ICME 2026).

---

## 🚀 Introduction

We propose **LFFG**, a lightweight framework that integrates **Wavelet-guided Spatial-Frequency Attention (WSFA)** and **Relative Angular Geometry Attention (RAGA)**.

* **Performance:** +0.54 dB PSNR improvement over HLFRN on the STFLytro dataset.
* **Parameters:** 0.73M (Lightweight).

---

## ⚙️ Environment

```bash
pip install torch torchvision tensorboard matplotlib scikit-image imageio scipy einops
