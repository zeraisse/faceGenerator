# Face Creation using GAN

## Overview

This project is made in the IPSSI student context to learn how GAN and image generation work. 
The main objective is to learn to create a GAN model from scratch.

---

## Project Structure

### Classes Explanation

| Class | File | Description |
|-------|------|-------------|
| **CelebDataset** | `CelebDataset.py` | The class that transforms and handles my dataset |
| **Generator** | `Generator.py` | The class that will generate images |
| **Discriminator** | `Discriminator.py` | The class that is trained to be able to determine if the image is a fake one made by the generator or a real image |

---

## Getting Started

### Prerequisites

- Python >= 3.11
- UV package manager

### Installation

```bash
# Install dependencies
uv add torch torchvision pillow
```

---

## How GAN Works

**GAN (Generative Adversarial Network)** consists of two antagonist neural networks:

1. **Generator** - Creates fake images
2. **Discriminator** - Distinguishes between real and fake images

These two networks compete against each other, improving their performance over time.

