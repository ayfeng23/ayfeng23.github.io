---
title: "Photonic Quantum GAN"
excerpt: "1st place at MIT iQuHack 2024 - Quantum generative model using photonic hardware"
github: https://github.com/jwei302/iquHack
tags:
  - quantum-computing
  - generative-models
  - photonics
  - hackathon
---

## Overview

**Photonic Quantum GAN** won **1st place** in the Photonic Quantum Computing track at MIT iQuHack 2024. We implemented a quantum generative adversarial network (QGAN) using photonic quantum computing hardware, demonstrating that quantum systems can learn to generate classical data distributions.

## The Challenge

MIT iQuHack challenged teams to explore cutting-edge quantum computing platforms. We chose the photonic track, which uses photons (particles of light) as qubits. Photonic quantum computers offer advantages in operating at room temperature and integrating with existing fiber optic infrastructure.

## Our Approach

### Quantum Generator
- Implemented a parameterized quantum circuit (PQC) that generates quantum states
- Used photonic gates (beam splitters, phase shifters) to create entanglement
- Mapped quantum measurement outcomes to classical data samples

### Classical Discriminator
- Traditional neural network that distinguishes real data from generated samples
- Provides feedback to improve the generator through adversarial training

### Training Loop
1. Generator produces samples using photonic quantum circuit
2. Discriminator evaluates samples against real data
3. Update generator parameters to fool the discriminator
4. Repeat until generator learns the target distribution

## Results

- **Successfully learned simple distributions**: Generated samples closely matched target Gaussian distributions
- **Demonstrated quantum advantage potential**: Quantum generator required fewer parameters than classical baseline
- **1st place finish**: Judges recognized our novel application of photonic quantum computing to generative modeling

## Tech Stack

- **Xanadu Strawberry Fields**: Photonic quantum computing framework
- **PennyLane**: Quantum machine learning library
- **PyTorch**: Classical discriminator network
- **NumPy/Matplotlib**: Data analysis and visualization

## Key Challenges

- **Limited qubit count**: Worked with 4-8 photonic modes
- **Noise and decoherence**: Photonic systems are sensitive to loss and imperfect gates
- **Training instability**: Balancing quantum circuit expressivity with trainability

## What We Learned

- Photonic quantum computing offers a promising platform for near-term quantum ML
- Hybrid quantum-classical algorithms can leverage the strengths of both paradigms
- Adversarial training in quantum systems requires careful hyperparameter tuning

## Team

This project was built with an amazing team of quantum computing enthusiasts during a 24-hour hackathon sprint at MIT.

[View on GitHub]({{ page.github }})
