---
title: "paper2code"
excerpt: "Minimal PyTorch implementations of ML papers from scratch"
github: https://github.com/jwei302/paper2code
tags:
  - deep-learning
  - pytorch
  - transformers
  - computer-vision
---

## Overview

**paper2code** is a collection of minimal, educational PyTorch implementations of foundational machine learning papers. The goal is to provide clean, well-documented code that helps others understand the core concepts behind influential models.

## Implementations

### ResNet-18
- Full implementation of the residual network architecture from "Deep Residual Learning for Image Recognition" (He et al., 2015)
- Includes skip connections, batch normalization, and modular residual blocks
- Trained on CIFAR-10 with data augmentation

### GPT-2
- Transformer-based language model following "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- Implements multi-head self-attention, positional encoding, and autoregressive generation
- Trained on a subset of OpenWebText for text generation

## Tech Stack

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Training visualization

## Key Features

- **Minimal dependencies**: Only essential libraries (PyTorch, NumPy)
- **Educational focus**: Extensive comments explaining architecture choices
- **Modular design**: Reusable components (attention layers, residual blocks)
- **Training scripts**: End-to-end training loops with logging and checkpointing

## Learning Outcomes

This project deepened my understanding of:
- Transformer architectures and attention mechanisms
- Residual connections and their role in deep networks
- Training dynamics and optimization techniques
- PyTorch internals and best practices

[View on GitHub]({{ page.github }})
