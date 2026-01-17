---
title: "Filtered Behavior Cloning for VLAs"
excerpt: "Self-training method for vision-language-action models achieving 90% on LIBERO-90"
github: https://github.com/jwei302/fbc
tags:
  - robotics
  - imitation-learning
  - self-training
  - vision-language-models
---

## Overview

**Filtered Behavior Cloning (FBC)** is a self-training approach for vision-language-action (VLA) models that improves robotic manipulation performance by iteratively filtering and learning from high-quality rollouts. We achieve **90% success rate** on the challenging LIBERO-90 benchmark, demonstrating the effectiveness of combining behavior cloning with online data filtering.

## Key Idea

Traditional behavior cloning learns directly from expert demonstrations, but struggles to generalize to diverse tasks. FBC addresses this by:

1. **Collecting diverse rollouts**: Generate trajectories using a base policy trained on demonstration data
2. **Filtering for quality**: Keep only successful trajectories that achieve the task goal
3. **Self-training**: Retrain the policy on filtered data, iteratively improving performance

This creates a virtuous cycle where the policy improves by learning from its own best attempts.

## Results

- **90% success rate on LIBERO-90**: State-of-the-art performance on a diverse set of 90 robotic manipulation tasks
- **Transfer learning**: Successfully transfers knowledge across different manipulation skills
- **Data efficiency**: Achieves strong performance with limited expert demonstrations

## Technical Approach

### Architecture
- **Vision encoder**: Pre-trained CLIP for processing RGB observations
- **Language encoder**: Pre-trained T5 for encoding task instructions
- **Policy network**: Transformer-based action decoder with cross-attention to vision and language

### Training Pipeline
1. Pre-train on expert demonstrations using behavior cloning
2. Collect rollouts in simulation (LIBERO benchmark)
3. Filter rollouts based on task success
4. Retrain policy on combined expert + filtered data
5. Repeat steps 2-4 for multiple iterations

### Key Innovations
- **Automatic filtering criteria**: Uses task-specific success metrics (object positions, states)
- **Balanced sampling**: Maintains distribution over task types during self-training
- **Stability techniques**: Gradient clipping and learning rate scheduling prevent catastrophic forgetting

## Tech Stack

- **PyTorch**: Deep learning framework
- **LIBERO**: Benchmark suite for lifelong robot learning
- **MuJoCo**: Physics simulation
- **Transformers (Hugging Face)**: Pre-trained vision and language models

## Impact

This work demonstrates that self-training can significantly boost VLA performance without requiring additional human demonstrations. The filtering approach is simple yet effective, making it practical for real-world robotics applications.

## Future Work

- Extending to real-world robot hardware
- Exploring multi-task self-training strategies
- Submitting results to ICML 2026

[View on GitHub]({{ page.github }})
