---
title: "Implementing ResNet-18 from Scratch in PyTorch"
date: 2025-01-15
categories:
  - machine-learning
  - computer-vision
tags:
  - deep-learning
  - pytorch
  - resnet
  - implementation
excerpt: "A deep dive into implementing ResNet-18 from scratch, understanding residual connections and why they matter for training deep networks."
---

## Introduction

When I first learned about ResNet (Residual Networks), I was fascinated by how such a simple idea—skip connections—could enable training networks with hundreds of layers. But reading the paper and truly understanding the implementation are two different things. This post walks through my experience implementing ResNet-18 from scratch in PyTorch as part of my [paper2code](https://github.com/jwei302/paper2code) project.

## The Problem: Vanishing Gradients

Before ResNet, training very deep networks was notoriously difficult. As gradients backpropagate through many layers, they can vanish (approach zero) or explode, making learning nearly impossible. This is the infamous **vanishing gradient problem**.

Surprisingly, adding more layers to a network often made it *worse*, not better. Deeper networks would have higher training error than shallower ones, even though theoretically they should be able to at least match the shallower network's performance by learning identity mappings in the extra layers.

## The Solution: Residual Connections

ResNet's key insight is the **residual connection** (or skip connection). Instead of learning a mapping $H(x)$ directly, we learn the residual:

$$F(x) = H(x) - x$$

The output becomes:

$$H(x) = F(x) + x$$

This simple change makes it much easier for the network to learn identity mappings. If the optimal function is close to identity, the network just needs to drive $F(x)$ to zero, which is easier than learning $H(x) = x$ from scratch.

## Implementation

### Basic Residual Block

Here's the core residual block:

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # First conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second conv layer
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Main path
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Skip connection
        out += self.shortcut(x)
        out = torch.relu(out)

        return out
```

### Key Design Choices

1. **Batch Normalization**: Applied after each convolution to normalize activations
2. **No bias in conv layers**: Bias is redundant when followed by batch norm
3. **Projection shortcut**: When dimensions change (stride ≠ 1 or channel mismatch), use 1x1 conv to match dimensions

### Full ResNet-18 Architecture

ResNet-18 stacks these basic blocks:

```python
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # First block may downsample
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv + pooling
        x = self.maxpool(torch.relu(self.bn1(self.conv1(x))))

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling + classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
```

## Training on CIFAR-10

I trained ResNet-18 on CIFAR-10 with standard data augmentation:

- Random crop (32x32 with padding=4)
- Random horizontal flip
- Normalization

Training details:
- **Optimizer**: SGD with momentum (0.9)
- **Learning rate**: 0.1 with cosine decay
- **Batch size**: 128
- **Epochs**: 200

Results: **~93% test accuracy** after 200 epochs, comparable to published results.

## What I Learned

1. **Skip connections are crucial**: Without them, training deep networks is extremely difficult
2. **Batch normalization matters**: Stabilizes training and allows higher learning rates
3. **Details matter**: Small choices (initialization, learning rate schedule) significantly impact final performance
4. **PyTorch is powerful**: The framework makes it easy to implement complex architectures cleanly

## Next Steps

I'm planning to:
- Implement ResNet-50 with bottleneck blocks
- Experiment with other skip connection designs (DenseNet, ResNeXt)
- Try pre-training on ImageNet

Check out the full implementation on [GitHub](https://github.com/jwei302/paper2code)!

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in deep residual networks. ECCV.
