---
title: "Self-Training for Vision-Language-Action Models"
date: 2025-01-10
categories:
  - robotics
  - machine-learning
tags:
  - imitation-learning
  - self-training
  - vision-language-models
  - robotics
excerpt: "How filtering your robot's successful attempts can dramatically improve manipulation performance, achieving 90% success on LIBERO-90."
---

## The Challenge

Training robots to perform manipulation tasks is hard. Traditional approaches like reinforcement learning require extensive trial and error, while imitation learning (behavior cloning) needs many expert demonstrations. What if we could get more mileage out of limited demonstrations by having the robot learn from its own successful attempts?

This is the core idea behind my **Filtered Behavior Cloning (FBC)** project, where we achieved 90% success on the challenging LIBERO-90 benchmark.

## Vision-Language-Action Models

Recent work in robotics has shown that **vision-language-action (VLA)** models—which combine visual perception, language understanding, and action prediction—can generalize across diverse manipulation tasks. These models:

- Take as input: RGB images + natural language task descriptions
- Output: Robot actions (joint positions or end-effector poses)

The promise is that by grounding actions in both vision and language, VLAs can generalize better than vision-only policies.

## The Self-Training Approach

### Standard Behavior Cloning

Traditional behavior cloning learns a policy $\pi(a|s, g)$ that predicts actions $a$ given states $s$ and goals $g$ by minimizing:

$$\mathcal{L} = \mathbb{E}_{(s,a,g) \sim \mathcal{D}_{\text{expert}}} \left[ \| \pi(a|s,g) - a_{\text{expert}} \|^2 \right]$$

This works well when expert data is abundant and covers the state distribution the policy will encounter. But robots often drift off-distribution, and collecting more expert data is expensive.

### Filtered Behavior Cloning

FBC extends behavior cloning with self-training:

1. **Train initial policy**: Use behavior cloning on expert demonstrations
2. **Collect rollouts**: Execute policy in environment to gather trajectories
3. **Filter by success**: Keep only trajectories that achieve the task goal
4. **Retrain**: Combine filtered data with expert data and retrain policy
5. **Repeat**: Iterate steps 2-4 for multiple rounds

The key insight: **the policy's own successful rollouts are valid training data**. Over time, the policy improves by learning from its best attempts.

## Implementation Details

### Architecture

Our VLA model consists of:

```python
class VLAPolicy(nn.Module):
    def __init__(self):
        # Vision encoder (frozen CLIP)
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        # Language encoder (frozen T5)
        self.language_encoder = T5EncoderModel.from_pretrained("t5-base")

        # Action decoder (trainable)
        self.action_decoder = TransformerDecoder(
            d_model=512,
            nhead=8,
            num_layers=6
        )

        # Action head
        self.action_head = nn.Linear(512, action_dim)

    def forward(self, images, text, proprioception):
        # Encode vision and language
        vision_features = self.vision_encoder(images).pooler_output
        text_features = self.language_encoder(text).last_hidden_state

        # Concatenate proprioception
        state = torch.cat([vision_features, proprioception], dim=-1)

        # Cross-attention to language, decode action
        action_features = self.action_decoder(
            tgt=state,
            memory=text_features
        )

        return self.action_head(action_features)
```

### Filtering Criteria

We filter rollouts based on task-specific success metrics:

- **Object manipulation**: Check if target object reached goal position (Euclidean distance < threshold)
- **Articulated objects**: Verify joint state (e.g., door opened > 80°)
- **Multi-step tasks**: All subtasks completed in sequence

### Training Loop

```python
def self_training_iteration(policy, env, expert_data):
    # Collect rollouts
    rollouts = []
    for task in tasks:
        for _ in range(num_rollouts_per_task):
            trajectory = collect_rollout(policy, env, task)
            rollouts.append(trajectory)

    # Filter successful rollouts
    successful_rollouts = [r for r in rollouts if is_successful(r)]

    # Combine with expert data
    train_data = expert_data + successful_rollouts

    # Retrain policy
    train(policy, train_data, num_epochs=10)

    return policy
```

## Results

We evaluated on **LIBERO-90**, a benchmark with 90 diverse manipulation tasks:

| Method | Success Rate |
|--------|--------------|
| Behavior Cloning (baseline) | 67% |
| FBC (1 iteration) | 78% |
| FBC (3 iterations) | 85% |
| FBC (5 iterations) | **90%** |

Key observations:
- **Each self-training iteration improves performance**: Success rate increases monotonically
- **Data efficiency**: We used only 10 expert demos per task, far less than typical RL baselines
- **Stable training**: No catastrophic forgetting with proper learning rate scheduling

## Why Does This Work?

FBC succeeds because:

1. **Exploration**: Early rollouts explore variations around expert behavior
2. **Filtering**: Keeping only successful trajectories provides an implicit curriculum
3. **Distribution shift**: As the policy improves, it encounters states it will actually visit, improving robustness

Essentially, we're doing **online imitation learning** but only imitating the policy's successes.

## Challenges

### Sparse Success Early On

Initial policy may have <30% success rate, limiting filtered data. Solutions:
- Use lower thresholds early (partial credit for progress)
- Ensure enough diversity in initial expert data

### Task Distribution

Need to maintain balanced sampling across tasks to avoid forgetting:

```python
# Sample uniformly over tasks
for task in tasks:
    task_data = [r for r in successful_rollouts if r.task == task]
    batch.append(random.sample(task_data, batch_size_per_task))
```

## Future Directions

I'm excited to explore:
- **Real-world transfer**: Testing FBC on physical robot hardware
- **Multi-task self-training**: Can the policy generalize to new tasks with zero-shot transfer?
- **Active learning**: Selecting which tasks to collect more rollouts for

We're preparing a submission to **ICML 2026** with more comprehensive experiments and analysis.

## Conclusion

Self-training is a powerful technique for improving robot learning with limited expert data. By iteratively collecting and filtering the policy's own rollouts, we can achieve strong performance on diverse manipulation tasks.

Check out the code on [GitHub](https://github.com/jwei302/fbc)!

## References

1. Brohan, A., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.
2. Rosete-Beas, C., et al. (2023). LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning.
3. Ross, S., Gordon, G., & Bagnell, D. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.
