---
title: "Reflections on My First Year in ML Research"
date: 2025-01-12
categories:
  - personal
  - research
tags:
  - machine-learning
  - academia
  - graduate-school
excerpt: "Lessons learned, mistakes made, and insights gained from diving into machine learning research at Yale."
---

## Introduction

A year ago, I walked into my first research meeting at the Yale Applied Physics Lab with a mixture of excitement and impostor syndrome. I had taken a couple ML courses and built some toy projects, but real research felt like uncharted territory. Now, after a year of failed experiments, late-night debugging sessions, and occasional breakthroughs, I wanted to reflect on what I've learned.

## The Messy Reality of Research

### Papers Make It Look Easy

Reading ML papers, you might think research is a linear process: identify problem → propose solution → run experiments → achieve state-of-the-art results. In reality, it's more like:

1. Identify problem
2. Read 50 papers
3. Propose solution
4. Realize someone already tried it (buried in Appendix C of a 2019 NeurIPS paper)
5. Tweak approach
6. Implement baseline
7. Baseline doesn't work
8. Debug for 3 weeks
9. Realize you had a sign error in the loss function
10. Baseline now works *too well*
11. Your method barely beats baseline
12. Question your entire research direction
13. Try 10 more variations
14. One works!
15. Write paper, submit, get rejected
16. Revise, resubmit, (hopefully) accept

Nobody tells you about steps 7-12.

### Experiments Fail. A Lot.

Early on, I thought every idea I had was brilliant. Why wouldn't filtered behavior cloning work? Just filter the good rollouts!

Reality check: my first implementation had 10% success rate after self-training. Worse than the baseline.

What went wrong?
- Filtering was too strict → almost no data
- Didn't balance task distribution → catastrophic forgetting
- Learning rate too high → unstable training

It took weeks of ablation studies and debugging to diagnose these issues. The final version that hit 90% success on LIBERO-90 was the result of dozens of failed attempts.

**Lesson**: Research is about systematic debugging, not just having clever ideas.

## What I Wish I Knew Earlier

### 1. Start Simple, Stay Simple

My first instinct is always to build something fancy. Multi-head attention? Sure. Adversarial training? Why not. Meta-learning? Let's add that too.

This is a trap.

I've learned to:
- **Start with the simplest possible baseline**: Can a linear model solve this?
- **Add complexity incrementally**: Only add components when you can justify them with ablations
- **Keep experiments reproducible**: Version control everything, log all hyperparameters

Simple baselines are often embarrassingly strong. And when they fail, you learn *why* complexity is needed.

### 2. Read Code, Not Just Papers

Papers omit crucial implementation details. The appendix might say "we used Adam with learning rate 3e-4," but not mention:
- Warmup schedule
- Gradient clipping threshold
- Batch size sensitivity
- Random seed importance

Reading others' code taught me more than any paper. Open-source implementations (especially from DeepMind, OpenAI, Meta) are goldmines of engineering wisdom.

### 3. Embrace Negative Results

My biggest lesson: **negative results are valuable**.

When filtered behavior cloning failed initially, my instinct was to abandon it. But by carefully analyzing *why* it failed, I discovered:
- Which task types benefited from self-training (manipulation) vs. didn't (navigation)
- How data quality mattered more than data quantity
- Why success-based filtering beats threshold-based filtering

These insights made the final version much stronger.

## The Joy of Discovery

Despite the frustrations, there are moments that make it all worthwhile.

### When the Curves Finally Go Up

After weeks of failed experiments, seeing the success rate curve climb from 70% to 80% to 90% was euphoric. It meant something I built was actually working. The robot could solve tasks it couldn't before.

### Learning from Failures

When experiments fail in unexpected ways, you learn something new about the problem. One time, my VLA policy learned to *avoid* objects instead of manipulating them. Bizarre, right?

Turns out the vision encoder was overfitting to background features. A valuable lesson about pre-training and fine-tuning.

### Collaborating with Smart People

Some of my best ideas came from casual conversations with labmates. Someone would mention a paper, or notice a weird pattern in my results, and suddenly a new research direction would emerge.

Research is collaborative, even when you're working on individual projects.

## Advice for Aspiring ML Researchers

If I could tell my past self anything, it would be:

1. **Embrace the grind**: Research is 10% eureka moments, 90% debugging. Get comfortable with uncertainty.

2. **Ask for help**: Don't waste days stuck on a bug when a 5-minute conversation could solve it.

3. **Write as you go**: Don't wait until paper deadline to write up results. Document experiments immediately.

4. **Balance breadth and depth**: Read widely to get ideas, but focus deeply on one problem.

5. **Take care of yourself**: Research marathons burn you out. Exercise, sleep, and hobbies keep you sane.

6. **Celebrate small wins**: Every working baseline, every ablation, every insightful failure is progress.

## Looking Ahead

This year I'm working on:
- Extending FBC to real-world robot hardware (sim-to-real transfer)
- Exploring multi-task self-training for VLAs
- Submitting to ICML 2026

I'm also TAing Introduction to Machine Learning, which I'm excited about. Teaching forces you to understand concepts deeply, and I love helping others discover ML.

## Conclusion

My first year in ML research has been humbling, challenging, and incredibly rewarding. I've learned that research is less about brilliance and more about persistence, curiosity, and systematic thinking.

If you're considering research, my advice: dive in. Embrace the failures. Ask questions. Build things. The learning curve is steep, but the view from the top is worth it.

---

Thanks for reading! If you're working on similar problems or just want to chat about ML research, feel free to reach out at jeffrey.wei@yale.edu.
