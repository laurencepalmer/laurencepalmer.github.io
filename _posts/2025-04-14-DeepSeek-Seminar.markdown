---
layout: post
title:  "MCL Seminar on DeepSeek-V3 and R1"
date:   2025-04-14 01:08:51 +0000
categories: jekyll update
---

Recently, I delivered an MCL lab seminar focused on tracing the development of the DeepSeek-V3 and R1 models released earlier this year. There was quite a bit of material to unpack given the amount of advancements DeepSeek has made in the past few years across model architecture, infrastructure, and pre/post training phases. 

The seminar traces each development from their inceptions in previous DeepSeek models, like DeepSeek-Math, and explains the fundamental motivation behind these advancements. For example, their advancement in the post-training phase with group relative policy optimization (GRPO) was to remove the critic/value model in the traditional proximal policy optimization (PPO) algorithm, effectively reducing both the cost and complexity of RL-fine tuning by orders of magnitude. Some highlights from the seminar include: 

- Architecture 
    - Multi-head Latent Attention (MLA) + Decoupled Rotary Positional Encoding (RoPE)
    - Fine-grained Mixture of Experts (aka DeepSeekMoE)
    - Multi-token Prediction (MTP)
- Infrastructure 
    - FP8 Mixed Precision
    - HAI-LLM (High-flier adapted insfrastructure)
- Pre-training
    - Long Context Extension with YaRN
- Post-Training 
    - RL with GRPO 

The slides are here: 
<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vT1sMUEgK48lPBegl9HJB0Kd18OvCQTjTKP38-XNjHHFEFSHY32AUm_h27koS2D-w/pubembed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>