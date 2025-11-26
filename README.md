# Awesome-Visual-GRPO-RLHF-Alignment

---

## 目录

- [教程与综述](#教程与综述)
- [各子方向早期代表性工作](#各子方向早期代表性工作)
- [1. RL / Policy Gradient / GRPO 基础](#1-rl--policy-gradient--grpo-基础)
- [2. 扩散--flow-模型上的-rlhf--grpo](#2-扩散--flow-模型上的-rlhf--grpo)
- [3. Reward Hacking 与帕累托--kl-约束](#3-reward-hacking-与帕累托--kl-约束)
- [4. 视觉-reward-model--benchmark](#4-视觉-reward-model--benchmark)


---

## 教程与综述

- **Spinning Up in Deep RL** – OpenAI 的经典强化学习入门教程，涵盖 policy gradient、TRPO、PPO 等基础。
  - [[网站]](https://spinningup.openai.com/)
- **CS285: Deep Reinforcement Learning @ Berkeley**
  - [[课程主页]](https://rail.eecs.berkeley.edu/deeprlcourse/)

---

## 各子方向早期代表性工作

### 传统 Policy Gradient / KL 正则 RL

- **2015 — Trust Region Policy Optimization (TRPO)**  
  提出基于 KL 约束的 trust-region policy gradient，为后续 PPO / KL-正则 RL 奠定基础。  
  [[paper]](https://arxiv.org/abs/1502.05477)

- **2017 — Proximal Policy Optimization Algorithms (PPO)**  
  通过 clip surrogate 近似 trust-region，把 TRPO 变成大模型可用的工程标准。  
  [[paper]](https://arxiv.org/abs/1707.06347)
  
### 流形视角 RL

### Policy Optimization

- **2024.02 — DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**  
  首次系统提出 **Group Relative Policy Optimization (GRPO)**，用 group 内相对得分替代 critic，大幅简化 RLHF 训练。  
  [[paper]](https://arxiv.org/abs/2402.03300) [[code]](https://github.com/deepseek-ai/DeepSeek-Math)

- **2025.01 — DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via GRPO**  
  大规模展示 GRPO 在推理模型上的威力，把 GRPO 变成 RLHF 新主流。  
  [[paper]](https://arxiv.org/abs/2501.12948)

### 扩散模型上的人类偏好对齐
（RLHF、ReFL、Sft）

- **2023.05 — Reinforcement Learning for Fine-Tuning Text-to-Image Diffusion Models (DPOK)**  
  较早直接在 text-to-image diffusion 上做 RLHF，引入 KL 约束控制 policy 偏离基线。  
  [[paper]](https://arxiv.org/abs/2305.16381)

- **2023.05 — Training Diffusion Models with Reinforcement Learning (DDPO)**  
  将 denoising 过程视为多步 MDP，提出 DDPO，用 policy gradient 直接优化下游 reward。  
  [[paper]](https://arxiv.org/abs/2305.13301) [[project]](http://rl-diffusion.github.io/) [[code]](https://github.com/jannerm/ddpo)

- imageReFL、DraftK、AlignProp、Sft相关的工作

### Flow / Rectified Flow + GRPO 代表

- **2025.05 — Flow-GRPO: Training Flow Matching Models via Online RL**  
  在 Flow Matching 上引入 GRPO，把 rectified flow 训练为在线 RL。  
  [[code / paper]](https://github.com/yifan123/flow_grpo)

- **2025.05 — DanceGRPO: Unleashing GRPO on Visual Generation**  
  把 GRPO 扩展到 diffusion & rectified flow & text-to-video / image-to-video，多 reward 统一框架。  
  [[paper]](https://arxiv.org/abs/2505.07818) [[code]](https://github.com/XueZeyue/DanceGRPO)

### 多目标 / 帕累托 Alignment

- **2023.06 — Rewarded Soups: Towards Pareto-Optimal Alignment by Interpolating Weights Fine-Tuned on Diverse Rewards**  
  多 reward 分别 fine-tune，再做 weight interpolation，显式追求帕累托最优。  
  [[paper]](https://arxiv.org/abs/2306.04488) [[code]](https://github.com/alexrame/rewardedsoups)

- **2024.02 — Panacea: Pareto Alignment via Preference Adaptation for LLMs**  
  将对齐视为多维偏好优化问题，理论上能恢复整个帕累托前沿。  
  [[paper]](https://arxiv.org/abs/2402.02030)

### 视觉 Reward Model & Benchmark

- **2023.03 — Human Preference Score (HPS): Better Aligning Text-to-Image Models with Human Preference**  
  训练人偏好分类器并提出 HPS，用于微调 Stable Diffusion。  
  [[paper]](https://arxiv.org/abs/2303.14420) [[code]](https://github.com/tgxs002/align_sd)

- **2023.04 — ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation**  
  首个通用 t2i 人偏好 Reward Model，支撑 ReFL 等方法。  
  [[paper]](https://arxiv.org/abs/2304.05977) [[code]](https://github.com/THUDM/ImageReward)

---

## 1. RL / Policy Gradient / GRPO 


- **2015 — Trust Region Policy Optimization (TRPO)**  
  - 核心：用 KL 约束保证 update 在 trust region 内，提升稳定性。  
  [[paper]](https://arxiv.org/abs/1502.05477)

- **2017 — Proximal Policy Optimization Algorithms (PPO)**  
  - 核心：用 clip surrogate 近似 TRPO 的 constrained update，变成简单易用的 actor-critic 算法。  
  [[paper]](https://arxiv.org/abs/1707.06347)

- **2024.02 — DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**  
  - 核心：提出 **GRPO**，不再依赖 critic，用 group 内均值作为 baseline，极大节省大模型 RLHF 的资源。  
  [[paper]](https://arxiv.org/abs/2402.03300) [[code]](https://github.com/deepseek-ai/DeepSeek-Math)

- **2025.01 — DeepSeek-R1: Incentivizing Reasoning Capability in LLMs**  
  - 核心：用 GRPO scaled-up 训练纯 RL 的推理模型（R1-Zero），展示多阶段 RL pipeline。  
  [[paper]](https://arxiv.org/abs/2501.12948)

- **2025.03 — DAPO: Decoupled Clip and Dynamic sAmpling Policy Optimization**  
  - 核心：在 GRPO 之上引入 **非对称 clip（Clip-Higher）+ 动态采样 + 长序列稳定训练**，可视作「GRPO++」。  
  [[paper]](https://arxiv.org/abs/2503.14476) [[code]](https://github.com/BytedTsinghua-SIA/DAPO)


---

## 2. 扩散 / Flow 模型上的 RLHF / GRPO

### 2.1 扩散模型上的 RLHF（DDPO / DPOK / Sparse Reward 等）

- **2023.05 — Reinforcement Learning for Fine-Tuning Text-to-Image Diffusion Models (DPOK)**  
  - 将 RLHF 直接用于 Stable Diffusion，使用 KL 正则和 reward model 进行 fine-tuning，关注 prompt-image alignment。  
  [[paper]](https://arxiv.org/abs/2305.16381)

- **2023.05 — Training Diffusion Models with Reinforcement Learning (DDPO)**  
  - 将 denoising 看成一个多步决策过程，使用 policy gradient 直接优化下游 reward（包括人偏好 / Aesthetics 等）。  
  [[paper]](https://arxiv.org/abs/2305.13301) [[project]](http://rl-diffusion.github.io/) [[code]](https://github.com/jannerm/ddpo)

- **2025.03 — B²-DiffuRL: Towards Better Alignment — Training Diffusion Models with Reinforcement Learning Against Sparse Rewards**  
  - 针对扩散 RL 中 reward 仅在末步出现的 **稀疏奖励问题**，提出 backward progressive training + branch-based sampling，改善 credit assignment。  
  [[paper]](https://arxiv.org/abs/2503.11240)


---

### 2.2 Flow / Rectified Flow 模型上的 GRPO


- **2025.05 — Flow-GRPO: Training Flow Matching Models via Online RL**  
  - Flow Matching, Online RL, GRPO, KL-free baseline  
  - 把 Flow Matching 视作多步 MDP，将 GRPO 用于 text-to-image flow 模型上，给出大规模 RLHF pipeline。  
  [[code / paper]](https://github.com/yifan123/flow_grpo)

- **2025.05 — DanceGRPO: Unleashing GRPO on Visual Generation**  
  - text-to-image / text-to-video / image-to-video、多 Reward 组合  
  - 在 Stable Diffusion, FLUX, HunyuanVideo 等不同 backbone 上统一使用 GRPO，实证证明 RLHF 在视觉生成任务的通用性和可扩展性。  
  [[paper]](https://arxiv.org/abs/2505.07818) [[code]](https://github.com/XueZeyue/DanceGRPO)

- **2025.07 — MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE**  
  - 混合 ODE-SDE 采样、Sliding Window、效率优化  
  - 在 Flow-GRPO 之上，通过在小窗口内采用 SDE + GRPO、窗口外用 ODE 采样，降低优化开销，并提出 MixGRPO-Flash 进一步提速。  
  [[paper]](https://arxiv.org/abs/2507.21802) [[code]](https://github.com/Tencent-Hunyuan/MixGRPO)

- **2025.08 — TempFlow-GRPO: When Timing Matters for GRPO in Flow Models**  
  - Temporal credit assignment、trajectory branching、noise-aware weighting  
  - 指出现有 Flow-GRPO 假设时间步均匀，对关键时间步 credit assignment 不够准确；提出分支采样和时间步加权，使 RL 对「何时更新」更敏感。  
  [[paper]](https://arxiv.org/abs/2508.04324)
- **2025.08 — Pref-GRPO: PREF-GRPO: PAIRWISE PREFERENCE REWARD-BASED GRPO FOR STABLE TEXT-TO-IMAGE REINFORCEMENT LEARNING**  
  - pairwise reward,vlm-as-a-judge
  - 使用vlm对图像偏好对进行人类偏好winrate判断，代替标量奖励，缓解reward hacking 
  - [[paper]](https://arxiv.org/pdf/2508.20751) [[code]](https://github.com/CodeGoat24/Pref-GRPO)

- **2025.10 — GRPO-Guard: Mitigating Implicit Over-Optimization in Flow Matching via Regulated Clipping**  
  - Implicit over-optimization、ratio normalization、gradient reweighting  
  - 通过对 importance ratio 做归一化 + 梯度重加权，修复 Flow-GRPO 中隐含的由于梯度裁切不对称导致的过优化问题。  
  [[paper]](https://arxiv.org/abs/2510.22319)


---

## 3. Reward Hacking 与帕累托 / KL 约束

### 3.1 多目标 / 帕累托 Preference Optimization（model free）

- **2023.06 — Rewarded Soups: Towards Pareto-Optimal Alignment by Interpolating Weights Fine-Tuned on Diverse Rewards**  
  - 思路：针对多个 reward（帮助性、无害性、风格等）分别 fine-tune，再在权重空间做线性插值，达到在多个 reward 上的帕累托最优。  
  [[paper]](https://arxiv.org/abs/2306.04488) [[code]](https://github.com/alexrame/rewardedsoups)

- **2024.02 — Panacea: Pareto Alignment via Preference Adaptation for LLMs**  
  - 思路：把 alignment 看成 **多维偏好优化 (MDPO)**，学习一个可以在在线阶段适配不同偏好权重的模型；在理论上可以恢复整个帕累托前沿。  
  [[paper]](https://arxiv.org/abs/2402.02030)

- **2025.02 — CaPO: Calibrated Multi-Preference Optimization for Aligning Diffusion Models**  
  - 思路：针对多 Reward Model（美学、语义对齐、安全性等），通过「校准 + 帕累托前沿采样」得到 general preference，再用 regression loss 微调扩散模型。  
  - 特别适合作为你们工作的视觉多目标对照组。  
  [[paper]](https://arxiv.org/abs/2502.02588)

- **2025.05 — MOPO: Multi-Objective Preference Optimization — Improving Human Alignment of Generative Models**  
  - 思路：从 constrained KL-regularized optimization 角度出发：  
    - 最大化主目标 reward；  
    - 将其他目标视作 **下界约束**；  
    - 理论上可恢复帕累托前沿，实践上给出简单的 closed-form 更新。  
  [[paper]](https://arxiv.org/abs/2505.10892)

- **2025.06 — AMoPO: Adaptive Multi-Objective Preference Optimization without Reward Models and Reference Models**  
  - 思路：不显式训练 Reward Model / Reference Model，而是把多维评测指标当作隐式 reward，通过自适应权重实现多目标对齐。  
  [[paper]](https://aclanthology.org/2025.findings-acl.462.pdf)

---

## 4. 视觉 Reward Model & Benchmark

### 4.1 Reward Models for Text-to-Image / Visual Generation

- **2023.03 — Human Preference Score (HPS): Better Aligning Text-to-Image Models with Human Preference**  
  - 构造人类偏好数据集，训练 Human Preference Score，用于提升 Stable Diffusion 生成质量与人偏好一致性。  
  [[paper]](https://arxiv.org/abs/2303.14420) [[code]](https://github.com/tgxs002/align_sd)

- **2023.04 — ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation**  
  - 通用 text-to-image Reward Model（ImageReward）构建，可作为训练 reward 和评测指标。  
  [[paper]](https://arxiv.org/abs/2304.05977) [[code]](https://github.com/THUDM/ImageReward)

> TODO：  
> - UnifiedReward / UnifiedReward-Think，多任务统一 Reward
> - 用于视频的 Reward（VideoAlign、VisionReward 等） 


### 4.2 Benchmarks for Text-to-Image / Visual Alignment

- **2023.09 — GenEval: An Object-Focused Framework for Evaluating Text-to-Image Generation**  
  - 用 object detection + color classifier 等判别模型，对 compositional 属性（co-occurrence / count / color / position 等）进行细粒度评测。  
  - 和人类评测高度相关。  
  [[paper]](https://arxiv.org/abs/2310.11513) [[code]](https://github.com/djghosh13/geneval)

- **2025.10 — UniGenBench++: A Unified Semantic Evaluation Benchmark for Text-to-Image Generation**  
  - 内容：600 条多语言 prompt，覆盖 5 大主题、20 个子主题，10 类主维度、27 个子维度的语义一致性评测，并提供 offline evaluator。  
  - 非常适合用来评估语义。  
  [[paper]](https://arxiv.org/abs/2510.18701) [[code]](https://github.com/CodeGoat24/UniGenBench)


