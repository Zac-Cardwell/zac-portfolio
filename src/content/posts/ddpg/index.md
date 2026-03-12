---
title: DDPG
date: 2026-03-09
description: ""
tags:
  - RL
image: "[[attachments/mountains.png]]"
imageAlt: ""
imageOG: false
hideCoverImage: false
hideTOC: false
targetKeyword: ""
draft: false
---
**DDPG** is an **actor-critic**, **model-free**, **off-policy** reinforcement learning algorithm.  
It combines ideas from **Deterministic Policy Gradients** and **Deep Q-Networks (DQN)**, and is particularly suited for **continuous action spaces**.

## Core Components
| Component             | Description                                                               |
| --------------------- | ------------------------------------------------------------------------- |
| **Actor Network**     | Approximates deterministic policy ($\mu(s)$)                              |
| **Critic Network**    | Approximates Q-function ( $Q(s, a)$)                                      |
| **Target Networks**   | Soft-updated copies of actor and critic used for stable training          |
| **Replay Buffer**     | Stores past transitions ( $(s, a, r, s')$ $) for off-policy learning      |
| **Exploration Noise** | Adds noise (e.g., Ornstein-Uhlenbeck) to the actor output for exploration |

## Mathematical Foundations
### **Objective**
Maximize the expected return:
$$
\large J(\theta^\mu) = \mathbb{E}_{s_t \sim \rho^\mu}[ Q(s_t, \mu(s_t|\theta^\mu)) ]
$$
### **Critic Loss**
Minimize the Temporal Difference (TD) error:
$$
\large L = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( Q(s,a|\theta^Q) - y \right)^2 \right]
$$
where
$$
\large y = r + \gamma Q'(s', \mu'(s'|\theta^{\mu'})|\theta^{Q'})
$$
### **Actor Gradient**
Update the policy using the chain rule:
$$
\large \nabla_{\theta^\mu} J \approx \mathbb{E}_{s_t \sim D} \left[ \nabla_a Q(s,a|\theta^Q)|_{a=\mu(s)} \nabla_{\theta^\mu} \mu(s|\theta^\mu) \right]
$$
### **Target Network Update**
Soft update with parameter  $\tau \in (0,1)$
$$
\large \theta' \leftarrow \tau \theta + (1 - \tau) \theta'
$$

## Algorithm Steps

1. **Initialize**:
    - Actor $\mu(s|\theta^\mu)$, Critic $Q(s,a|\theta^Q)$
    - Target networks $\mu', Q'$
    - Replay buffer D
        
2. **For each episode**:
    - Receive initial state $s_0s$
    - For each step t:
        1. Select action $\large a_t = \mu(s_t|\theta^\mu) + \mathcal{N}_t$
        2. Execute $a_t$, observe $r_t, s_{t+1}$
        3. Store transition $(s_t, a_t, r_t, s_{t+1}$ in D
        4. Sample random minibatch from D
        5. Compute target $y_i$
        6. Update critic by minimizing L
        7. Update actor using policy gradient
        8. Soft update target networks

---

## Common Hyperparameters
| Parameter            | Symbol       | Typical Range                 |
| -------------------- | ------------ | ----------------------------- |
| Discount factor      | ( $\gamma$ ) | 0.95 – 0.99                   |
| Soft update rate     | ( $\tau$ )   | 0.001 – 0.01                  |
| Replay buffer size   | —            | ( 10^5 – 10^6 )               |
| Batch size           | —            | 64 – 256                      |
| Actor learning rate  | —            | 1e-4                          |
| Critic learning rate | —            | 1e-3                          |
| Noise type           | —            | Ornstein-Uhlenbeck / Gaussian |
