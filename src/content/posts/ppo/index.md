---
title: PPO
date: 2026-03-09
description: ""
tags:
  - RL
image: "[[attachments/robo.png]]"
imageAlt: ""
imageOG: false
hideCoverImage: false
hideTOC: false
targetKeyword: ""
draft: false
---
**PPO** is an **actor-critic**, **model-free**, **on-policy** reinforcement learning algorithm.  
It is designed to **stably improve policy performance** by limiting the size of policy updates using a **surrogate objective**.  
PPO is particularly popular due to its **simplicity**, **robustness**, and strong performance across both **discrete and continuous action spaces**.

## Core Components
| Component                | Description                                                       |
| ------------------------ | ----------------------------------------------------------------- |
| **Actor Network**        | Outputs a stochastic policy $\pi_\theta(a,s)$                     |
| **Critic Network**       | Estimates state value function $V_\phi(s)$                        |
| **Surrogate Objective**  | Optimizes a clipped policy ratio to prevent large updates         |
| **Advantage Estimation** | Uses generalized advantage estimation (GAE) for lower variance    |
| **Mini-batch Training**  | On-policy updates via multiple epochs over collected trajectories |

## Mathematical Foundations
### **Objective**
PPO uses a clipped objective to avoid large policy updates:
$$
\large L^{CLIP}(\theta) = \mathbb{E}_t \Big[ \min \big( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \big) \Big]
$$
where  
$$
\large r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}
$$  
The **advantage $\large A_t$** is what _weights_ the update, it determines **how strongly** the policy should be nudged for each sampled action.

### Advantage
The **advantage function** is defined as:
$$
\large A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)
$$
where:
- $\large Q^\pi(s_t, a_t)$ is the expected return starting from $\large s_t$, taking action $\large a_t$​, and following policy $\large \pi$.
- $\large V^\pi(s_t)$ is the expected mean starting from $\large s_t$ and following policy $\large \pi$

In practice, $\large A_t$​ is **estimated** from sampled trajectories. A common and effective estimator is the **Generalized Advantage Estimation (GAE)**:
$$
\large \hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l}​
$$
where
$$
\large \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$
Parameters:
- $\large \gamma$: discount factor (e.g., 0.99)
- $\large \lambda$: GAE parameter (e.g., 0.95) that balances bias vs variance
    
GAE smooths out the advantage estimate, making learning more stable.

### **Critic Loss**
Minimize the value function error:
$$
\large L^{VF}(\phi) = \mathbb{E}_t \big[ (V_\phi(s_t) - R_t)^2 \big]
$$
where $R_t$ is the cumulative return.  

### **Entropy Bonus (Optional)**
Encourages exploration by adding entropy regularization:
$$
\large L^{S}(\theta) = \mathbb{E}_t [ \beta \, \mathcal{H}[\pi_\theta](s_t) ]
$$

### **Total PPO Loss**
$$
\large L(\theta, \phi) = L^{CLIP}(\theta) - c_1 L^{VF}(\phi) + c_2 L^{S}(\theta)
$$
where $c_1, c_2$ are coefficients for value loss and entropy bonus.

## Algorithm Steps

1. **Initialize** actor $\pi_\theta$ and critic $V_\phi$.  
2. **For each iteration**:
    1. Collect trajectories using current policy $\pi_\theta$.  
    2. Compute advantages $\hat{A}_t$ (e.g., using GAE).  
    3. For $K$ epochs:
        1. Sample mini-batches from collected trajectories.  
        2. Compute policy ratio $r_t(\theta)$.  
        3. Compute clipped surrogate objective $L^{CLIP}(\theta)$.  
        4. Update actor by maximizing $L^{CLIP}$.  
        5. Update critic by minimizing $L^{VF}$.  
        6. Apply entropy bonus if needed.  

---

## Common Hyperparameters
| Parameter                   | Symbol         | Typical Range              |
| --------------------------- | -------------- | -------------------------- |
| Discount factor             | $\gamma$       | 0.95 – 0.99               |
| GAE parameter               | $\lambda$      | 0.90 – 0.97               |
| Clipping parameter          | $\epsilon$     | 0.1 – 0.3                 |
| Value loss coefficient      | $c_1$          | 0.5 – 1.0                 |
| Entropy coefficient         | $c_2$          | 0.01 – 0.1                |
| Learning rate (actor & critic)| —            | 3e-4 – 1e-3               |
| Mini-batch size             | —              | 64 – 256                  |
| Number of epochs per update | —              | 3 – 10                     |
