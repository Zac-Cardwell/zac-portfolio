---
title: DDPM
date: 2026-03-09
description: ""
tags:
  - Generative
  - CV
image: "[[attachments/ddpm-diffusion-process.png]]"
imageAlt: ""
imageOG: false
hideCoverImage: false
hideTOC: false
targetKeyword: ""
draft: false
---
A **Denoising Diffusion Probabilistic Model (DDPM)** learns to generate data (e.g., images) by _reversing a diffusion process_ that gradually destroys data structures.
- **Forward process (diffusion):** gradually adds Gaussian noise to a real image until it becomes pure noise.
- **Reverse process (denoising):** the model learns to reverse this process, denoising step by step until we get a new sample that looks like real data.

---

## The Forward Diffusion Process

We define a sequence of noisy images:
$$
\large x_0, x_1, \dots, x_T
$$
where:
- $x_0$ is a real data sample (e.g., an image),
- $x_T$ is nearly pure Gaussian noise,
- each step adds a small amount of Gaussian noise.

Formally:
$$
\large q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1 - \beta_t}\, x_{t-1},\ \beta_t I \right)
$$
where:
- $\beta_t \in (0, 1)$ is a small variance term (the **noise schedule**),
- $\sqrt{1 - \beta_t}$ scales the signal,
- and $\beta_t I$ is the added Gaussian noise covariance.

By chaining these distributions:
$$
\large q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})
$$
So $x_t$ becomes progressively noisier as $t$ increases.

Becacuse each step adds Gaussian noise, we can directly sample $x_t$ from $x_0$ without iterating all previous steps:

$$
\large q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar{\alpha}_t}\, x_0,\ (1 - \bar{\alpha}_t) I \right)
$$

where:
$$
\large \alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
$$

Therefore:
$$
\large x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon,
\quad \epsilon \sim \mathcal{N}(0, I)
$$

**This form is key:** it allows us to sample noisy images directly at any time step, showing how the signal and noise components evolve with $t$.

---

## The Reverse (Generative) Process

We want to **reverse the noise process**: sample $\large x_{t-1}$ given $\large x_t$.

We define the reverse conditional:
$$
\large p_\theta(x_{t-1} | x_t) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)\right)
$$

| Symbol                           | Meaning                                                                                                                                   |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| $\large x_{t-1}$                 | the **random variable** whose distribution we’re describing (e.g. the previous latent in a diffusion process)                             |
| $\large \tilde{\mu}_t(x_t, x_0)$ | the **mean** of the Gaussian, which is a function of $x_t$ and $x_0$; it can depend on other variables                                    |
| $\large \tilde{\beta}_t I$       | the **covariance matrix**, which is a scaled identity matrix; all dimensions have the same variance $\tilde{\beta}_t$ and are independent |

The mean $\large \mu_\theta$​ is learned by a neural network (usually a **U-Net**), parameterized by $\large \theta$.
Training aims to make $\large p_\theta$​ approximate the true reverse process:
$$
\large q(x_{t-1} | x_t, x_0)
$$
which has a closed form (because everything’s Gaussian):
$$
\large q(x_{t-1} | x_t, x_0) = \mathcal{N}\left(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I \right)
$$
with:
$$
\large \tilde{\mu}_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)​
$$
---

## Training Objective

The DDPM objective is derived by **variational inference**, we minimize a variational bound on the negative log-likelihood:
$$
\large L_{\text{vlb}} = \mathbb{E}_q \left[ -\log p_\theta(x_{0:T}) + \log q(x_{1:T} | x_0) \right]
$$
which simplifies (Ho et al., 2020) to a **mean-squared error (MSE) loss** between the model’s predicted noise and the true noise added in the forward process:
$$
\large L_{\text{simple}} = \mathbb{E}_{x_0, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \|^2 \right]
$$
Here:
- $\large \epsilon$ is the true Gaussian noise used to generate $\large x_t$​,
- $\large \epsilon_\theta(\cdot)$ is the model’s predicted noise.
So, the model learns to **predict the noise** added at each step.

---

## Sampling (Generation)

To generate data:
1. Start from Gaussian noise: $\large x_T \sim \mathcal{N}(0, I)$
2. For $\large t = T, T-1, \dots, 1$:
    - Predict noise: $\large \hat{\epsilon}_\theta = \epsilon_\theta(x_t, t)$
    - Compute mean:
        $$
       \large \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\epsilon}_\theta \right)
        $$
	-  Sample:
        $$
     \large x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$
After all steps, $\large x_0$​ is a generated sample.