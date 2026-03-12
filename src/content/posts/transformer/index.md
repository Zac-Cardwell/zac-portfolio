---
title: Transformer
date: 2026-03-09
description: ""
tags:
  - LLMs
  - Transformer
  - Generative
image: "[[attachments/transformer.png]]"
imageAlt: ""
imageOG: false
hideCoverImage: false
hideTOC: true
targetKeyword: ""
draft: false
---
The **Transformer** is a deep learning architecture introduced in _"Attention is All You Need"_ (Vaswani et al., 2017).  
It replaced recurrence (RNNs, LSTMs) and convolutions with **self-attention**, enabling models to process entire sequences in parallel and capture **long-range dependencies** efficiently.

Transformers form the backbone of most modern language models, from **BERT** and **T5** (understanding tasks) to **GPT** and **LLaMA** (text generation).  
Their success stems from scalability, parallelization, and the ability to **model contextual relationships** between all tokens in a sequence.

At a high level:
- **Encoder blocks** learn contextual representations of inputs.
- **Decoder blocks** use attention over those representations (and prior outputs) to generate predictions.

---

## Learning Objectives 

During training, the **goal of a Transformer** is to learn to represent and predict structured sequences , such as text, by modeling dependencies between tokens.

### **Core Learning Objectives**
1. **Learn contextual token representations**  
    Each token embedding $\large h_i$ should encode meaning conditioned on the entire sequence context.
$$
\large h_i = f(t_i, t_{1..n})
$$
2. **Model conditional probability distributions**  
    The model learns $\large P(t_i | t_{<i})$ (for autoregressive decoders) or $\large P(t_i | t_{\neq i})$ (for bidirectional encoders).  
    The objective is to **minimize the negative log-likelihood**:
    $$
    \large\mathcal{L}_{NLL} = - \sum_{t=1}^{T} \log P(t_t | t_{<t}; \theta)
    $$
3. **Learn alignment between inputs and outputs** (in encoder–decoder models)  
    Through cross-attention, the decoder learns to attend to relevant input positions when generating outputs.
4. **Learn efficient attention patterns**  
    Attention weights $\large A_{ij}$ encode relationships like syntax, semantics, or coreference, allowing the model to dynamically focus on the most relevant context.

---

### **Common Training Objectives**

|Objective|Used In|Description|
|---|---|---|
|**Autoregressive LM (Next Token Prediction)**|GPT, LLaMA|Predict the next token given all previous ones|
|**Masked LM (Denoising / MLM)**|BERT, RoBERTa|Predict masked tokens from bidirectional context|
|**Sequence-to-Sequence (Translation, Summarization)**|T5, BART|Generate target sequence conditioned on input|
|**Prefix LM**|T5.1.1, UL2|Combines masked + autoregressive objectives|
|**Contrastive / Distillation Losses**|DistilBERT, MiniLM|Match logits or hidden states of a teacher model|

---

### **Loss Functions**

#### 1. **Cross-Entropy Loss (standard)**
$$
\large \mathcal{L}_{CE} = -\sum_{t} \log p_\theta(y_t | y_{<t})
$$
#### 2. **Masked LM Loss**
Only compute loss for masked positions:
$$
\large \mathcal{L}_{MLM} = -\sum_{i \in M} \log p_\theta(y_i | y_{\setminus M})
$$
#### 3. **Label Smoothing**
Improves generalization by preventing overconfidence:
$$
\large \mathcal{L}_{smooth} = (1 - \epsilon)\mathcal{L}_{CE} + \epsilon H(u, p_\theta)
$$
where $u$ is the uniform distribution and $\large \epsilon \approx 0.1$

#### 4. **Distillation Loss**
Matches teacher and student distributions (see section on Knowledge Distillation).

---

## Architecture Summary

Transformers are built from **repeating blocks** composed of:
1. **Multi-Head Self-Attention (MHSA)**
    - Each token attends to every other token in the input sequence.
    - This allows the model to capture long-range dependencies and contextual relationships (e.g., subject–verb agreement).

2.  **Feed-Forward Network (FFN)**
    - A fully connected network applied independently to each token’s representation.
    - Typically composed of two linear transformations with a nonlinear activation (such as ReLU or GELU) in between.

3. **Residual Connections + Layer Normalization**
	- applied around each sublayer to improve gradient flow and stabilize training. 

|Type|Components|Example Models|Description|
|---|---|---|---|
|**Encoder-only**|Self-Attention|BERT|Learns bidirectional context for understanding tasks|
|**Decoder-only**|Causal Self-Attention|GPT|Generates text autoregressively|
|**Encoder–Decoder**|Encoder + Cross-Attention|T5, BART|Generates outputs conditioned on input context|

---

## Encoder / Decoder

### **Encoder**
- Input: sequence of embeddings (with positional encodings)
- Self-attention layers attend to **all tokens**
- Outputs: contextual representations for all positions
### **Decoder**
- Input: previously generated tokens
- Layers:
    - **Causal Self-Attention** (can’t see future tokens)
    - **Cross-Attention** (to encoder output)
    - **Feed-forward network**
- Output: next-token logits for generation

---

## Masking

|Mask Type|Purpose|Where Used|
|---|---|---|
|**Padding Mask**|Ignore `<PAD>` tokens in batches|Encoder/Decoder|
|**Causal Mask**|Prevent looking ahead|Decoder-only models|
|**Training Mask**|Mask random tokens for prediction|MLM pretraining|
### **Causal Mask Matrix**
$$
\large M_{ij} = \begin{cases} 0, & j \leq i \\ -\infty, & j > i \end{cases}​
$$
Applied to attention logits before softmax.

---

## Shifted Inputs / Outputs

During training, sequence-to-sequence models shift the decoder input:
- **Input:** `<BOS> token₁ token₂ … tokenₙ₋₁`
- **Target:** `token₁ token₂ … tokenₙ <EOS>`
Ensures the model predicts each token given all previous tokens.

---

## Positional Encoding

Since attention is permutation-invariant, position info must be added.

### **1. Sinusoidal Encoding**
$$
\large PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$
### **2. Rotary Positional Embedding (RoPE)**
**rotates** the query and key vectors in attention space by a position-dependent angle. This means that **positional information is encoded directly in the attention mechanism** itself, rather than being added beforehand.

Given query q and key k vectors (each of size d) for a token position p:
1. Split q and k into pairs: $\large (q_1, q_2), (q_3, q_4), …$
2. Apply a 2D rotation for each pair based on the position p:

$$\large \text{RoPE}(q, p) = [q_1 \cos \theta_p - q_2 \sin \theta_p, q_1 \sin \theta_p + q_2 \cos \theta_p, \dots]$$
where $\large \theta_p$​ is a frequency term depending on the dimension. This rotation makes the dot-product $\large q_i^\text{rot} \cdot k_j^\text{rot}$ depend only on the **relative distance** (i−j), allowing RoPE to generalize to unseen sequence lengths.
### **3. ALiBi (Attention with Linear Biases)**
Adds position-dependent bias directly to attention logits:
$$
\large \text{Attn}(Q,K) = \frac{QK^T}{\sqrt{d_k}} + m \cdot \text{bias}​
$$
---

## Attention Mechanism

### **Scaled Dot-Product Attention**
$$
\large \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V​
$$
- Q: queries (current token)
- K: keys (all tokens)
- V: values (token contents)
### **Multi-Head Attention**
$$
\large \text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O
$$
Each head learns to attend to different relational aspects.

---

## Knowledge Distillation

Train a smaller **student** model to mimic a larger **teacher** model.
$$
\large \mathcal{L}_{KD} = (1 - \alpha)\mathcal{L}_{CE} + \alpha T^2 \text{KL}(p_t^T || p_s^T)
$$
- T: temperature (softens distribution)
- $\large p_t^T, p_s^T$​: teacher/student probabilities at temperature T

**Top-k Distillation:**  
Use only the teacher’s top-k logits to reduce computation.
-  Best practice: k=10-50; useful in NLP to reduce noise in vocabulary.

---

## Text Generation

Transformers can generate text **autoregressively**:
$$
\large P(x) = \prod_t P(x_t | x_{<t})
$$
### **Autoregressive** (GPT-style)
- One token at a time
- Causal attention mask
### **Non-Autoregressive** (NAT, BERT-style)
- Predict all tokens in parallel
- Requires iterative refinement

---

##  Decoding Strategies

|Method|Description|Pros|Cons|
|---|---|---|---|
|**Greedy**|Pick max-prob token each step|Fast|Repetitive|
|**Beam Search**|Keep top-k sequences by log-prob|High quality|Expensive|
|**Top-k Sampling**|Sample from top-k tokens|Diverse|Needs tuning|
|**Top-p (Nucleus)**|Sample from smallest set with cumulative prob ≥ p|Adaptive diversity|More complex|
|**Temperature Scaling**|Adjust randomness via T|Simple|Needs tuning|

---

## Best Practices

|Category|Tip|
|---|---|
|**Masking**|Align masks carefully for batch processing|
|**Positional Encodings**|Use RoPE or ALiBi for long-context models|
|**Regularization**|Label smoothing improves stability|
|**Training**|Use gradient clipping + mixed precision|
|**Generation**|Tune (temperature, top-p) jointly for fluency vs diversity|
|**Distillation**|Match both logits and hidden representations
