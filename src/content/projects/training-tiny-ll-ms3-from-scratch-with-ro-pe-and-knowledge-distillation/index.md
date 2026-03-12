---
title: Training Tiny LLMs from Scratch with RoPE and Knowledge Distillation
description: ""
date: 2026-03-09
categories:
  - Generative
  - LLM
  - Transformer
repositoryUrl: https://github.com/Zac-Cardwell/Training-Tiny-LLMs-from-Scratch-with-RoPE-and-Knowledge-Distillation
projectUrl: ""
status: ""
image: "[[attachments/galaxy.png]]"
imageAlt: ""
hideCoverImage: false
hideTOC: false
draft: false
featured: true
---
## Introduction 

Large language models have achieved remarkable capabilities, but their size makes them impractical for many applications. This project explores a fundamental question: **how much intelligence can we compress into a truly small model?**

Rather than simply scaling down existing architectures, I investigated whether compact Transformers (~15M and ~35M parameters) could learn complex behaviors through **knowledge distillation** from a stronger teacher (DistilGPT-2, 82M parameters). The goal wasn't to match state-of-the-art performance, but to understand the learning dynamics of tiny models across increasingly challenging tasks.

### Key Questions

1. **Capacity limits**: Can models with <5% of GPT-2's parameters learn coherent language modeling?
2. **Scaling effects**: How does performance change between 15M and 35M parameters?
3. **Task complexity**: Where do small models succeed, and where do they break down?
4. **Multi-task learning**: Does training on diverse tasks help or hurt individual task performance?

### Approach
I designed a **five-tier evaluation framework** that progresses from basic language modeling to complex multi-task reasoning:

- **Tier 0**: Raw next-token prediction (foundation)
- **Tier 1**: In-domain generalization (Shakespeare, Wikipedia)
- **Tier 2**: Algorithmic tasks (copy, count, reverse)
- **Tier 3**: Symbolic reasoning (addition, logic)
- **Tier 4**: Multi-task instruction following

Each tier builds on the previous, allowing me to isolate where model capacity becomes the bottleneck.

---

## Background: Transformer Architecture

### Core Components

A **Transformer** processes sequences through stacked layers of **self-attention** and **feedforward networks**. Unlike RNNs, which process tokens sequentially, Transformers compute relationships between all tokens in parallel, making them highly efficient to train.

Each **decoder block** (the building block of GPT-style models) contains:
1. **Masked Multi-Head Self-Attention**
    - Allows each token to attend to all previous tokens
    - Prevents information leakage from future positions (autoregressive property)
    - Multiple "heads" learn different types of relationships in parallel
2. **Feedforward Network**
    - Two linear transformations with nonlinear activation (GELU)
    - Applied identically to each position
    - Expands then compresses the representation
3. **Residual Connections & Layer Normalization**
    - Stabilize training and enable deeper networks
    - Applied around each sublayer

**Example decoder block implementation:**

```python
b, seq_len, _ = x.shape
if mask is None:
    mask = self.generate_causal_mask(seq_len, x.device)

# Pre-LN Self-Attention
x_norm = self.layernorm1(x)
attn_out = self.self_attention(x_norm, mask)
x = x + self.dropout1(attn_out)

# Pre-LN Feedforward
ffn_norm = self.layernorm2(x)
ffn_out = self.linear2(self.dropout3(self.activation(self.linear1(ffn_norm))))
x = x + self.dropout2(ffn_out)

return x
```

### Attention

**Attention** determines how much focus each token should give to others in a sequence.  
Given a **query** (what we’re looking for), attention compares it to **keys** (potential matches) to compute relevance scores, which are then used to weight the corresponding **values** (information sources).

**Mathematical formulation:**
$$
\large \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$
Implementation example:

```python
attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
attn_weights = F.softmax(attn_scores, dim=-1)
attn_weights = self.dropout(attn_weights)
out = torch.matmul(attn_weights, v)
```

**Multi-Head Attention** extends this concept by allowing multiple attention heads to operate in parallel. Each head learns to focus on different types of relationships (e.g., syntactic or semantic), and their outputs are concatenated and linearly projected back into the model dimension.

```python 
# x: (batch, seq_len, embed_dim)
qkv = self.qkv(x)
q, k, v = qkv.chunk(3, dim=-1)

# Reshape to (b, num_heads, seq_len, head_dim)
q = q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
k = k.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
v = v.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
```


### Positional Encoding

Since the Transformer lacks any inherent notion of sequence order, **positional encodings** are added to input embeddings to provide information about token positions.
These can be:
- **Fixed sinusoidal functions** (as used in the original paper), or
- **Learned positional embeddings**, which adapt during training.

Modern architectures often use rotary or bias-based encodings (e.g., **RoPE**, **ALiBi**) for improved scalability and generalization to longer sequences.

**Mathematical formulation (sinusoidal positional encoding):**

$$\large PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$\large PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Unlike sinusoidal or learned embeddings, which inject fixed positional vectors into token embeddings, **RoPE encodes relative distances directly within the attention mechanism**, making it inherently position-aware and more robust to long sequences.

**Rotary Positional Embedding (RoPE)** **rotates** the query and key vectors in attention space by a position-dependent angle. This means that **positional information is encoded directly in the attention mechanism** itself, rather than being added beforehand.

Given query q and key k vectors (each of size d) for a token position p:
1. Split q and k into pairs: $\large (q_1, q_2), (q_3, q_4), …$
2. Apply a 2D rotation for each pair based on the position p:

$$\large \text{RoPE}(q, p) = [q_1 \cos \theta_p - q_2 \sin \theta_p, q_1 \sin \theta_p + q_2 \cos \theta_p, \dots]$$
where $\large \theta_p$​ is a frequency term depending on the dimension. This rotation makes the dot-product $\large q_i^\text{rot} \cdot k_j^\text{rot}$ depend only on the **relative distance** (i−j), allowing RoPE to generalize to unseen sequence lengths.

example of **RoPe implementation**:

```python
def apply_rope(q, k, cos, sin):
	# q,k:[batch, seq_len, dim]
	cos = cos.unsqueeze(0)
	sin = sin.unsqueeze(0)
	
	q1, q2 = q[..., ::2], q[..., 1::2] # split even/odd dims → rotation pairs
	k1, k2 = k[..., ::2], k[..., 1::2]
	
	q_rot = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
	k_rot = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
	
	q_out = q_rot.flatten(-2)
	k_out = k_rot.flatten(-2)
	return q_out, k_out
```

## Knowledge Distillation 

In order to make this project more interesting, I've decided to train my model using  knowledge distillation. **Knowledge Distillation (KD)** is a technique where a **smaller model (student)** learns not only from the ground truth labels but also from the **soft predictions of a larger pretrained model (teacher)**.

- Teacher: large pretrained model (e.g., GPT-2 small/medium)
- Student: Smaller untrained model
- Goal: student mimics teacher’s behavior while being smaller/faster

Instead of just learning the one-hot ground truth token:
$$
\large \mathcal{L}_{\text{CE}} = -\sum_i y_i \log p_i​
$$
You also encourage the student to match the teacher’s **soft probability distribution** $\large q_i^{\text{teacher}}$​:
$$
\large \mathcal{L}_{\text{KD}} = \text{KL} \big( q^{\text{teacher}} \parallel p^{\text{student}} \big)
$$
- Often combined with temperature scaling:
$$
\large q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}​
$$
- `T` (temperature) softens probabilities so the student learns **relative token similarities**.
- Total loss:
$$
\large \mathcal{L} = \alpha \cdot \mathcal{L}_{\text{CE}} + (1-\alpha) \cdot \mathcal{L}_{\text{KD}}
$$
Implemented this can look like:

```python
teacher_logits = teacher(input_ids).logits.detach()  # no gradient
student_logits = student(input_ids)

loss_ce = F.cross_entropy(student_logits.view(-1, vocab_size), target_ids.view(-1))
loss_kd = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),
    F.softmax(teacher_logits / T, dim=-1),
    reduction='batchmean'
)
loss = alpha * loss_ce + (1 - alpha) * loss_kd
```

### Top-K Teacher Probabilities
To make knowledge distillation practical for large-vocabulary models like GPT-2, storing the full softmax for each token is prohibitively expensive. Instead, we store only the **top-k most probable tokens** from the teacher model at each output position. This drastically reduces memory and storage requirements while retaining the most informative predictions for training the student.

In my implementation, the teacher’s top-k predictions are saved as two tensors per example:
- `teacher_indices`: token IDs of the top-k predictions
- `teacher_values`: corresponding probabilities

During dataset creation, input and output tokens are combined into a single sequence, and the top-k probabilities are aligned with the output portion. Zeros are prepended for input tokens to maintain proper alignment with masked labels. In the data loader, the collator dynamically pads both the sequences and the top-k tensors per batch:
```python
if "teacher_indices" in batch[0]:
	# Determine max sequence length and max top_k for this batch
	max_seq_len = max(b["teacher_indices"].shape[0] for b in batch)
	max_top_k = max(b["teacher_indices"].shape[1] for b in batch)
	
	# Prepare padded tensors
	batch_size = len(batch)
	teacher_indices_padded = torch.zeros((batch_size, max_seq_len, max_top_k), dtype=torch.long)
	teacher_values_padded = torch.zeros((batch_size, max_seq_len, max_top_k), dtype=torch.float)
	
	for i, b in enumerate(batch):
		seq_len, top_k = b["teacher_indices"].shape
		teacher_indices_padded[i, :seq_len, :top_k] = b["teacher_indices"]
		teacher_values_padded[i, :seq_len, :top_k] = b["teacher_values"]
	
	out["teacher_indices"] = teacher_indices_padded
	out["teacher_values"] = teacher_values_padded
```

This design allows flexible experimentation with different `top_k` values and efficient computation of distillation losses without excessive memory overhead.
## Design and Implementation 

This section outlines the architecture and training setup for the Transformer student model and the teacher used during distillation.

The **student model** follows a **decoder-only Transformer** architecture composed of stacked decoder blocks, each containing **multi-head self-attention** and **feedforward layers** with residual connections and layer normalization.  Positional information is encoded using **Rotary Positional Embeddings (RoPE)** applied directly to the **query** and **key** projections.  
Attention scores are **scaled** and **masked** for autoregressive generation, and **dropout** is applied for regularization.

Each feedforward sublayer expands the hidden dimension, applies a **GELU activation**, and projects back to the embedding size.  This architecture provides efficient context modeling while maintaining simplicity and modularity for experimentation.

The **teacher model** is **DistilGPT-2**, a compact, distilled version of GPT-2 that retains most of GPT-2’s performance while significantly reducing model size and computational cost.  
DistilGPT-2 preserves the autoregressive transformer architecture and key capabilities such as **coherent text generation** and **contextual understanding**, making it ideal for efficient training and deployment.  Its reduced depth (roughly half the number of layers) enables faster inference and lower memory usage while maintaining alignment with the student’s training objective.

## Model Architecture

### Student Models

I trained two student variants to quantify scaling effects:

|Model|Parameters|Embed Dim|Heads|Layers|FFN Dim|
|---|---|---|---|---|---|
|**Medium**|35M|512|4|3|2048|
|**Small**|15M|256|4|3|1024|

Both models share the same **decoder-only architecture**:

- **3 transformer blocks** (compared to DistilGPT-2's 6)
- **RoPE positional encoding** for better length generalization
- **Pre-LayerNorm** configuration for training stability
- **GPT-2 tokenizer** (50,257 vocabulary) for compatibility

The Medium model has ~2.3× the parameters of Small, primarily due to the larger embedding dimension (512 vs 256) and feedforward expansion (2048 vs 1024).

### Teacher Model

**DistilGPT-2** (82M parameters) serves as the teacher:

- Distilled version of GPT-2 (124M) with 40% fewer parameters
- Retains ~97% of GPT-2's performance on language modeling
- Fast inference makes it practical for dataset generation

### Training Configuration

All models were trained on a **single NVIDIA RTX 3070 Ti** with:

---
## Experiments and results 

### Set-Up

To systematically assess model capabilities, I developed a **diagnostic suite** spanning language modeling, memory, compositionality, and reasoning.

Tasks are organized into **five tiers (Tier 0–4)**, ranging from simple to complex. Performance is evaluated using:
- **Hard outputs:** cross-entropy or exact match against ground truth.
- **Soft outputs:** teacher probabilities from DistilGPT-2 for **knowledge distillation on Tier 0–1 only**.
    
**Two student models** (~35M and ~15M parameters) are trained and compared to:
- Quantify how model size impacts retention of teacher knowledge.
- Examine generalization and reasoning ability at different scales.

This setup enables evaluation of learning efficiency, generalization, and instruction-following ability across tasks of increasing complexity, as well as a comparison of how parameter count affects performance.

---
#### Tier 0: Foundational LM Pretraining
**Goal:** Build the base language modeling capability that underpins all higher-tier tasks.

**Description:**
- Model is trained on **general text** using a next-token prediction (causal LM) objective.
- Data combines **Tiny Shakespeare** (poetic / literary style) and **WikiText-2** (factual, diverse content) to expose the model to varied syntax, vocabulary, and token distributions.
- Input is minimal / dummy (`""`) with a `"Task: next_token_pred\nInput:\nOutput:"` prefix to maintain consistency with higher tiers.
- Loss is computed **only on the output tokens**.

**Benefits:**
1. **Stable embeddings and attention heads:** Model learns token relationships, punctuation, and local dependencies.
2. **Cross-domain robustness:** Exposure to both literary and encyclopedic text improves transfer to downstream instruction and reasoning tasks.
3. **Faster convergence in higher tiers:** Pretrained LM weights provide a solid foundation for instruction-tuned or symbolic tasks.

**Metrics:**
- Perplexity
- Cross-entropy / training loss
- Optional token-level accuracy (sanity check)

---

#### Tier 1: Basic LM / Sanity Checks

**Goal:** Verify that embeddings, attention, and optimization mechanisms work correctly in an **instruction-tuned format**.

| Task                  | Example                | Metrics                      |
| --------------------- | ---------------------- | ---------------------------- |
| **Tiny Shakespeare**  | “To be or not to be …” | Token-level accuracy, loss   |
| **Wikitext-2 subset** | Small real text        | Loss convergence, perplexity |
**Notes:**
- Data is formatted with `Task/Input/Output` prefixes.
- Input tokens are masked during loss computation so the model learns to generate the **Output** continuation.
- Confirms the model can adapt pretrained LM knowledge to structured, prompt-style data.

---
#### Tier 2: Memory & Positional Reasoning

**Goal:** Test whether the model can **retain and manipulate sequences**.

|Task|Example|Metrics|
|---|---|---|
|**Copy task**|Input: `a b c d` → Output: `a b c d`|Exact sequence match|
|**Reverse task**|Input: `a b c d` → Output: `d c b a`|Exact sequence match|
|**Counting task**|Input: `count a a a a` → Output: `4`|Accuracy, token-level match|
**Notes:**
- Tier 2 uses **actual sequences generated from Tier 0** rather than simplified or synthetic inputs.
- This tier tests local memory, positional understanding, and sequence manipulation, essential for reasoning in higher tiers.
- Knowledge distillation is **not applied**, as DistilGPT-2 cannot reliably perform these tasks.

---

#### Tier 3: Compositional / Symbolic Reasoning

**Goal:** Evaluate multi-step processing, rule-based reasoning, and extrapolation.

|Task|Example|Metrics|
|---|---|---|
|**Addition (digit-level)**|Input: `12 + 7 =` → Output: `19`|Accuracy, exact match|
|**Arithmetic extrapolation**|Train on 0–99, test on 100–199|Generalization|
|**Relational logic**|Input: `If A>B and B>C, is A>C?` → Output: `Yes`|Exact match|
|**Pattern continuation**|Input: `ABABAB → ?` → Output: `AB`|Exact match|

**Notes:**
- Tests the model’s ability to **compose learned primitives** into multi-step reasoning.
- Helps distinguish memorization from genuine rule-following.

---
#### Tier 4: Multi-task / Instruction Following

The final tier integrates multiple task formats into a unified instruction-tuned setting, testing whether the model can interpret and adapt to task directives dynamically.

**Goal:** Test contextual task conditioning and meta-learning.
- **Setup:** Combine several of the above tasks into one dataset with clear prefixes:
```text
Task: reverse
Input: a b c
Output: c b a

Task: count
Input: count a a a a
Output: 4

Task: add
Input: 12 + 7 =
Output: 19
```
**Metrics:**
- Per-task accuracy
- Exact match
- Ability to **switch behavior** based on the “Task:” prefix
- Optional zero-shot evaluation: introduce new tasks at test time

**Notes:**
- Tests whether the model can **generalize instruction-following** from a mixture of tasks.
- Validates the combination of Tier 0 LM knowledge, Tier 1 text continuity, and Tier 2–3 reasoning skills.

---
#### Dataset and Loading Pipeline
- Each example combines a task prompt and output into a single token sequence. Input tokens are masked in labels for proper loss computation.
- KD from DistilGPT-2 is applied only on Tier 0–1 tasks, with top-k probabilities stored efficiently to reduce memory.
- A dynamic collator pads sequences and top-k vectors per batch, supporting variable lengths and KD configurations.
- This flexible setup enables experimentation across causal LM training, KD, and multi-tier reasoning tasks.
---

## Results

### Tier 0: Foundation - Next Token Prediction

Both models successfully learned from the teacher's distribution, though final perplexities remained high due to limited capacity.

|Model|Val Loss|Perplexity|Token Acc.|KD Loss|
|---|---|---|---|---|
|**Medium (35M)**|**3.59**|**548.8**|**19.9%**|0.87|
|Small (15M)|3.85|915.2|16.7%|0.89|

**Key findings:**

- **Knowledge distillation worked:** KD loss dropped below 0.9, indicating students successfully tracked the teacher's softened probability distributions
- **Size matters for language modeling:** Medium achieved 40% lower perplexity (548 vs 915)
- **High perplexity is expected:** With <20% of the teacher's parameters, perfect language modeling is impossible
- **Zero exact match:** Neither model memorized full sequences, as intended

**Interpretation:** Despite mediocre absolute performance, Tier 0 established stable embeddings and attention patterns that proved crucial for downstream tasks. The KD signal provided meaningful guidance even when students couldn't match teacher performance.

---

### Tier 1: In-Domain Generalization

Performance improved dramatically on held-out sequences from the training distribution.

#### Shakespeare

|Model|Perplexity|Token Acc.|Exact Match|
|---|---|---|---|
|**Medium**|**1.94**|**97.5%**|**22.5%**|
|Small|2.18|96.9%|15.5%|

#### WikiText-2

|Model|Perplexity|Token Acc.|Exact Match|
|---|---|---|---|
|**Medium**|**2.13**|**96.0%**|**19.8%**|
|Small|2.69|94.5%|13.3%|

**Key findings:**

- **Massive perplexity improvement:** From 500+ (Tier 0) to ~2 (Tier 1)
- **Consistent size advantage:** Medium outperformed Small by ~10% perplexity, ~1% token accuracy, and ~50% exact match
- **Strong in-distribution generalization:** 96–97% token accuracy shows both models learned the underlying patterns
- **Modest exact match:** 15–22% suggests shallow context tracking without long-horizon coherence

**Interpretation:** Even tiny LLMs can achieve near-perfect token-level predictions when the distribution is familiar. The gap between token accuracy (97%) and exact match (22%) reveals that errors compound over longer sequences, a fundamental challenge for small models.

---

### Tier 2: Algorithmic Tasks

This tier exposed the first major capacity differences between models.

#### Copy

|Model|Token Acc.|Exact Match|
|---|---|---|
|Medium|99.5%|75.6%|
|Small|99.5%|74.5%|

**Both models solved identity mapping nearly perfectly.** This task is essentially a memory test, no transformation required.

#### Count

|Model|Token Acc.|Exact Match|
|---|---|---|
|Medium|**100%**|**100%**|
|Small|**100%**|**100%**|

**Perfect performance from both models.** Counting short sequences is trivial once the model learns basic number concepts.

#### Reverse

|Model|Token Acc.|Exact Match|
|---|---|---|
|**Medium**|**93.7%**|**22.5%**|
|Small|88.4%|10.8%|

**The first task where capacity clearly matters.** Reversing requires:

1. Encoding the full input sequence
2. Maintaining positional information
3. Generating output in reverse order

The Small model's exact match (10.8%) is less than half the Medium model's (22.5%), suggesting it struggles to maintain long-range dependencies during sequence manipulation.

**Interpretation:** Local operations (copy, count) are within reach of even 15M-parameter models. Global transformations (reverse) expose capacity limits and reveal that parameter count matters for complex positional reasoning.

---

### Tier 3: Symbolic Reasoning

Surprisingly, both models mastered symbolic reasoning tasks.

#### Addition

|Model|Token Acc.|Exact Match|
|---|---|---|
|Medium|99.9%|99.7%|
|Small|**99.9%**|**99.9%**|

**Near-perfect performance from both models.** Addition benefits from:

- Strong local structure (digit-by-digit computation)
- Deterministic rules (carry operations)
- Limited output space (single-digit results + carries)

#### Logic

|Model|Token Acc.|Exact Match|
|---|---|---|
|Medium|**100%**|**100%**|
|Small|**100%**|**100%**|

**Both models achieved perfect logical reasoning.** Convergence was rapid (6–9 epochs), indicating that transitive relations and logical operators are easily learnable with clean supervision.

**Most surprising result:** The 15M-parameter model matched or exceeded the 35M model on deterministic reasoning tasks. This suggests:

- Symbolic reasoning requires **precision, not capacity**
- Small models can learn perfect mappings when the rule space is finite
- Memorization of logical patterns is efficient

**Interpretation:** When tasks have deterministic, compositional structure, even tiny Transformers can achieve human-level accuracy. This stands in stark contrast to Tier 2's reverse task, where stochastic sequence manipulation proved much harder.

---

### Tier 4: Multi-Task Instruction Following

The final tier tested whether models could maintain multiple skills simultaneously and switch between them based on task prefixes.

#### Shakespeare + WikiText-2 (Language Only)

|Model|Perplexity|Token Acc.|Exact Match|Epochs|Time (min)|
|---|---|---|---|---|---|
|Medium|**1.36**|96.4%|21.1%|**29**|68.8|
|**Small**|**1.32**|**96.7%**|**24.5%**|81|**54.5**|

**Only tier where Small matched Medium.** Both models achieved excellent perplexity (~1.3), significantly better than single-domain Tier 1 results.

**Key observations:**

- Multi-task exposure to both literary and encyclopedic styles **improved** generalization
- Small model's unexpected advantage may reflect better regularization at its capacity limit
- Medium converged 2.8× faster (29 vs 81 epochs), suggesting more efficient optimization

---

#### Copy + Reverse (Algorithmic Only)

|Model|Perplexity|Token Acc.|Exact Match|Epochs|Time (min)|
|---|---|---|---|---|---|
|**Medium**|**1.14**|**98.5%**|**51.4%**|91|192.1|
|Small|1.26|97.1%|31.3%|99|115.6|

**Multi-task learning dramatically improved reverse performance:**

- Medium: 22.5% (isolated) → **51.4%** (with copy) - **2.3× improvement**
- Small: 10.8% (isolated) → **31.3%** (with copy) - **2.9× improvement**

**Interpretation:** Training on copy (identity mapping) appears to **regularize** reverse learning. The model may learn better positional representations when exposed to both forward and backward transformations simultaneously, similar to how bidirectional context helps in BERT.

---

#### Shakespeare + Wiki + Copy + Reverse (Mixed)

|Model|Perplexity|Token Acc.|Exact Match|Epochs|Time (min)|
|---|---|---|---|---|---|
|**Medium**|**1.18**|**98.2%**|**39.7%**|62|353.3|
|Small|1.20|97.9%|34.7%|100|212.6|

**No catastrophic interference.** Combining fundamentally different task types (language generation vs. sequence manipulation) did not harm either capability:

- Language perplexity remained near-optimal (~1.2)
- Algorithmic exact match fell between pure algorithmic (51%) and pure language (21%) tasks
- Medium model maintained 5% advantage in exact match

**Interpretation:** Small Transformers can maintain heterogeneous skills simultaneously. The slight degradation in exact match compared to algorithmic-only training suggests some competition for representational capacity, but the effect is modest.

---

#### Count + Add + Logic (Pure Reasoning)

|Model|Perplexity|Token Acc.|Exact Match|Epochs|Time (min)|
|---|---|---|---|---|---|
|Medium|1.0005|99.99%|99.97%|81|62.2|
|**Small**|**1.0001**|**100%**|**100%**|90|**40.5**|

**Both models achieved near-perfect performance.** The Small model actually reached perfect accuracy (100% token + exact match), confirming that deterministic reasoning tasks are trivial for even 15M-parameter Transformers.

**Rapid convergence** (40–62 minutes) shows these patterns are efficiently learnable; the model doesn't need extensive capacity to memorize arithmetic and logical rules.

---

#### Count + Add + Copy + Reverse (Reasoning + Algorithmic)

|Model|Perplexity|Token Acc.|Exact Match|Epochs|Time (min)|
|---|---|---|---|---|---|
|**Medium**|**1.14**|**98.6%**|**75.5%**|69|353.9|
|Small|1.16|98.3%|69.7%|99|211.4|

**Strongest multi-task result.** Both models achieved >75% exact match on heterogeneous reasoning and algorithmic tasks combined.

**Reverse performance improved dramatically:**

- Medium: 22.5% (isolated) → 51.4% (with copy) → **75.5%** (with all tasks)
- Small: 10.8% (isolated) → 31.3% (with copy) → **69.7%** (with all tasks)

**Interpretation:** Exposure to related structured tasks (counting, addition, copying) creates a **curriculum effect** that helps the model learn better positional and sequential reasoning. This is the most compelling evidence that multi-task learning improves difficult tasks rather than merely avoiding interference.

---

### Cross-Tier Analysis

#### Multi-Task Learning Effects

The most striking finding: **multi-task training improved performance on difficult tasks**, especially reverse:

|Training Regime|Medium EM|Small EM|
|---|---|---|
|Reverse alone|22.5%|10.8%|
|+ Copy|51.4%|31.3%|
|+ All reasoning/algorithmic|75.5%|69.7%|

This suggests algorithmic tasks benefit from curriculum-like exposure to related patterns, contradicting the common assumption that multi-task learning primarily causes interference in small models.

---

#### Task Interference Analysis

Mixing language modeling with algorithmic tasks caused **minimal interference**—in fact, multi-task training often improved perplexity:

|Task Combination|Medium PPL|Small PPL|
|---|---|---|
|Shakespeare (isolated)|1.94|2.18|
|Shakespeare + Wiki|1.36|1.32|
|Shakespeare + Wiki + Copy + Reverse|1.18|1.20|

Possible explanations:

- Multi-task learning provides **implicit regularization**
- Diverse task structures prevent overfitting to any single distribution
- Shared low-level features (positional encoding, attention patterns) transfer across tasks

---

#### Model Size Scaling

**Medium model advantages** were most pronounced in:

1. **Raw language modeling** (Tier 0): 40% perplexity improvement
2. **Complex algorithmic tasks** (reverse): 2× exact match improvement
3. **Mixed multi-task scenarios**: 5–6% exact match advantage
4. **Convergence speed**: 2–3× fewer epochs in many settings

**Small model performed surprisingly well on:**

1. **Pure reasoning tasks** (Tier 3): Perfect or near-perfect accuracy
2. **Homogeneous tasks** (language-only Tier 4): Matched or exceeded Medium
3. **Training efficiency**: 1.5–2× faster training time

**Key insight:** Parameter count matters most for tasks requiring **capacity** (language modeling, long-range dependencies) but less for tasks requiring **precision** (arithmetic, logic). The 15M model proves that instruction-following and multi-task learning are achievable at remarkably small scales.

---

#### Training Efficiency

Training time scaled with task heterogeneity and model size:

|Task Type|Medium Time|Small Time|
|---|---|---|
|Pure reasoning|60–70 min|40–55 min|
|Pure language|55–70 min|55–70 min|
|Mixed tasks|190–350 min|110–210 min|

The Medium model required ~1.5–2× more time but consistently delivered stronger exact match performance. For resource-constrained applications, the Small model offers an attractive trade-off: 70% of Medium's performance at 40% of the parameter count and 50% of the training time.

---

## Summary

This project demonstrates that **tiny Transformers can learn sophisticated behaviors** through knowledge distillation and multi-task training:

1. **Knowledge distillation accelerates learning:** Even when students can't match teacher perplexity, soft targets provide valuable learning signal for downstream tasks
2. **Multi-task learning helps, not hurts:** Training on diverse tasks improved performance on difficult operations like reverse through beneficial transfer and implicit curriculum effects
3. **Size matters—but less than expected:** The 15M-parameter model matched 35M on reasoning tasks and came within 5–10% on complex multi-task scenarios, suggesting diminishing returns at small scales
4. **Precision vs. capacity:** Deterministic tasks (arithmetic, logic) require minimal parameters, while open-ended generation (language modeling, long sequences) scales with model size

The tier-based evaluation framework proved effective for isolating model capabilities and understanding where small models succeed or fail. Future work could explore:

- Longer sequence lengths (current: 256 tokens)
- More sophisticated KD techniques (progressive distillation, attention transfer)
- Curriculum learning with gradually increasing task complexity
- Deployment on edge devices to measure real-world efficiency gains