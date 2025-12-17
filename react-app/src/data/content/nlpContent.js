// Comprehensive NLP Content for Study Modal
// Each topic has DEEP explanations with intuition, derivations, and the "why" behind everything

export const nlpContent = {

    "Word2Vec Embeddings": `
# Word2Vec: Learning Word Representations

## 1. The Fundamental Problem

How do you represent a word as a number? The naive approach—one-hot encoding—fails because:

1. **No similarity**: "cat" and "dog" are as different as "cat" and "quantum"
2. **Huge vectors**: 100,000 words = 100,000-dimensional vectors
3. **No generalization**: Learning about "cat" tells you nothing about "kitten"

**Word2Vec's insight**: A word is defined by the company it keeps.

Words appearing in similar contexts should have similar representations.

---

## 2. The Distributional Hypothesis

> "You shall know a word by the company it keeps." — J.R. Firth (1957)

This is the foundation of all word embeddings. Consider:

- "The **cat** sat on the mat"
- "The **dog** sat on the mat"

Since "cat" and "dog" appear in identical contexts, they should have similar representations.

---

## 3. Skip-gram: Predicting Context from Words

### The Objective

Given a center word, predict the surrounding context words.

Input: "cat" → Output: ["the", "sat", "on"]

The objective function:

$$J = -\\frac{1}{T}\\sum_{t=1}^T \\sum_{-c \\leq j \\leq c, j \\neq 0} \\log P(w_{t+j} | w_t)$$

where $c$ is the context window size.

### The Probability Model

$$P(w_o | w_c) = \\frac{\\exp(\\mathbf{u}_o^T \\mathbf{v}_c)}{\\sum_{w \\in V} \\exp(\\mathbf{u}_w^T \\mathbf{v}_c)}$$

Notice: We have **two embedding matrices**:
- $\\mathbf{v}_c$: Embedding when word is the center
- $\\mathbf{u}_o$: Embedding when word is context

**Why two matrices?** Using the same matrix creates a degenerate solution where all embeddings become identical.

---

## 4. The Computational Problem

The denominator sums over the **entire vocabulary**:

$$\\sum_{w \\in V} \\exp(\\mathbf{u}_w^T \\mathbf{v}_c)$$

With $|V| = 100,000$ words, this is computed for EVERY training example. Far too slow!

### Solution: Negative Sampling

Replace the expensive softmax with binary classification:

Instead of asking "Which of 100,000 words is correct?", ask:
- "Is (cat, sat) a real pair?" → Yes
- "Is (cat, quantum) a real pair?" → No

The new objective:

$$J = \\log \\sigma(\\mathbf{u}_o^T \\mathbf{v}_c) + \\sum_{k=1}^K \\mathbb{E}_{w_k \\sim P_n}[\\log \\sigma(-\\mathbf{u}_k^T \\mathbf{v}_c)]$$

where:
- First term: Push positive pairs together
- Second term: Push negative pairs apart
- $K$: Number of negative samples (typically 5-20)
- $P_n$: Noise distribution (usually $P(w)^{0.75}$ to upweight rare words)

---

## 5. Why the 3/4 Power?

The noise distribution is:

$$P_n(w) \\propto f(w)^{0.75}$$

**Intuition**: Pure frequency would sample "the" and "a" constantly. The 0.75 power slightly flattens the distribution, giving rare words more representation in negative samples.

---

## 6. Implementation from Scratch

\`\`\`python
import numpy as np

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim):
        # Two embedding matrices
        self.W_center = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_context = np.random.randn(vocab_size, embedding_dim) * 0.01
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def train_step(self, center_word, context_word, neg_samples, lr=0.025):
        """
        center_word: index of center word
        context_word: index of positive context word
        neg_samples: list of indices for negative samples
        """
        v_c = self.W_center[center_word]  # Center embedding
        
        # Positive example: push together
        u_o = self.W_context[context_word]
        score = self.sigmoid(np.dot(u_o, v_c))
        
        # Gradient: (sigmoid - 1) for positive
        grad_pos = (score - 1)
        self.W_context[context_word] -= lr * grad_pos * v_c
        grad_center = grad_pos * u_o
        
        # Negative examples: push apart
        for neg in neg_samples:
            u_neg = self.W_context[neg]
            score_neg = self.sigmoid(np.dot(u_neg, v_c))
            
            # Gradient: sigmoid for negative
            self.W_context[neg] -= lr * score_neg * v_c
            grad_center += score_neg * u_neg
        
        self.W_center[center_word] -= lr * grad_center
        
    def get_embeddings(self):
        # Often average both matrices
        return (self.W_center + self.W_context) / 2
\`\`\`

---

## 7. The Magic of Word Arithmetic

Famous result:

$$\\text{king} - \\text{man} + \\text{woman} \\approx \\text{queen}$$

**Why does this work?**

The model learns that certain dimensions encode semantic concepts like "royalty" and "gender." The vector $king - man$ captures "royale male → female," and adding it to "woman" gives "royal female."

---

## 8. Skip-gram vs CBOW

| Aspect | Skip-gram | CBOW |
|--------|-----------|------|
| Task | Predict context from word | Predict word from context |
| Speed | Slower (multiple predictions) | Faster (one prediction) |
| Rare words | Better (each gets its own training) | Worse (averaged in context) |
| Best for | Small datasets | Large datasets |

---

## 9. Key Takeaways

1. **Words are defined by context**—the distributional hypothesis
2. **Skip-gram predicts context from words**; CBOW does the reverse
3. **Negative sampling makes training tractable** by replacing softmax
4. **Two embedding matrices** prevent degenerate solutions
5. **The 0.75 power** balances frequent and rare words in negative sampling
`,

    "Transformer Attention Code": `
# The Transformer: Attention Is All You Need

## 1. Why Transformers?

RNNs process sequences **sequentially**—token by token. This creates two problems:

1. **No parallelization**: Can't use GPUs effectively
2. **Long-range dependencies**: Information must flow through many steps, degrading

The Transformer's key insight: **Replace recurrence with attention.**

Every token can directly attend to every other token in one step.

---

## 2. The Attention Mechanism: Core Intuition

Imagine you're reading: "The animal didn't cross the street because **it** was too tired."

What does "it" refer to? You need to look back at "animal." This is attention: dynamically looking at relevant parts of the input.

### Query-Key-Value Framework

Think of it like a search engine:
- **Query (Q)**: What you're looking for ("I'm at position 8, what should I attend to?")
- **Key (K)**: What each position offers ("I'm 'animal' at position 2")
- **Value (V)**: The actual content to retrieve

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

---

## 3. Why Scale by $\\sqrt{d_k}$?

This is subtle but crucial.

When $d_k$ is large, the dot products $q^T k$ can become very large. This pushes softmax into regions where:
- One element dominates (~1)
- All others are ~0
- Gradients become tiny (vanishing gradient)

Scaling by $\\sqrt{d_k}$ keeps the variance of dot products around 1, regardless of dimension.

**Mathematical justification**: If $q$ and $k$ have entries with mean 0 and variance 1:
$$Var(q^T k) = \\sum_{i=1}^{d_k} Var(q_i k_i) = d_k$$

So we divide by $\\sqrt{d_k}$ to normalize variance back to 1.

---

## 4. Multi-Head Attention

One attention head might focus on syntax, another on coreference, another on topic.

Instead of one $d_{model}$-dimensional attention, use $h$ heads with dimension $d_k = d_{model}/h$:

$$\\text{MultiHead}(Q, K, V) = \\text{Concat}(head_1, ..., head_h)W^O$$

where $head_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

Each head can learn to attend to different things!

---

## 5. Implementation from Scratch

\`\`\`python
import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Initialize projections
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
    
    def split_heads(self, x, batch_size):
        """(batch, seq, d_model) -> (batch, heads, seq, d_k)"""
        x = x.reshape(batch_size, -1, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # Project queries, keys, values
        Q = Q @ self.W_q
        K = K @ self.W_k
        V = V @ self.W_v
        
        # Split into heads
        Q = self.split_heads(Q, batch_size)  # (batch, heads, seq_q, d_k)
        K = self.split_heads(K, batch_size)  # (batch, heads, seq_k, d_k)
        V = self.split_heads(V, batch_size)  # (batch, heads, seq_v, d_k)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        attention_weights = softmax(scores, axis=-1)
        attention_output = attention_weights @ V
        
        # Concatenate heads
        attention_output = attention_output.transpose(0, 2, 1, 3)
        attention_output = attention_output.reshape(batch_size, -1, self.d_model)
        
        # Final projection
        return attention_output @ self.W_o
\`\`\`

---

## 6. Positional Encoding: Where Am I?

Attention is **permutation invariant**—it doesn't know word order!

We inject position information:

$$PE_{(pos, 2i)} = \\sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \\cos(pos / 10000^{2i/d_{model}})$$

**Why sinusoidal?**

1. **Bounded**: Values stay in [-1, 1]
2. **Unique**: Each position has a distinct pattern
3. **Relative positions**: $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$
4. **Extrapolation**: Can handle sequences longer than training

---

## 7. The Full Encoder Block

\`\`\`python
class TransformerEncoderBlock:
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_output)
        
        return x
\`\`\`

---

## 8. Encoder vs Decoder

| Component | Self-Attention | Cross-Attention | Masking |
|-----------|---------------|-----------------|---------|
| Encoder | Q=K=V=input | None | Padding only |
| Decoder | Q=K=V=output | Q=output, K=V=encoder | Causal + padding |

The decoder adds **causal masking**: position $i$ can only attend to positions $< i$ (no peeking at the future).

---

## 9. Key Takeaways

1. **Attention replaces recurrence**, enabling parallelization
2. **Scaling by $\\sqrt{d_k}$** prevents softmax saturation
3. **Multi-head attention** lets the model attend differently
4. **Positional encoding** provides word order information
5. **Residual connections + LayerNorm** enable deep networks
`,

    "Decoder Architecture": `
# Decoder-Only Transformers & LLM Internals

## 1. The Autoregressive Language Model

GPT, LLaMA, and most modern LLMs are **decoder-only** transformers.

They model:
$$P(x_1, x_2, ..., x_n) = \\prod_{i=1}^n P(x_i | x_1, ..., x_{i-1})$$

Each token is predicted given all previous tokens.

---

## 2. Causal Masking: No Peeking!

The key difference from encoders: **causal (masked) self-attention**.

Position $i$ can only attend to positions $\\leq i$:

\`\`\`python
def causal_mask(seq_len):
    """Create lower-triangular mask for causal attention."""
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask  # 1 = attend, 0 = mask out
\`\`\`

The attention scores become:
$$\\text{Attention}_{ij} = \\begin{cases} \\text{softmax}(\\frac{q_i^T k_j}{\\sqrt{d_k}}) & \\text{if } j \\leq i \\\\ 0 & \\text{otherwise} \\end{cases}$$

---

## 3. KV Cache: The Inference Optimization

### The Problem

During generation, we compute attention for each new token. Without optimization, we recompute keys and values for ALL previous tokens at EVERY step.

For sequence length $n$: $O(n^2)$ total computation.

### The Solution: Cache K and V

The keys and values for positions $1...i-1$ don't change when generating token $i$.

\`\`\`python
class CachedAttention:
    def __init__(self, d_model, n_heads):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.k_cache = None
        self.v_cache = None
    
    def forward(self, x, use_cache=False):
        Q = x @ self.attention.W_q
        K = x @ self.attention.W_k
        V = x @ self.attention.W_v
        
        if use_cache and self.k_cache is not None:
            # Append new K, V to cache
            K = np.concatenate([self.k_cache, K], axis=1)
            V = np.concatenate([self.v_cache, V], axis=1)
        
        if use_cache:
            self.k_cache = K
            self.v_cache = V
        
        # Q is only for new tokens, K/V include history
        return self.attention.compute_attention(Q, K, V)
\`\`\`

Now generation is $O(n)$ instead of $O(n^2)$!

---

## 4. Rotary Position Embedding (RoPE)

### The Problem with Learned/Sinusoidal PE

- **Absolute positions**: The model sees "position 5," not "3 tokens before me"
- **Length extrapolation**: Can't handle sequences longer than training

### RoPE's Insight

Encode **relative** positions in the attention computation itself:

$$\\text{Attention}(q_m, k_n) = f(q, m)^T f(k, n) = g(q, k, m-n)$$

The attention between positions $m$ and $n$ depends only on $m - n$!

### How It Works

Rotate the query and key vectors by angles proportional to position:

$$f(x, m) = \\begin{pmatrix} \\cos(m\\theta) & -\\sin(m\\theta) \\\\ \\sin(m\\theta) & \\cos(m\\theta) \\end{pmatrix} \\begin{pmatrix} x_1 \\\\ x_2 \\end{pmatrix}$$

Applied pairwise to embedding dimensions.

\`\`\`python
def apply_rope(x, positions, dim):
    """Apply rotary position embeddings."""
    # Compute rotation angles
    freqs = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    angles = positions[:, None] * freqs[None, :]
    
    # Split into pairs and rotate
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    cos = np.cos(angles)[..., None]
    sin = np.sin(angles)[..., None]
    
    x_rotated = np.concatenate([
        x_pairs[..., 0:1] * cos - x_pairs[..., 1:2] * sin,
        x_pairs[..., 0:1] * sin + x_pairs[..., 1:2] * cos
    ], axis=-1)
    
    return x_rotated.reshape(x.shape)
\`\`\`

---

## 5. Flash Attention: Memory Efficiency

### The Memory Problem

Naive attention stores the full $N \\times N$ attention matrix in GPU memory.

For $N = 32,768$ and fp16: $32768^2 \\times 2 \\text{ bytes} = 2$ GB per layer!

### The Solution: Block-wise Computation

Flash Attention computes attention in blocks, keeping only what's needed in fast SRAM:

1. Load blocks of Q, K, V
2. Compute partial attention
3. Accumulate results
4. Never materialize full attention matrix

**Result**: $O(N)$ memory instead of $O(N^2)$, plus faster due to better memory access patterns.

---

## 6. The Generation Loop

\`\`\`python
def generate(model, prompt_tokens, max_new_tokens, temperature=1.0):
    tokens = prompt_tokens.copy()
    
    # Initial forward pass (fills KV cache)
    _ = model.forward(tokens, use_cache=True)
    
    for _ in range(max_new_tokens):
        # Forward pass with only the last token
        logits = model.forward(tokens[-1:], use_cache=True)
        
        # Sample next token
        probs = softmax(logits[-1] / temperature)
        next_token = np.random.choice(len(probs), p=probs)
        
        tokens.append(next_token)
        
        if next_token == EOS_TOKEN:
            break
    
    return tokens
\`\`\`

---

## 7. Key Takeaways

1. **Decoder-only models** use causal masking for autoregressive generation
2. **KV caching** reduces generation from $O(n^2)$ to $O(n)$
3. **RoPE** encodes relative positions, enabling length extrapolation
4. **Flash Attention** reduces memory from $O(n^2)$ to $O(n)$
5. **Temperature** controls randomness in sampling
`,

    "RNN & Sequence Models": `
# Recurrent Neural Networks: Learning from Sequences

## 1. The Sequence Problem

Standard neural networks assume **fixed-size inputs**. But language has variable length:
- "Hello" (1 word)
- "The quick brown fox jumps over the lazy dog" (9 words)

RNNs process sequences **one element at a time**, maintaining a **hidden state** that summarizes everything seen so far.

---

## 2. The Vanilla RNN

### The Equations

At each time step $t$:

$$h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

where:
- $x_t$: Input at time $t$
- $h_t$: Hidden state (the "memory")
- $y_t$: Output

### Implementation

\`\`\`python
class VanillaRNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize weights
        self.W_xh = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_hy = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b_h = np.zeros((hidden_dim, 1))
        self.b_y = np.zeros((output_dim, 1))
        
    def forward(self, inputs, h_prev):
        """
        inputs: list of input vectors (seq_len, input_dim)
        h_prev: initial hidden state
        """
        h = h_prev
        outputs = []
        hiddens = [h]
        
        for x in inputs:
            h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)
            y = self.W_hy @ h + self.b_y
            
            hiddens.append(h)
            outputs.append(y)
        
        return outputs, hiddens
\`\`\`

---

## 3. The Vanishing Gradient Problem

### Why RNNs Struggle with Long Sequences

During backpropagation through time (BPTT), gradients flow backward through many time steps:

$$\\frac{\\partial L}{\\partial h_0} = \\frac{\\partial L}{\\partial h_T} \\prod_{t=1}^{T} \\frac{\\partial h_t}{\\partial h_{t-1}}$$

Each $\\frac{\\partial h_t}{\\partial h_{t-1}}$ involves multiplying by $W_{hh}$ and the tanh derivative.

If $|W_{hh}| < 1$: gradients **vanish** exponentially
If $|W_{hh}| > 1$: gradients **explode** exponentially

**Result**: RNNs can't learn long-range dependencies well.

---

## 4. LSTM: Long Short-Term Memory

### The Key Insight

Add an explicit **memory cell** $c_t$ with **gates** that control information flow:

1. **Forget gate**: What to erase from memory
2. **Input gate**: What to write to memory
3. **Output gate**: What to output from memory

### The Equations

$$f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)$$ (Forget gate)
$$i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i)$$ (Input gate)
$$\\tilde{c}_t = \\tanh(W_c [h_{t-1}, x_t] + b_c)$$ (Candidate memory)
$$c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t$$ (New memory)
$$o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o)$$ (Output gate)
$$h_t = o_t \\odot \\tanh(c_t)$$ (Hidden state)

### Why This Solves Vanishing Gradients

The cell state $c_t$ has an **additive** update:
$$c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t$$

Gradients can flow through the $+$ without multiplication. If $f_t \\approx 1$, information (and gradients) flow unchanged.

---

## 5. LSTM Implementation

\`\`\`python
class LSTM:
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        combined_dim = input_dim + hidden_dim
        
        # All gates share similar structure
        self.W_f = np.random.randn(hidden_dim, combined_dim) * 0.01
        self.W_i = np.random.randn(hidden_dim, combined_dim) * 0.01
        self.W_c = np.random.randn(hidden_dim, combined_dim) * 0.01
        self.W_o = np.random.randn(hidden_dim, combined_dim) * 0.01
        
        self.b_f = np.ones((hidden_dim, 1))  # Initialize forget bias to 1
        self.b_i = np.zeros((hidden_dim, 1))
        self.b_c = np.zeros((hidden_dim, 1))
        self.b_o = np.zeros((hidden_dim, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def forward_step(self, x, h_prev, c_prev):
        # Concatenate input and previous hidden state
        combined = np.vstack([h_prev, x])
        
        # Compute gates
        f = self.sigmoid(self.W_f @ combined + self.b_f)
        i = self.sigmoid(self.W_i @ combined + self.b_i)
        c_tilde = np.tanh(self.W_c @ combined + self.b_c)
        o = self.sigmoid(self.W_o @ combined + self.b_o)
        
        # Update cell state and hidden state
        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)
        
        return h, c
\`\`\`

---

## 6. GRU: A Simpler Alternative

GRU combines the forget and input gates into a single **update gate**:

$$z_t = \\sigma(W_z [h_{t-1}, x_t])$$ (Update gate)
$$r_t = \\sigma(W_r [h_{t-1}, x_t])$$ (Reset gate)
$$\\tilde{h}_t = \\tanh(W [r_t \\odot h_{t-1}, x_t])$$ (Candidate)
$$h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t$$

**Fewer parameters**, similar performance to LSTM.

---

## 7. Bidirectional RNNs

Process sequence in both directions:

$$\\overrightarrow{h_t} = \\text{RNN}(x_t, \\overrightarrow{h_{t-1}})$$
$$\\overleftarrow{h_t} = \\text{RNN}(x_t, \\overleftarrow{h_{t+1}})$$
$$h_t = [\\overrightarrow{h_t}; \\overleftarrow{h_t}]$$

**Use case**: When you have access to the full sequence (e.g., classification, not generation).

---

## 8. Key Takeaways

1. **RNNs maintain hidden state** to process variable-length sequences
2. **Vanilla RNNs suffer from vanishing gradients** through long sequences
3. **LSTM adds a cell state** with gates that control information flow
4. **The forget gate initialized to 1** helps preserve information by default
5. **GRU is simpler** with fewer parameters, often similar performance
`,

    "Text Preprocessing": `
# Text Preprocessing & Tokenization

## 1. Why Tokenization Matters

Machine learning models don't understand text; they understand numbers. Tokenization is the bridge.

**Naive approach**: Split by space.
- "Don't" → ["Don", "'", "t"]? ["Don't"]?
- "unhappiness" → ["unhappiness"] (Model misses connection to "happy")

**Modern approach**: Subword tokenization (BPE, WordPiece).

---

## 2. Byte Pair Encoding (BPE)

### The Algorithm

1. Start with vocabulary = all individual characters.
2. Find the most frequent pair of adjacent tokens (e.g., "e" + "s" → "es").
3. Merge them into a new token.
4. Repeat until vocab size reached.

### Example Trace

Corpus: "hug", "pug", "pun", "bun"

1. Base vocab: {h, u, g, p, n, b}
2. Pair frequency: "u"+"g" appears most. Merge → "ug".
3. New vocab: {h, ug, p, n, b}
4. Now "p"+"ug" → "pug".

### Why It Works

- **Common words** become single tokens ("the", "apple").
- **Rare words** are broken down ("unfriendly" → "un", "friend", "ly").
- **Zero OOV**: Any string can be represented by characters.

### Implementation

\`\`\`python
def train_bpe(corpus, vocab_size):
    # Initialize vocab with characters
    vocab = set(list("".join(corpus)))
    splits = {word: list(word) for word in corpus}
    
    while len(vocab) < vocab_size:
        pairs = {}
        for word, char_list in splits.items():
            for i in range(len(char_list) - 1):
                pair = (char_list[i], char_list[i+1])
                pairs[pair] = pairs.get(pair, 0) + 1
        
        if not pairs: break
            
        # Find best pair
        best_pair = max(pairs, key=pairs.get)
        vocab.add("".join(best_pair))
        
        # Merge pair in all words
        for word in splits:
            new_list = []
            i = 0
            while i < len(splits[word]):
                if i < len(splits[word])-1 and \\
                   (splits[word][i], splits[word][i+1]) == best_pair:
                    new_list.append("".join(best_pair))
                    i += 2
                else:
                    new_list.append(splits[word][i])
                    i += 1
            splits[word] = new_list
            
    return vocab
\`\`\`

---

## 3. Advanced Tokenizers

| Method | Used In | Logic |
|--------|---------|-------|
| **BPE** | GPT-2, RoBERTa | Merge frequent pairs |
| **WordPiece** | BERT | Merge pairs maximizing likelihood |
| **Unigram** | T5, ALBERT | Start huge, prune tokens minimizing loss |

**SentencePiece**: Treats space as just another character ("_"), allowing reversible tokenization without pre-segmentation.

---

## 4. Key Takeaways

1. **Subword tokenization** solves the Out-Of-Vocabulary (OOV) problem.
2. **Frequency-based merging** balances vocab size and sequence length.
3. **Reversibility** is crucial for generation (detokenization).
`,

    "Classical NLP": `
# Classical NLP: TF-IDF & N-grams

## 1. Bag of Words (BoW)

Represent text as specific word counts. Order is lost.

Doc 1: "The cat sat"
Doc 2: "The dog sat"

Vocab: [The, cat, dog, sat]
Vec 1: [1, 1, 0, 1]
Vec 2: [1, 0, 1, 1]

**Problem**: "The" is frequent but useless. "Cat" is rare but important.

---

## 2. TF-IDF: Reweighting by Rarity

**Term Frequency (TF)**: How often word $t$ appears in doc $d$.
$$TF(t, d) = \\frac{\\text{count}(t, d)}{\\text{total words in } d}$$

**Inverse Document Frequency (IDF)**: How informative word $t$ is.
$$IDF(t) = \\log \\frac{N}{|\\{d \\in D : t \\in d\\}|}$$

**TF-IDF**:
$$TF\\text{-}IDF = TF(t, d) \\times IDF(t)$$

### Intuition

- **High TF-IDF**: Frequent in *this* document, rare *elsewhere* (Signature word).
- **Low TF-IDF**: Frequent everywhere (Stopwords).

### Implementation

\`\`\`python
import numpy as np

def compute_tfidf(corpus):
    # 1. Build Vocab
    vocab = sorted(list(set(" ".join(corpus).split())))
    N = len(corpus)
    
    # 2. Compute IDF
    idf = {}
    for word in vocab:
        count = sum(1 for doc in corpus if word in doc.split())
        idf[word] = np.log(N / (count + 1e-10))
    
    # 3. Compute TF-IDF Vectors
    vectors = []
    for doc in corpus:
        words = doc.split()
        vec = []
        for word in vocab:
            tf = words.count(word) / len(words)
            vec.append(tf * idf[word])
        vectors.append(vec)
        
    return vocab, np.array(vectors)
\`\`\`

---

## 3. N-gram Language Models

Estimate prob of word given previous $N-1$ words.

$$P(w_t | w_{t-N+1}...w_{t-1}) = \\frac{C(w_{t-N+1}...w_t)}{C(w_{t-N+1}...w_{t-1})}$$

**Markov Assumption**: Next word depends only on recent history.

### The Sparsity Problem

If "cat sat on **mat**" never appears in training, prob is 0.

**Solution: Smoothing**

**Laplace (Add-one) Smoothing**:
$$P_{laplace} = \\frac{C(context, word) + 1}{C(context) + |V|}$$

**Kneser-Ney Smoothing**:
Uses "continuation probability"—how likely is a word to complete *any* context?
Essential for n-grams to work in practice.

---

## 4. Key Takeaways

1. **TF-IDF** highlights specific, informative words in a document.
2. **N-grams** capture local context but suffer from sparsity.
3. **Smoothing** is mathematically required to handle unseen events.
4. **Cosine Similarity** on TF-IDF vectors is the baseline for information retrieval.
`,

    "Evaluation & Metrics": `
## 1. Perplexity: The Language Model Metric


### Definition

Perplexity measures how "surprised" the model is by the test data:

        $$PPL = \\exp\\left(-\\frac{ 1}{ N }\\sum_{ i = 1 }^ N \\log P(w_i | context) \\right) $$

### Intuition

    - PPL of 100: Model is as uncertain as choosing uniformly among 100 words
        - Lower is better: The model assigns high probability to the true next word

### Example

\`\`\`python
def perplexity(log_probs):
    """
    log_probs: log probability assigned to each true token
    """
    avg_neg_log_prob = -np.mean(log_probs)
    return np.exp(avg_neg_log_prob)

# If model prediction:
# P("cat") = 0.6, P("sat") = 0.1, P("mat") = 0.3
# PPL = exp(-(log(0.6) + log(0.1) + log(0.3))/3) = 4.93
\`\`\`

### Why Perplexity Works

It's the **geometric mean** of 1/probability. If your model assigns high probability to true words, perplexity is low.

---

## 2. BLEU Score: Translation Quality

### The Problem

How do you compare "The cat sat on the mat" to "A cat is sitting on the mat"?

Both are valid translations, but word-for-word comparison would score 0%.

### BLEU's Solution

Measure **n-gram precision**: What fraction of generated n-grams appear in the reference?

$$BLEU = BP \\cdot \\exp\\left(\\sum_{n=1}^N w_n \\log p_n\\right)$$

where:
- $p_n$ = precision of n-grams
- $w_n$ = weight (usually $1/N$)
- $BP$ = brevity penalty (penalize short outputs)

### Modified Precision

To prevent gaming (generating "the the the..."), use **clipped counts**:

\`\`\`python
from collections import Counter

def modified_precision(reference, hypothesis, n):
    # Get n-grams
    ref_ngrams = Counter(zip(*[reference[i:] for i in range(n)]))
    hyp_ngrams = Counter(zip(*[hypothesis[i:] for i in range(n)]))
    
    # Clip counts to reference counts
    clipped = {ng: min(count, ref_ngrams.get(ng, 0)) 
               for ng, count in hyp_ngrams.items()}
    
    return sum(clipped.values()) / max(sum(hyp_ngrams.values()), 1)
\`\`\`

### Brevity Penalty

Prevent models from outputting single high-confidence words:

$$BP = \\begin{cases} 1 & \\text{if } c > r \\\\ e^{1-r/c} & \\text{if } c \\leq r \\end{cases}$$

where $c$ = candidate length, $r$ = reference length.

---

## 3. ROUGE: Summarization Quality

### The Difference from BLEU

- BLEU: **Precision** (how much of output is correct?)
- ROUGE: **Recall** (how much of reference is captured?)

### ROUGE-N

$$ROUGE\\text{-}N = \\frac{\\sum_{gram \\in ref} Count_{match}}{\\sum_{gram \\in ref} Count}$$

### ROUGE-L

Based on **Longest Common Subsequence**:

$$ROUGE\\text{-}L = F_1(LCS)$$

\`\`\`python
def lcs_length(X, Y):
    """Dynamic programming LCS."""
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

def rouge_l(reference, hypothesis):
    lcs = lcs_length(reference, hypothesis)
    precision = lcs / len(hypothesis) if hypothesis else 0
    recall = lcs / len(reference) if reference else 0
    
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)
\`\`\`

---

## 4. BERTScore: Semantic Similarity

### The Problem with N-gram Metrics

"The dog ran quickly" and "The canine sprinted fast" have low BLEU but identical meaning.

### BERTScore's Solution

Compare embeddings, not exact matches:

$$P = \\frac{1}{|\\hat{y}|}\\sum_{\\hat{y}_j} \\max_{y_i} \\cos(y_i, \\hat{y}_j)$$
$$R = \\frac{1}{|y|}\\sum_{y_i} \\max_{\\hat{y}_j} \\cos(y_i, \\hat{y}_j)$$
$$F_1 = 2 \\cdot \\frac{P \\cdot R}{P + R}$$

Each word finds its best semantic match in the other sentence.

---

## 5. Task-Specific Important Metrics

| Task | Metrics | Why |
|------|---------|-----|
| Translation | BLEU, COMET | N-gram overlap, neural scoring |
| Summarization | ROUGE-1/2/L | Recall of content |
| Generation | Perplexity, Human eval | Language quality |
| QA | Exact Match, F1 | Answer correctness |
| Classification | Accuracy, F1 | Class correctness |

---

## 6. Key Takeaways

1. **Perplexity** measures how well a model predicts text (lower = better)
2. **BLEU** measures precision of n-grams for translation
3. **ROUGE** measures recall of n-grams for summarization
4. **BERTScore** captures semantic similarity, not just exact matches
5. **No single metric is perfect**—always include human evaluation for generation
`,

    "Sequence Labeling & NER": `
# Sequence Labeling & NER

## 1. The Problem

Given a sequence of words, assign a tag to each word.
- **POS Tagging**: Noun, Verb, Adjective
- **NER**: Person, Org, Location, O (Outside)

Input: "Apple bought a startup"
Output: [ORG, O, O, O]

**Challenge**: Decisions are independent. "Apple" is usually a fruit, but here it's an Org.

---

## 2. Hidden Markov Models (HMM)

A generative model $P(X, Y) = P(Y)P(X|Y)$.

Two assumptions:
1. **Markov Assumption**: $P(y_t | y_{t-1}...y_1) \\approx P(y_t | y_{t-1})$
2. **Output Independence**: $P(x_t | y_t...x_t...) \\approx P(x_t | y_t)$

Parameters:
- **Transition**: $A_{ij} = P(y_t = j | y_{t-1} = i)$
- **Emission**: $B_{jk} = P(x_t = k | y_t = j)$

### The Viterbi Algorithm

How to find the best sequence $y^* = \\arg\\max_y P(y|x)$?
Brute force is $O(T^N)$. Viterbi is $O(TN^2)$ using dynamic programming.

$$V_t(j) = \\max_{i} [V_{t-1}(i) \\cdot A_{ij}] \\cdot B_{j, x_t}$$

- $V_t(j)$: Prob of best path ending in state $j$ at time $t$.
- We keep backpointers to reconstruct the path.

\`\`\`python
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    # Initialize (t=0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max(
                (V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) 
                for y0 in states
            )
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        path = newpath

    # Termination
    (prob, state) = max((V[len(obs) - 1][y], y) for y in states)
    return prob, path[state]
\`\`\`

---

## 3. Conditional Random Fields (CRF)

HMMs are generative and strictly local. CRFs are **discriminative** $P(Y|X)$ and can use global features.

$$P(y|x) = \\frac{1}{Z(x)} \\exp\\left(\\sum_{t} \\sum_{k} \\lambda_k f_k(y_{t-1}, y_t, x, t)\\right)$$

where $f_k$ are feature functions (e.g., "word starts with capital AND prev tag was DET").

**Why CRF?** It solves the "Label Bias Problem" of MEMMs and models the *entire sequence* jointly.

---

## 4. Key Takeaways

1. **Viterbi Algorithm** finds the global optimal path in $O(TN^2)$.
2. **HMMs** generate data (Bayesian), **CRFs** discriminate (Conditional).
3. **CRF** is the gold standard for sequence labeling before Transformers replaced everything.
`,

    "Text Classification": `
# Text Classification: Beyond Simple Baselines

## 1. Text CNN (Kim 2014)

**Idea**: Treat text like an image.
- **Image**: 2D grid of pixels.
- **Text**: 1D sequence of word vectors (or 2D if you count embedding dim).

### The Architecture

1. **Input**: Sentence matrix (Seq Len $\\times$ Embed Dim)
2. **Convolution**: Filters of size $2, 3, 4, 5$ (sliding over 2-5 words at a time).
3. **Max Pooling**: Take the *maximum* value from each filter map.
   - Captures "Did this n-gram appear anywhere?"
4. **Output**: Concatenate pooled features $\\to$ Softmax.

**Why it works**: Excellent at detecting **local patterns** (e.g., "not good", "very happy") regardless of position.

\`\`\`python
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes):
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (k, embed_dim)) 
            for k in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1) # (N, 1, Seq, Dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return self.fc(x)
\`\`\`

---

## 2. Hierarchical Attention Networks (HAN)

**Insight**: Documents have structure.
Words $\\to$ Sentences $\\to$ Document

HAN mirrors this:
1. **Word Encoder**: RNN + Attention $\\to$ Sentence Vector
2. **Sentence Encoder**: RNN + Attention $\\to$ Document Vector

**Interpretability**: You can see which *sentences* and which *words* mattered most.

---

## 3. Focal Loss for Imbalanced Data

In text classification, some classes (e.g., "Fraud") are rare. Cross-entropy is overwhelmed by easy negatives.

$$FL(p_t) = -(1 - p_t)^\\gamma \\log(p_t)$$

- When $p_t \\approx 1$ (easy example), $(1-p_t)^\\gamma \\approx 0$. Loss is down-weighted.
- When $p_t$ is small (hard example), weight is high.
- **Focuses training on hard, misclassified examples.**

---

## 4. Key Takeaways

1. **Text CNNs** are fast and effective for spotting key phrases.
2. **Hierarchical Attention** matches document structure and offers explainability.
3. **Focal Loss** is crucial when classes are heavily imbalanced (common in content moderation).
`,

    "LLM Quantization": `
# LLM Quantization: Running 70B Models on Your Laptop

## 1. The Fundamental Problem

**LLaMA-70B in FP32**: 70B params × 4 bytes = **280GB**
**MacBook Pro M3 Max**: 64GB RAM

**Question**: Can we compress the model without losing much quality?

**Answer**: Yes! Weights are over-parameterized. Most information is in the top bits.

---

## 2. Why Quantization Works

### The Key Insight

Neural network weights follow approximately **Gaussian distributions**.

Most weights are small. A few are large (outliers). The outliers matter most!

**Observation**: Weights trained with billions of tokens are highly redundant.

### Information Theory Perspective

32 bits per weight × 70B weights = 280GB of "information"

But most of those bits are **noise** from training randomness. The actual "useful" information is much smaller.

---

## 3. Quantization Basics

### Uniform Quantization

Map floating-point values to integers:

$$x_q = \\text{round}\\left(\\frac{x}{s}\\right) + z$$

where:
- $s$ = scale factor (determines range)
- $z$ = zero point (for asymmetric quantization)

**De-quantization**:
$$x \\approx s(x_q - z)$$

### Per-Tensor vs Per-Channel

| Granularity | Description | Quality | Overhead |
|-------------|-------------|---------|----------|
| Per-tensor | One scale for entire matrix | Lowest | Smallest |
| Per-channel | One scale per output channel | Medium | Medium |
| Per-group | One scale per 128 weights | Highest | Larger |

**Modern methods use per-group** for best accuracy.

---

## 4. The Outlier Problem

### Why Naïve Quantization Fails

Some weights are **100× larger** than typical weights.

If you set the quantization range to cover outliers:
- Outliers fit: ✓
- Small weights lose precision: ✗

If you clip outliers:
- Small weights fit: ✓
- Outliers are destroyed: ✗

### The Solution: Mixed Precision

Keep a small fraction of weights in higher precision (FP16).
Quantize the rest to INT4/INT8.

---

## 5. GPTQ: The Gold Standard

### The Idea

**Optimal Brain Quantization**: Quantize weights one at a time, compensating for error in remaining weights.

**The Math**: For a linear layer $Y = XW$, minimize:
$$\\|XW - XW_q\\|_F^2$$

This is equivalent to:
$$\\min_{w_q} (w - w_q)^T H (w - w_q)$$

where $H = X^T X$ is the Hessian.

### The Algorithm (Simplified)

1. Compute Hessian $H = X^T X$ from calibration data
2. For each column of $W$:
   - Quantize the weight
   - Compute quantization error
   - Adjust remaining weights to compensate (using $H^{-1}$)

**Key insight**: Errors propagate. Compensating early prevents cascading errors.

### Why It Works So Well

- Uses **second-order information** (Hessian)
- Processes weights in order of importance (larger Hessian = more important)
- Compensation distributes error across many weights

---

## 6. AWQ: Activation-Aware Quantization

### The Insight

Not all weights are equally important. **Weights connected to large activations matter more**.

If $Y = X \\cdot W$, and some activations $X_i$ are consistently large, then $W_{:,i}$ must be preserved carefully.

### The Method

1. Run calibration data through the model
2. Measure activation magnitudes
3. Scale weights inversely: large-activation weights get less quantization error

$$W' = W \\cdot \\text{diag}(s)^{-1}$$

Compensate by scaling activations/inputs.

### Comparison to GPTQ

| Aspect | GPTQ | AWQ |
|--------|------|-----|
| Focus | Weight distribution | Activation importance |
| Speed | Slower (Hessian computation) | Faster |
| Quality | Excellent | Comparable |
| Best for | Highest quality | Speed + quality |

---

## 7. bitsandbytes: Dynamic Quantization

### The Approach

Quantize at inference time, not ahead of time.

**NF4** (Normal Float 4): Quantization levels optimized for Gaussian-distributed weights.

Instead of uniform spacing (0, 1, 2, 3...), use levels matched to Gaussian quantiles.

### When to Use

- **GPTQ/AWQ**: When you're deploying a specific model
- **bitsandbytes**: When you want flexibility (any model, any time)

---

## 8. Practical Recommendations

| Model Size | Hardware | Recommendation |
|------------|----------|----------------|
| 7B | 8GB GPU | INT4 (GPTQ/AWQ) |
| 13B | 16GB GPU | INT4 or INT8 |
| 70B | 48GB GPU | INT4 |
| 70B | CPU only | GGUF with llama.cpp |

### Quality Expectations

| Precision | Perplexity Increase | Practical Impact |
|-----------|--------------------|--------------------|
| FP16 | 0% (baseline) | None |
| INT8 | 0.1-0.5% | Negligible |
| INT4 | 1-3% | Minor |
| INT3 | 5-15% | Noticeable |

---

## 9. Key Takeaways

1. **Quantization works** because neural net weights are redundant
2. **Outliers are the challenge**—mixed precision or compensation needed
3. **GPTQ uses optimal brain quantization** with Hessian-based compensation
4. **AWQ focuses on activations**—weights connected to large activations matter most
5. **4-bit quantization** gives ~8× compression with minimal quality loss
6. **Per-group quantization** is crucial for maintaining accuracy
`,

    "Embedding Retrieval": `
# Embedding-Based Retrieval: Search at Scale

## 1. The Fundamental Problem

**Search engine**: Given a query, find relevant documents from billions.
**Recommendation**: Given a user, find matching items from millions.

**Brute force**: Compare query to every item = $O(n)$ = too slow!

---

## 2. The Embedding Approach

### The Idea

Map queries and items to vectors such that:
- **Similar items have similar vectors** (high dot product/low distance)
- **Search = finding nearest neighbors in vector space**

### Two-Tower Architecture (Revisited)

Query Encoder: $q = f_\\theta(\\text{query})$
Item Encoder: $d = g_\\phi(\\text{item})$

Similarity: $\\text{score}(q, d) = q^T d$

---

## 3. Approximate Nearest Neighbor (ANN)

### The Challenge

Even with embeddings, comparing to 1B vectors is expensive.

**ANN trades accuracy for speed**: Find approximately nearest neighbors.

### Key Algorithms

#### Inverted File Index (IVF)

1. Cluster vectors into $k$ centroids (k-means)
2. At search time:
   - Find closest centroids to query
   - Only search vectors in those clusters

**Complexity**: $O(k + n/k \\cdot \\text{nprobe})$ instead of $O(n)$

#### Hierarchical Navigable Small Worlds (HNSW)

Build a graph where:
- Each vector connects to its nearest neighbors
- Multiple layers: coarse → fine

**Search**: Start at top layer, greedily descend

**Complexity**: $O(\\log n)$ average case

#### Comparison

| Algorithm | Build Time | Query Time | Memory | Recall@10 |
|-----------|-----------|------------|--------|-----------|
| Brute Force | None | O(n) | Low | 100% |
| IVF | O(n) | O(n/k) | Low | 95%+ |
| HNSW | O(n log n) | O(log n) | High | 99%+ |

---

## 4. Product Quantization (PQ)

### The Memory Problem

1B vectors × 768 dimensions × 4 bytes = **3 TB!**

### The Solution

Split each vector into subvectors, quantize each independently.

768-dim vector → 8 sub-vectors of 96 dimensions each
Each sub-vector → 1 byte (256 clusters)

**Compression**: 768 × 4 = 3072 bytes → 8 bytes = **384× compression!**

### Why It Works

Distance computation:
$$d(q, v) = \\sum_{i=1}^m d(q_i, c_i[v_i])$$

Pre-compute distances from query to all centroids → **table lookup!**

---

## 5. Hybrid Search

### Dense + Sparse

Dense (embeddings) alone fails for:
- Rare entities ("John Smith from Peoria born 1983")
- Exact matches expected ("error code ABC123")

**Solution**: Combine dense retrieval with sparse (BM25/TF-IDF):

$$\\text{score} = \\alpha \\cdot \\text{dense}(q, d) + (1-\\alpha) \\cdot \\text{sparse}(q, d)$$

---

## 6. Re-ranking

### Two-Stage Retrieval

**Stage 1 (Retrieval)**: ANN search, fast, recall-focused
- Return top 1000 candidates

**Stage 2 (Re-ranking)**: Cross-encoder, slow, precision-focused
- Score each candidate with full attention over (query, document)

### Why Re-ranking Helps

Two-tower models are fast but limited (no query-doc interaction).
Cross-encoders are slow but accurate (full attention).

Use cheap model to filter, expensive model to rank.

---

## 7. Implementation Considerations

### Index Updates

**Problem**: New documents arrive constantly. How to update the index?

**Solutions**:
1. **Rebuild periodically** (simplest)
2. **Delta index**: Keep small index of new docs, merge periodically
3. **Real-time updates**: More complex, HNSW supports this

### Sharding

For billions of vectors, shard across machines:
- **Document sharding**: Each shard holds subset of docs
- **Query replication**: Send query to all shards, merge results

---

## 8. Key Takeaways

1. **Embeddings enable semantic search** beyond keyword matching
2. **ANN is essential at scale**—exact search is too slow
3. **HNSW** is often the best choice for quality/speed tradeoff
4. **Product Quantization** reduces memory 100×+
5. **Hybrid search** combines semantic understanding with exact matching
6. **Re-ranking with cross-encoders** improves precision
`,

    "Decoding Strategies": `
# Decoding Strategies: How LLMs Generate Text

## 1. The Fundamental Problem

An autoregressive language model outputs a probability distribution over next tokens.

**The question**: How do we convert probabilities into actual text?

---

## 2. Greedy Decoding

At each step, pick the most likely token.

**Problem**: Locally optimal does not equal globally optimal.

**Symptoms**:
- Repetitive text
- Generic outputs
- Gets stuck in loops

---

## 3. Beam Search

Keep track of top-k sequences ("beams").

At each step:
1. Expand each beam with all vocabulary tokens
2. Score all candidates
3. Keep only top-k sequences

**Length normalization**: Divide score by length to avoid favoring short sequences.

**Paradox**: Larger beam can produce WORSE outputs because it favors short, generic sequences.

---

## 4. Temperature Sampling

Instead of taking the max, sample from the distribution.

Temperature $T$ controls randomness:
- $T \\to 0$: Deterministic (greedy)
- $T = 1$: Original distribution
- $T > 1$: More random/creative

---

## 5. Top-k Sampling

Only sample from the top-k most likely tokens.

Typical k = 10-50.

**Problem**: Fixed k doesn't adapt to distribution shape.

---

## 6. Nucleus (Top-p) Sampling

Include tokens until cumulative probability reaches p.

**Adaptive**: When model is confident, nucleus is small. When uncertain, nucleus is large.

**Top-p is generally preferred** because it adapts to model confidence.

---

## 7. Combining Strategies

Modern systems combine:
- Temperature: 0.7-1.0
- Top-p: 0.9-0.95
- Top-k: 40-100 (optional)

---

## 8. Constrained Decoding

Mask tokens that would violate constraints:
- JSON output: Only allow valid keys/values
- Code: Only allow valid syntax
- Rhyming: Only allow rhyming words

---

## 9. Key Takeaways

1. **Greedy fails** because local optima ≠ global optima
2. **Beam search keeps options open** but favors short text
3. **Temperature controls randomness**
4. **Top-p adapts** to model confidence—generally preferred
5. **Constrained decoding** enforces output formats
`,

    "Advanced NLP Problems": `
# Advanced NLP Problems: Real-World Applications

## 1. Span Extraction for QA

Given: Context paragraph + Question
Output: Exact span from context

**Architecture**:
1. Encode context + question together (BERT-style)
2. Predict P(start) and P(end) for each token
3. Find span with highest P(start) × P(end)

---

## 2. Sentence Similarity (SBERT)

**Problem**: BERT requires encoding pairs = O(n²) for n sentences.

**Solution**: Train to produce fixed-size sentence embeddings.

Similarity = cosine(embedding_A, embedding_B)

**Why this matters**: O(n) embeddings + ANN search = O(n log n)

---

## 3. Document Retrieval with BM25

BM25 improves on TF-IDF with:
- Better term frequency saturation
- Document length normalization

**Why BM25 still works**:
1. No training required
2. Fast (inverted index)
3. Excels at exact match

**Best systems**: Combine BM25 + dense retrieval

---

## 4. Named Entity Recognition (NER)

Identify entities: "**Barack Obama** [PERSON] was born in **Hawaii** [LOCATION]"

**BIO Tagging**:
- B-XXX: Beginning of entity
- I-XXX: Inside entity
- O: Outside any entity

**Architecture**: BERT + Linear + CRF (CRF enforces valid sequences)

---

## 5. Text Summarization

| Type | Description | Pros | Cons |
|------|-------------|------|------|
| Extractive | Select sentences | Faithful | Choppy |
| Abstractive | Generate new text | Fluent | May hallucinate |

---

## 6. Machine Translation

Transformer encoder-decoder with:
- Teacher forcing during training
- Label smoothing
- Beam search decoding

---

## 7. Dialogue Systems

Components:
1. NLU: Understand intent
2. State Tracking: Remember context
3. Policy: Decide response
4. NLG: Generate text

**Modern approach**: Instruction-tuned LLM + RAG

---

## 8. Key Takeaways

1. **Span extraction** predicts start/end positions
2. **SBERT** enables efficient sentence comparison
3. **BM25 + dense** is the best retrieval combo
4. **NER uses BIO tagging** with CRF
5. **Summarization**: Extractive is faithful, abstractive is fluent
`
};

// DEDICATED CONTENT FOR EACH TOPIC - NO MORE ALIASES

nlpContent["CODE: Implement Tokenizer from Scratch"] = `
# Implementing a Tokenizer from Scratch

## Why Tokenization Matters

Tokenization is the first step in any NLP pipeline. A tokenizer converts raw text into discrete units (tokens) that models can process.

**Key decisions**:
- Word-level: "I love NLP" → ["I", "love", "NLP"]
- Subword-level: "unhappiness" → ["un", "happi", "ness"]
- Character-level: "cat" → ["c", "a", "t"]

## Simple Word Tokenizer

\`\`\`python
import re

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.id_to_token = {}
        self.next_id = 0
    
    def tokenize(self, text):
        # Lowercase and split on whitespace/punctuation
        tokens = re.findall(r"\\b\\w+\\b|[^\\w\\s]", text.lower())
        return tokens
    
    def build_vocab(self, texts):
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = self.next_id
                    self.id_to_token[self.next_id] = token
                    self.next_id += 1
    
    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(t, self.vocab.get("<UNK>", 0)) for t in tokens]
    
    def decode(self, ids):
        return " ".join([self.id_to_token.get(i, "<UNK>") for i in ids])
\`\`\`

## Key Insight

The tokenizer must handle:
1. **Unknown words** - Use <UNK> token or subword fallback
2. **Special tokens** - <PAD>, <BOS>, <EOS>, <SEP>
3. **Normalization** - Lowercase, unicode, whitespace
`;

nlpContent["CODE: Implement BPE (Byte Pair Encoding)"] = `
# Byte Pair Encoding (BPE) Implementation

## The Core Idea

BPE is a data compression algorithm adapted for NLP. It iteratively merges the most frequent pair of bytes (or characters) to build a vocabulary of subwords.

**Why BPE?**
- Handles rare words by breaking into known subwords
- Fixed vocabulary size
- Language-agnostic

## Algorithm

1. Start with character-level vocabulary
2. Count all adjacent pairs
3. Merge most frequent pair into new token
4. Repeat until vocabulary size reached

## Implementation

\`\`\`python
from collections import Counter, defaultdict

def get_pairs(word_freqs):
    pairs = defaultdict(int)
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_pair(pair, word_freqs):
    new_word_freqs = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    
    for word, freq in word_freqs.items():
        new_word = word.replace(bigram, replacement)
        new_word_freqs[new_word] = freq
    return new_word_freqs

def train_bpe(corpus, num_merges):
    # Initialize: split each word into characters
    word_freqs = Counter(corpus)
    word_freqs = {" ".join(word): freq for word, freq in word_freqs.items()}
    
    merges = []
    for i in range(num_merges):
        pairs = get_pairs(word_freqs)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        word_freqs = merge_pair(best_pair, word_freqs)
        merges.append(best_pair)
    
    return merges

# Usage
corpus = ["low", "lower", "newest", "widest"]
merges = train_bpe(corpus, num_merges=10)
\`\`\`

## Key Insight

BPE learns a compression scheme that naturally captures morphology: "un" + "happi" + "ness" emerges from data.
`;

nlpContent["CODE: Implement TF-IDF from Scratch"] = `
# TF-IDF Implementation from Scratch

## The Intuition

TF-IDF measures word importance in a document relative to a corpus.

- **TF (Term Frequency)**: How often does this word appear in THIS document?
- **IDF (Inverse Document Frequency)**: How rare is this word across ALL documents?

$$\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\text{IDF}(t)$$

## Mathematical Formulation

$$\\text{TF}(t, d) = \\frac{\\text{count}(t, d)}{\\text{total terms in } d}$$

$$\\text{IDF}(t) = \\log\\frac{N}{\\text{df}(t) + 1}$$

where N = total documents, df(t) = documents containing term t.

## Implementation

\`\`\`python
import numpy as np
from collections import Counter
import math

class TFIDFVectorizer:
    def __init__(self):
        self.vocab = {}
        self.idf = {}
    
    def fit(self, documents):
        # Build vocabulary
        all_words = set()
        for doc in documents:
            all_words.update(doc.lower().split())
        self.vocab = {word: i for i, word in enumerate(sorted(all_words))}
        
        # Compute IDF
        N = len(documents)
        df = Counter()
        for doc in documents:
            unique_words = set(doc.lower().split())
            for word in unique_words:
                df[word] += 1
        
        for word in self.vocab:
            self.idf[word] = math.log(N / (df[word] + 1))
    
    def transform(self, documents):
        vectors = []
        for doc in documents:
            words = doc.lower().split()
            tf = Counter(words)
            total = len(words)
            
            vec = np.zeros(len(self.vocab))
            for word, count in tf.items():
                if word in self.vocab:
                    tf_val = count / total
                    vec[self.vocab[word]] = tf_val * self.idf[word]
            vectors.append(vec)
        return np.array(vectors)

# Usage
docs = ["the cat sat on mat", "the dog ran in park", "cat and dog are pets"]
vectorizer = TFIDFVectorizer()
vectorizer.fit(docs)
vectors = vectorizer.transform(docs)
\`\`\`

## Key Insight

TF-IDF down-weights common words ("the", "is") and up-weights distinctive words, making it ideal for document retrieval and classification.
`;

// Continuing with more dedicated content...

nlpContent["CODE: Implement N-gram Language Model"] = `
# N-gram Language Model Implementation

## The Core Idea

An N-gram model predicts the next word based on the previous N-1 words.

$$P(w_n | w_1, ..., w_{n-1}) \\approx P(w_n | w_{n-N+1}, ..., w_{n-1})$$

## Implementation

\`\`\`python
from collections import Counter, defaultdict
import random

class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
    
    def train(self, corpus):
        # corpus is list of tokenized sentences
        for sentence in corpus:
            # Add start/end tokens
            tokens = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                word = tokens[i + self.n - 1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
    
    def probability(self, word, context):
        context = tuple(context[-(self.n-1):])
        if self.context_counts[context] == 0:
            return 1e-10  # Smoothing
        return self.ngram_counts[context][word] / self.context_counts[context]
    
    def generate(self, max_length=20):
        context = ['<s>'] * (self.n - 1)
        result = []
        
        for _ in range(max_length):
            ctx = tuple(context[-(self.n-1):])
            if ctx not in self.ngram_counts:
                break
            
            words = list(self.ngram_counts[ctx].keys())
            counts = list(self.ngram_counts[ctx].values())
            word = random.choices(words, weights=counts)[0]
            
            if word == '</s>':
                break
            result.append(word)
            context.append(word)
        
        return result

# Usage
corpus = [["the", "cat", "sat"], ["the", "dog", "ran"]]
model = NGramModel(n=2)
model.train(corpus)
\`\`\`

## Key Insight

N-grams capture local dependencies but suffer from data sparsity for large N. Solution: smoothing (Laplace, Kneser-Ney).
`;

nlpContent["CODE: Implement Word2Vec Skip-gram"] = `
# Word2Vec Skip-gram Implementation

## The Skip-gram Objective

Given a center word, predict context words. Maximize:

$$J = \\frac{1}{T}\\sum_{t=1}^T \\sum_{-c \\le j \\le c, j \\ne 0} \\log P(w_{t+j} | w_t)$$

## Implementation

\`\`\`python
import numpy as np

class SkipGram:
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Two embedding matrices
        self.W_in = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W_out = np.random.randn(embed_dim, vocab_size) * 0.01
    
    def forward(self, center_idx, context_idx):
        # Get center word embedding
        h = self.W_in[center_idx]  # (embed_dim,)
        
        # Score all words
        scores = np.dot(h, self.W_out)  # (vocab_size,)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        
        # Loss = -log P(context | center)
        loss = -np.log(probs[context_idx] + 1e-10)
        
        return loss, probs, h
    
    def backward(self, center_idx, context_idx, probs, h, lr=0.01):
        # Gradient of softmax cross-entropy
        dscores = probs.copy()
        dscores[context_idx] -= 1  # (vocab_size,)
        
        # Gradient for W_out
        dW_out = np.outer(h, dscores)  # (embed_dim, vocab_size)
        
        # Gradient for hidden
        dh = np.dot(self.W_out, dscores)  # (embed_dim,)
        
        # Gradient for W_in
        dW_in = dh  # Just for this center word
        
        # Update
        self.W_out -= lr * dW_out
        self.W_in[center_idx] -= lr * dW_in
    
    def get_embedding(self, word_idx):
        return self.W_in[word_idx]
\`\`\`

## Key Insight

The learned embeddings capture semantic relationships: vec("king") - vec("man") + vec("woman") ≈ vec("queen")
`;

nlpContent["CODE: Implement Vanilla RNN Cell"] = `
# Vanilla RNN Cell Implementation

## The RNN Equations

$$h_t = \\tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

## Implementation

\`\`\`python
import numpy as np

class RNNCell:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize weights
        self.Wxh = np.random.randn(hidden_dim, input_dim) * 0.01
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.Why = np.random.randn(output_dim, hidden_dim) * 0.01
        self.bh = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))
    
    def forward(self, x, h_prev):
        # x: (input_dim, 1), h_prev: (hidden_dim, 1)
        
        # Hidden state
        h_next = np.tanh(
            np.dot(self.Wxh, x) + 
            np.dot(self.Whh, h_prev) + 
            self.bh
        )
        
        # Output
        y = np.dot(self.Why, h_next) + self.by
        
        # Cache for backprop
        cache = (x, h_prev, h_next)
        
        return h_next, y, cache
    
    def backward(self, dy, dh_next, cache):
        x, h_prev, h_next = cache
        
        # Gradient through output layer
        dWhy = np.dot(dy, h_next.T)
        dby = dy
        dh = np.dot(self.Why.T, dy) + dh_next
        
        # Gradient through tanh
        dtanh = (1 - h_next ** 2) * dh
        
        # Gradients for weights
        dWxh = np.dot(dtanh, x.T)
        dWhh = np.dot(dtanh, h_prev.T)
        dbh = dtanh
        
        # Gradient to previous hidden state
        dh_prev = np.dot(self.Whh.T, dtanh)
        
        return dh_prev, dWxh, dWhh, dWhy, dbh, dby
\`\`\`

## Key Insight

Vanilla RNNs suffer from vanishing gradients because gradients flow through many tanh activations, each shrinking the gradient.
`;

nlpContent["CODE: Implement LSTM Cell from Scratch"] = `
# LSTM Cell Implementation from Scratch

## Why LSTM?

LSTMs solve the vanishing gradient problem with a **cell state** that flows through time with minimal transformation.

## The LSTM Equations

**Gates**:
$$f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)$$ (forget gate)
$$i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i)$$ (input gate)
$$o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o)$$ (output gate)

**Cell state**:
$$\\tilde{C}_t = \\tanh(W_C [h_{t-1}, x_t] + b_C)$$ (candidate)
$$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$$

**Hidden state**:
$$h_t = o_t \\odot \\tanh(C_t)$$

## Implementation

\`\`\`python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        
        # Combined weights for efficiency
        concat_dim = input_dim + hidden_dim
        self.Wf = np.random.randn(hidden_dim, concat_dim) * 0.01
        self.Wi = np.random.randn(hidden_dim, concat_dim) * 0.01
        self.Wc = np.random.randn(hidden_dim, concat_dim) * 0.01
        self.Wo = np.random.randn(hidden_dim, concat_dim) * 0.01
        
        self.bf = np.zeros((hidden_dim, 1))
        self.bi = np.zeros((hidden_dim, 1))
        self.bc = np.zeros((hidden_dim, 1))
        self.bo = np.zeros((hidden_dim, 1))
    
    def forward(self, x, h_prev, c_prev):
        # Concatenate input and previous hidden
        concat = np.vstack([h_prev, x])
        
        # Gates
        f = sigmoid(np.dot(self.Wf, concat) + self.bf)
        i = sigmoid(np.dot(self.Wi, concat) + self.bi)
        c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)
        o = sigmoid(np.dot(self.Wo, concat) + self.bo)
        
        # Cell state and hidden state
        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)
        
        cache = (x, h_prev, c_prev, f, i, c_tilde, o, c, h, concat)
        return h, c, cache
\`\`\`

## Key Insight

The cell state $C_t$ acts as a "highway" where gradients can flow unchanged (when $f_t = 1$, $i_t = 0$).
`;

nlpContent["CODE: Implement GRU Cell"] = `
# GRU Cell Implementation

## GRU vs LSTM

GRU is a simplified LSTM with only 2 gates (reset, update) instead of 3.

## The GRU Equations

$$z_t = \\sigma(W_z [h_{t-1}, x_t])$$ (update gate)
$$r_t = \\sigma(W_r [h_{t-1}, x_t])$$ (reset gate)
$$\\tilde{h}_t = \\tanh(W [r_t \\odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t$$

## Implementation

\`\`\`python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class GRUCell:
    def __init__(self, input_dim, hidden_dim):
        concat_dim = input_dim + hidden_dim
        
        self.Wz = np.random.randn(hidden_dim, concat_dim) * 0.01
        self.Wr = np.random.randn(hidden_dim, concat_dim) * 0.01
        self.Wh = np.random.randn(hidden_dim, concat_dim) * 0.01
        
        self.bz = np.zeros((hidden_dim, 1))
        self.br = np.zeros((hidden_dim, 1))
        self.bh = np.zeros((hidden_dim, 1))
    
    def forward(self, x, h_prev):
        concat = np.vstack([h_prev, x])
        
        # Update and reset gates
        z = sigmoid(np.dot(self.Wz, concat) + self.bz)
        r = sigmoid(np.dot(self.Wr, concat) + self.br)
        
        # Candidate hidden state
        concat_reset = np.vstack([r * h_prev, x])
        h_tilde = np.tanh(np.dot(self.Wh, concat_reset) + self.bh)
        
        # New hidden state
        h = (1 - z) * h_prev + z * h_tilde
        
        return h
\`\`\`

## Key Insight

GRU merges cell state and hidden state into one, making it faster to train with similar performance to LSTM.
`;

nlpContent["CODE: Implement Scaled Dot-Product Attention"] = `
# Scaled Dot-Product Attention Implementation

## The Core Formula

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

## Why Scale by sqrt(d_k)?

Without scaling, dot products grow with dimension, pushing softmax into saturation where gradients vanish.

## Implementation

\`\`\`python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, seq_len, d_k)
    Returns: attention output and weights
    """
    d_k = Q.shape[-1]
    
    # Compute attention scores
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Apply mask (for decoder self-attention)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # Softmax
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights /= attention_weights.sum(axis=-1, keepdims=True)
    
    # Weighted sum of values
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

# Example
batch, seq_len, d_k = 2, 5, 64
Q = np.random.randn(batch, seq_len, d_k)
K = np.random.randn(batch, seq_len, d_k)
V = np.random.randn(batch, seq_len, d_k)
output, weights = scaled_dot_product_attention(Q, K, V)
\`\`\`

## Key Insight

Attention allows each position to "look at" all other positions, capturing long-range dependencies that RNNs struggle with.
`;

nlpContent["CODE: Implement Positional Encoding"] = `
# Positional Encoding Implementation

## Why Positional Encoding?

Transformers have no recurrence—they process all tokens in parallel. Without positional information, "The cat ate the fish" = "The fish ate the cat".

## The Sinusoidal Formula

$$PE_{(pos, 2i)} = \\sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \\cos(pos / 10000^{2i/d})$$

## Implementation

\`\`\`python
import numpy as np

def positional_encoding(max_len, d_model):
    """
    Returns: (max_len, d_model) positional encoding matrix
    """
    PE = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)
    
    return PE

# Example
PE = positional_encoding(max_len=100, d_model=512)
# PE[pos] gives the encoding for position pos
\`\`\`

## Key Insight

Sinusoidal encodings allow the model to learn relative positions because PE[pos+k] can be expressed as a linear function of PE[pos].
`;

nlpContent["CODE: Implement BLEU Score from Scratch"] = `
# BLEU Score Implementation

## What BLEU Measures

BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between candidate and reference translations.

## The Formula

$$\\text{BLEU} = BP \\cdot \\exp\\left(\\sum_{n=1}^N w_n \\log p_n\\right)$$

where BP = brevity penalty, $p_n$ = modified n-gram precision.

## Implementation

\`\`\`python
from collections import Counter
import math

def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def bleu_score(candidate, references, max_n=4):
    """
    candidate: list of tokens
    references: list of list of tokens
    """
    # Brevity penalty
    c = len(candidate)
    r = min(len(ref) for ref in references)  # closest reference length
    if c > r:
        BP = 1
    else:
        BP = math.exp(1 - r / c) if c > 0 else 0
    
    # Modified n-gram precision
    precisions = []
    for n in range(1, max_n + 1):
        candidate_ngrams = Counter(get_ngrams(candidate, n))
        
        # Max count from any reference
        max_ref_counts = Counter()
        for ref in references:
            ref_ngrams = Counter(get_ngrams(ref, n))
            for ngram in ref_ngrams:
                max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
        
        # Clipped counts
        clipped_counts = {
            ngram: min(count, max_ref_counts[ngram])
            for ngram, count in candidate_ngrams.items()
        }
        
        precision = sum(clipped_counts.values()) / max(sum(candidate_ngrams.values()), 1)
        precisions.append(precision)
    
    # Geometric mean
    if any(p == 0 for p in precisions):
        return 0
    
    log_precision = sum(math.log(p) for p in precisions) / len(precisions)
    return BP * math.exp(log_precision)

# Example
candidate = ["the", "cat", "sat", "on", "mat"]
reference = [["the", "cat", "is", "on", "the", "mat"]]
print(bleu_score(candidate, reference))
\`\`\`

## Key Insight

BLEU punishes missing n-grams (precision) but not extra words unless they make the output shorter than reference (brevity penalty).
`;

nlpContent["CODE: Implement ROUGE Score"] = `
# ROUGE Score Implementation

## ROUGE vs BLEU

- **BLEU**: Precision-focused (for translation)
- **ROUGE**: Recall-focused (for summarization)

## ROUGE-N Formula

$$\\text{ROUGE-N} = \\frac{\\sum_{\\text{n-gram} \\in \\text{ref}} \\text{Count}_{\\text{match}}}{\\sum_{\\text{n-gram} \\in \\text{ref}} \\text{Count}}$$

## Implementation

\`\`\`python
from collections import Counter

def rouge_n(candidate, reference, n=1):
    """
    Compute ROUGE-N score
    """
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    cand_ngrams = Counter(get_ngrams(candidate, n))
    ref_ngrams = Counter(get_ngrams(reference, n))
    
    # Count overlapping n-grams
    overlap = sum((cand_ngrams & ref_ngrams).values())
    total_ref = sum(ref_ngrams.values())
    
    recall = overlap / total_ref if total_ref > 0 else 0
    
    total_cand = sum(cand_ngrams.values())
    precision = overlap / total_cand if total_cand > 0 else 0
    
    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    
    return {"precision": precision, "recall": recall, "f1": f1}

# ROUGE-1 example
candidate = ["the", "cat", "sat"]
reference = ["the", "cat", "is", "sitting"]
print(rouge_n(candidate, reference, n=1))
\`\`\`

## Key Insight

ROUGE-L uses longest common subsequence (LCS) to capture sentence-level structure, not just n-gram matches.
`;

nlpContent["CODE: Implement Perplexity Calculation"] = `
# Perplexity Calculation

## What is Perplexity?

Perplexity measures how "surprised" a language model is by test data. Lower = better.

$$\\text{PPL} = \\exp\\left(-\\frac{1}{N}\\sum_{i=1}^N \\log P(w_i | w_{<i})\\right)$$

## Intuition

Perplexity = weighted average branching factor. PPL=100 means the model is as uncertain as if choosing uniformly among 100 words.

## Implementation

\`\`\`python
import numpy as np

def perplexity(log_probs):
    """
    log_probs: list of log probabilities for each token
    """
    avg_neg_log_prob = -np.mean(log_probs)
    return np.exp(avg_neg_log_prob)

def compute_perplexity(model, tokens):
    """
    Compute perplexity of a sequence using a language model
    """
    log_probs = []
    for i in range(1, len(tokens)):
        context = tokens[:i]
        target = tokens[i]
        prob = model.probability(target, context)
        log_probs.append(np.log(prob + 1e-10))
    
    return perplexity(log_probs)

# Example with mock probabilities
log_probs = [-2.3, -1.5, -3.0, -2.0]  # log P(word|context)
ppl = perplexity(log_probs)
print(f"Perplexity: {ppl:.2f}")
\`\`\`

## Key Insight

GPT-2 achieves ~20-30 PPL on standard benchmarks. Human parity is ~10-15.
`;
nlpContent["CODE: Implement Multi-Head Attention"] = `
# Multi-Head Attention Implementation

## Why Multiple Heads?

Single attention can only focus on one aspect. Multiple heads let the model attend to different representation subspaces.

## The Math

$$\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h) W^O$$

where each head:
$$\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

## Implementation

\`\`\`python
import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def split_heads(self, x, batch_size):
        # x: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        
        Q = np.dot(q, self.W_q)
        K = np.dot(k, self.W_k)
        V = np.dot(v, self.W_v)
        
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # Scaled dot-product attention per head
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        attention = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention /= attention.sum(axis=-1, keepdims=True)
        
        context = np.matmul(attention, V)
        
        # Concatenate heads
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.num_heads * self.d_k)
        
        return np.dot(context, self.W_o)
\`\`\`

## Key Insight

8 heads with d_k=64 is equivalent to one head with d=512 in parameters, but captures 8 different attention patterns.
`;

nlpContent["CODE: Implement KV Cache for Inference"] = `
# KV Cache for Efficient Inference

## The Problem

In autoregressive generation, we recompute attention for all previous tokens at each step.

For length L: O(L²) per token → O(L³) total. Intractable for long sequences!

## The Solution: Cache K and V

Since previous tokens don't change, cache their K and V projections.

## Implementation

\`\`\`python
class KVCacheDecoder:
    def __init__(self, d_model, num_heads):
        self.d_k = d_model // num_heads
        self.k_cache = []  # List of (batch, seq_len, d_model)
        self.v_cache = []
        
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
    
    def forward(self, x, use_cache=True):
        # x: (batch, 1, d_model) - just the new token
        
        Q = np.dot(x, self.W_q)
        K_new = np.dot(x, self.W_k)
        V_new = np.dot(x, self.W_v)
        
        if use_cache and self.k_cache:
            # Append new K, V to cache
            K = np.concatenate([self.k_cache[-1], K_new], axis=1)
            V = np.concatenate([self.v_cache[-1], V_new], axis=1)
        else:
            K, V = K_new, V_new
        
        # Store in cache
        self.k_cache.append(K)
        self.v_cache.append(V)
        
        # Attention: Q attends to all cached K, V
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        attention = np.exp(scores - np.max(scores))
        attention /= attention.sum(axis=-1, keepdims=True)
        
        return np.matmul(attention, V)
    
    def reset_cache(self):
        self.k_cache = []
        self.v_cache = []
\`\`\`

## Complexity Improvement

- Without cache: O(L²) per token
- With cache: O(L) per token
- Total generation: O(L³) → O(L²)
`;

nlpContent["CODE: Implement RoPE (Rotary Position Embedding)"] = `
# Rotary Position Embedding (RoPE)

## Why RoPE?

Sinusoidal positional encodings are added to embeddings. RoPE instead rotates embeddings, which naturally encodes relative positions.

## The Core Idea

For position m, rotate the query/key vectors:

$$q_m = R_m q$$
$$k_n = R_n k$$

Then $q_m^T k_n$ depends only on relative position $(m-n)$.

## Implementation

\`\`\`python
import numpy as np

def get_rotary_matrix(seq_len, d_model):
    """Compute rotation matrices for each position"""
    position = np.arange(seq_len)[:, np.newaxis]
    dim = np.arange(0, d_model, 2)[np.newaxis, :]
    
    theta = position / (10000 ** (dim / d_model))
    
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    return cos, sin

def apply_rope(x, cos, sin):
    """Apply rotary embedding to x"""
    # x: (batch, seq_len, d_model)
    x1 = x[..., 0::2]  # Even dimensions
    x2 = x[..., 1::2]  # Odd dimensions
    
    # Rotate
    x_rotated = np.zeros_like(x)
    x_rotated[..., 0::2] = x1 * cos - x2 * sin
    x_rotated[..., 1::2] = x1 * sin + x2 * cos
    
    return x_rotated

# Usage
seq_len, d_model = 100, 64
cos, sin = get_rotary_matrix(seq_len, d_model)

Q = np.random.randn(2, seq_len, d_model)
K = np.random.randn(2, seq_len, d_model)

Q_rotated = apply_rope(Q, cos, sin)
K_rotated = apply_rope(K, cos, sin)
\`\`\`

## Key Insight

RoPE allows attention to naturally decay with distance and generalizes to longer sequences than seen in training.
`;

nlpContent["CODE: Implement Flash Attention Concept"] = `
# Flash Attention: Memory-Efficient Attention

## The Memory Problem

Standard attention materializes the full N×N attention matrix.

For N=100K tokens: 100K × 100K × 4 bytes = 40GB just for attention!

## Flash Attention Insight

Instead of computing full attention matrix, compute it in tiles that fit in fast SRAM.

## Simplified Concept

\`\`\`python
import numpy as np

def flash_attention_concept(Q, K, V, block_size=64):
    """
    Simplified Flash Attention showing tiling concept.
    Real implementation uses CUDA kernels.
    """
    seq_len = Q.shape[0]
    d_k = Q.shape[-1]
    
    # Output accumulator
    O = np.zeros_like(Q)
    
    # Running statistics for stable softmax
    m = np.full((seq_len,), -np.inf)  # max scores
    l = np.zeros((seq_len,))  # sum of exp(scores - max)
    
    # Process in blocks
    for j in range(0, seq_len, block_size):
        j_end = min(j + block_size, seq_len)
        
        K_block = K[j:j_end]
        V_block = V[j:j_end]
        
        # Compute scores for this block
        S_block = np.dot(Q, K_block.T) / np.sqrt(d_k)
        
        # Update running max
        m_new = np.maximum(m, S_block.max(axis=1))
        
        # Rescale previous sum
        l = l * np.exp(m - m_new)
        
        # Add new block contribution
        P_block = np.exp(S_block - m_new[:, np.newaxis])
        l += P_block.sum(axis=1)
        
        # Update output
        O = O * np.exp(m - m_new)[:, np.newaxis]
        O += np.dot(P_block, V_block)
        
        m = m_new
    
    # Final normalization
    return O / l[:, np.newaxis]
\`\`\`

## Key Insight

Flash Attention is O(N) memory instead of O(N²), enabling much longer context lengths.
`;

nlpContent["LLM Internals & Efficiency"] = nlpContent["Decoder Architecture"];

// New L5+ content aliases
nlpContent["GPTQ Quantization"] = nlpContent["LLM Quantization"];
nlpContent["AWQ Quantization"] = nlpContent["LLM Quantization"];
nlpContent["INT4 Quantization"] = nlpContent["LLM Quantization"];
nlpContent["Model Compression"] = nlpContent["LLM Quantization"];
nlpContent["HNSW Search"] = nlpContent["Embedding Retrieval"];
nlpContent["Product Quantization"] = nlpContent["Embedding Retrieval"];
nlpContent["Vector Search"] = nlpContent["Embedding Retrieval"];
nlpContent["Semantic Search"] = nlpContent["Embedding Retrieval"];
nlpContent["ANN Algorithms"] = nlpContent["Embedding Retrieval"];

// Decoding Strategies and Advanced NLP Problems aliases
nlpContent["CODE: Implement Greedy Decoding"] = nlpContent["Decoding Strategies"];
nlpContent["CODE: Implement Beam Search"] = nlpContent["Decoding Strategies"];
nlpContent["CODE: Implement Nucleus (Top-p) Sampling"] = nlpContent["Decoding Strategies"];
nlpContent["CODE: Implement Constrained Decoding"] = nlpContent["Decoding Strategies"];
nlpContent["CODE: Implement Span Extraction for QA"] = nlpContent["Advanced NLP Problems"];
nlpContent["CODE: Implement Sentence Similarity (SBERT)"] = nlpContent["Advanced NLP Problems"];
nlpContent["CODE: Implement Document Retrieval with BM25"] = nlpContent["Advanced NLP Problems"];

export default nlpContent;
