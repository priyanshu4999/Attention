# Understanding Attention Models and Computational Complexity

## Questions and Discussions
### 1. How do we start with word representation?
We start with a sentence containing 10 lexicographical root words. Each word is converted into an embedding vector with dimension $d = 512$.

### 2. How do we add positional information to embeddings?
Since word embeddings alone do not capture positional information, positional embeddings are added. This can be done using:
- **Sinusoidal functions** (used in Transformers)
- **Learnable position embeddings**

### 3. Is the initial embedding dimension $d$ a unit vector?
The embedding dimension $d$ represents a **vector in high-dimensional space**, but it is not necessarily a unit vector. It can have any magnitude based on the learned representation.

### 4. What are common embedding methods?
- **Word2Vec (CBOW, Skip-gram)**
- **GloVe (Global Co-occurrence Matrix Factorization)**
- **Transformer-based embeddings (BERT, GPT)**

### 5. How do we map words to other words?
Mapping a word to another word involves different methods, including:
1. **Linear transformation** $Y = XW$
2. **Nearest neighbor search** (e.g., cosine similarity in vector space models)
3. **Self-attention mechanism** in Transformers

### 6. Is this mapping step computationally expensive?
Yes, depending on the method used. Self-attention and nearest neighbor searches can be computationally expensive, especially for long sequences or large vocabularies.

## Computational Complexity of Word Mapping Methods

### Linear Transformation (Fast)
A simple linear transformation maps word embeddings as:
```math
Y = X W
```
where:
- $X \in \mathbb{R}^{n \times d}$: Word embeddings of the sentence.
- $W \in \mathbb{R}^{d \times d}$: Learnable mapping matrix.
- $Y \in \mathbb{R}^{n \times d}$: Transformed word embeddings.

The computational complexity is:
```math
O(n d^2)
```
This is efficient with GPUs and TPUs.

### Nearest Neighbor Search (Expensive for Large Vocabularies)
Each word embedding $X_i$ is mapped to the closest word in the vocabulary using cosine similarity:
```math
\text{similarity}(X_i, E_j) = \frac{X_i \cdot E_j}{\|X_i\| \|E_j\|}
```
where $E_j$ represents all vocabulary embeddings.

The computational complexity is:
```math
O(n V d)
```
For large $V$ (e.g., 50,000 words), this is expensive. Optimizations include:
- **Approximate Nearest Neighbor (ANN)**: $O(n \log V)$ using FAISS or HNSW.
- **Dimensionality Reduction (PCA, SVD)** to reduce $d$.

### Self-Attention (Expensive for Long Sequences)
Self-attention dynamically maps words based on:
```math
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
```
The computational complexity is:
```math
O(n^2 d)
```
which scales poorly for long sequences. Optimizations include:
- **Sparse Attention (Longformer, BigBird)**: $O(n d \log n)$.
- **Efficient Transformers (Linformer, Performer)**: Lower-rank approximations.

## Comparison of Methods

| **Method**                  | **Complexity**    | **Bottleneck**              |
|-----------------------------|------------------|----------------------------|
| Linear Transformation       | $O(n d^2)$      | Matrix multiplication      |
| Nearest Neighbor Search     | $O(n V d)$      | Large vocabulary search    |
| Self-Attention              | $O(n^2 d)$      | Attention computation      |

## Conclusion
For short sentences ($n < 50$), nearest neighbor search is expensive if $V$ is large. For long sequences ($n > 1000$), self-attention is the main bottleneck due to $O(n^2 d)$ complexity.
