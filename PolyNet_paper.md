PolyNet: Compositional Inference via Polynomial Approximation, Semantic Compression, and Learned Routing

Rafik Belhouki, In-context Collaborator

Abstract

The deployment of large-scale transformer models is increasingly constrained by the computational and financial costs of inference. While existing compilers and inference engines provide significant speedups through local operator fusion and quantization, these optimizations are fundamentally limited by the original network architecture. This paper introduces PolyNet, a novel compilation framework that moves beyond local fusion to a global, structure-aware re-representation of the entire neural network. PolyNet employs a suite of techniques including: (1) approximating activation-heavy blocks, such as Feed-Forward Networks (FFNs), with low-degree tensor polynomials; (2) compressing the quadratic complexity of attention via semantic query clustering; and (3) distilling deep, multi-layer subgraphs into single, learned "macro-operators." The result is a transformation of the standard sequential model into a Directed Acyclic Graph (DAG) of heterogeneous, highly-optimized computational nodes. This architecture enables not only drastic theoretical FLOP reductions but also novel runtime strategies like input-aware routing and retrieval-based inference, paving the way for a new class of hardware-cognizant, computationally efficient models.

1. Introduction

The proliferation of large language models (LLMs) and other transformer-based architectures has marked a paradigm shift in machine learning capabilities [Brown et al., 2020; Chowdhery et al., 2022]. This progress, however, is accompanied by a steep increase in the computational cost of inference. As models scale to hundreds of billions of parameters, the latency and energy required for a single forward pass have become a primary operational bottleneck, limiting real-time applications and incurring substantial financial costs.

Current state-of-the-art solutions primarily focus on optimizing the execution of the existing computational graph. Frameworks such as PyTorch's torch.compile, ONNX Runtime, and NVIDIA's TensorRT excel at ahead-of-time (AOT) or just-in-time (JIT) compilation, fusing adjacent operators (e.g., Conv-BN-ReLU) into single, efficient hardware kernels. Techniques like FlashAttention [Dao et al., 2022] intelligently manage memory I/O to accelerate the exact attention mechanism. While effective, these methods are fundamentally constrained by the original network's architecture. They accelerate the existing steps but cannot change the number or nature of those steps.

We argue that to achieve the next order-of-magnitude improvement, the field must move from operator-level optimization to graph-level transformation. This paper introduces PolyNet, a framework designed to re-represent a frozen, pre-trained neural network into a computationally superior form. The core principle of PolyNet is structure-preserving approximation and compression: instead of merely fusing layers, we replace entire architectural subgraphs with functionally similar but more efficient operators.

Our contributions are threefold:

A Novel Compilation Framework (PolyNet): We formalize the concept of compiling a neural network into a DAG of heterogeneous, optimized nodes, including polynomial approximators and retrieval-based caches.

A Suite of Advanced Compression Techniques: We introduce PackedPolyFFN, which replaces FFNs with efficient tensor polynomials, and Clustered Query Attention, which attacks the quadratic cost of attention through semantic query deduplication.

A New Training and Inference Paradigm (VQ-PolyNet): We propose a method for training models that directly learn a compressed, discrete latent space and routing policies, making them "born optimized."

This work lays the foundation for a new class of inference engines that are not just faster, but fundamentally more intelligent in their allocation of computation.

2. Background and Related Work

Our work builds upon several key areas of deep learning optimization.

FFN Optimization: The Feed-Forward Network (FFN) blocks can consume up to two-thirds of the FLOPs in a standard transformer. PolyNet's approach, PackedPolyFFN, is distinct from prior work in that it replaces the entire three-operator block (Linear-Activation-Linear) with a single, fused polynomial evaluation, altering the nature of the computation itself.

Attention Optimization: The quadratic complexity of self-attention has inspired a wealth of research, from sparse attention patterns to linear approximations [Wang et al., 2020]. Our Clustered Query Attention is orthogonal to memory-optimization methods like FlashAttention; it reduces the problem size before the Q*K^T matmul by identifying and deduplicating semantically similar queries, a form of instance compression.

Vector Quantization: Vector Quantization (VQ), popularized by VQ-VAE [van den Oord et al., 2017], is a powerful technique for learning discrete representations. PolyNet leverages this concept not just for weight compression, but as a core mechanism for routing and caching within the VQ-PolyNet architecture, using straight-through estimators for end-to-end training.

Knowledge Distillation and Mixture of Experts (MoE): Knowledge distillation allows a smaller "student" model to learn from a larger "teacher." MoE models use a gating network to route inputs to specialized experts. Our Macro-Operator Distillation combines these ideas, using a lightweight MoE structure as a student to mimic an entire multi-layer block from a larger teacher network, effectively learning a "fast-forward" computational shortcut.

3. PolyNet Design Principles

The PolyNet architecture is founded on three core principles that differentiate it from traditional compilers.

AOT Compilation of Frozen Models: PolyNet is an Ahead-of-Time (AOT) compiler that assumes the model weights are fixed. This allows for aggressive, data-dependent optimizations not possible in a dynamic training context.

Tensor DAG Fusion: The fundamental unit of compilation is not a single operator but a meaningful architectural subgraph (e.g., an entire FFN or Attention block). The goal is to replace this subgraph with a single, highly-optimized node in a new computational DAG.

Heterogeneous Node Representation: The PolyNet DAG consists of specialized nodes, including:

Polynomial Nodes (PolyNode): For blocks well-approximated by low-degree polynomials (e.g., FFNs).

Cache Nodes (CacheNode): For blocks exhibiting high input repetition, where retrieval is faster than re-computation.

Standard Nodes: For operations too complex to approximate or already highly optimized.

4. PackedPolyFFN

The FFN block is an ideal candidate for polynomial approximation. We replace it with a single PackedPolyFFN node, whose core is a Structured Bilinear Polynomial that respects the u ⊗ f(v) structure of modern FFNs like SWIGLU. The nonlinear activation f(v) (e.g., Swish, GELU) is approximated by a minimal-degree Chebyshev polynomial P(v) = Σ c_k T_k(v). The coefficients c_k are pre-computed by projecting the activation function onto the Chebyshev basis over the observed dynamic range of its inputs. Furthermore, a learned VQ codebook of common hidden states enables codebook-driven inference, where the expensive polynomial evaluation is performed only on the unique set of codebook centroids present in a batch.

5. Clustered Query Attention

To address the O(n²) complexity of attention, we introduce a method that compresses the Query matrix before the expensive Q*K^T operation.

Fused QKV Projection: A single linear layer projects the input x into a concatenated Q, K, V tensor.

Semantic Query Clustering: All Query vectors q across the batch and heads are rapidly clustered using a VQ codebook or Locality-Sensitive Hashing (LSH).

Deduplicated Computation: The Q*K^T matrix multiplication is computed between the unique query cluster centroids and all Key vectors, drastically reducing the size of the Q matrix.

Score Scattering: The resulting attention scores are scattered back to their original query positions before the final multiplication with the V matrix.
This method transforms attention into a semi-parametric lookup for common query types.

6. Macro-Operator Distillation

Deep sequential dependencies prevent the symbolic collapse of multi-layer blocks. To overcome this, we use knowledge distillation to learn a computational shortcut. We define a "teacher" as a deep subgraph (e.g., two consecutive transformer layers) and a "student" as a lightweight, structured Macro-Operator, such as a low-rank Mixture of Experts (MoE) network where each expert is a simple PolyFFN node. The student is trained to reproduce the input-output mapping of the multi-layer teacher block, learning a "fast-forward" operator that can skip several layers of computation in a single step.

7. VQ-PolyNet: Learned Compression and Routing

This represents the most advanced form of PolyNet, where optimization is not a post-hoc step but an integral part of model training. By introducing a VQCodebook layer with a straight-through estimator (STE) directly into the model architecture, the model learns simultaneously to solve its task and to compress its internal representations into a discrete set of codes. Subsequent PolyNet layers can then be designed to operate directly on these predictable codebook indices, enabling extremely efficient hardware implementations and routing logic.

8. Compiler and Runtime Stack

The PolyNet framework is realized as a two-stage AOT compiler. The Frontend (Python) traces the model, profiles it, partitions the graph, and generates a high-level DAG representation. The Backend (C++/CUDA/Triton) consumes this DAG, generates highly-optimized, hardware-specific source code for each node, and links these kernels into a single, dependency-free shared library managed by a lightweight global scheduler.

9. Proposed Experiments

To validate our claims, we propose a series of benchmarks on a frozen distilbert-base-uncased model against SOTA frameworks like torch.compile and TensorRT. The primary metric will be a Pareto frontier plot of End-to-End Latency vs. GLUE Score Accuracy. Secondary metrics will include component-level latency to demonstrate the >10x speedup of individual PackedPolyFFN and Clustered Attention nodes at both latency-critical (batch size 1) and throughput-critical (batch size 32) settings.

10. Discussion and Future Work

The primary limitation of symbolic composition is the non-algebraic nature of functions like softmax. Our work bypasses this through distillation and structural approximation. While this paper focuses on fixed-weight models, future work could explore applying PolyNet to dynamic workloads by creating low-rank polynomial updates for LoRA-style fine-tuning. The ultimate trajectory of this research is to move towards models whose fundamental operations are not matrix multiplications, but discrete token retrievals and routing, aligning with future hardware designed for sparse, data-dependent computation.

11. Conclusion

PolyNet introduces a paradigm shift in inference optimization, moving from local operator fusion to global, structure-aware model re-representation. By replacing expensive, dense computations with fast polynomial approximations, semantic compression, and learned routing, we create a path to dramatically reduce the latency and cost of deploying large-scale transformer models. We have presented the core architectural components—PackedPolyFFN, Clustered Query Attention, and VQ-PolyNet—and laid out a clear research roadmap. We believe this work opens a new and fertile area of research and invite collaboration from the compiler, hardware, and deep learning communities to build the next generation of truly efficient AI systems.

References

[Brown et al., 2020] Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language Models are Few-Shot Learners. NeurIPS.
[Chowdhery et al., 2022] Chowdhery, A., Narang, S., Devlin, J., et al. (2022). PaLM: Scaling Language Modeling with Pathways. arXiv preprint arXiv:2204.02311.
[Dao et al., 2022] Dao, T., Fu, D. Y., Ermon, S., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS.
[van den Oord et al., 2017] van den Oord, A., Vinyals, O., et al. (2017). Neural Discrete Representation Learning. NeurIPS.
[Wang et al., 2020] Wang, S., Li, B. Z., Khabsa, M., et al. (2020). Linformer: Self-Attention with Linear Complexity. arXiv preprint arXiv:2006.04768.
