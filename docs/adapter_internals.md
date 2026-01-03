# Adapter Internals

This note explains how adapter-style fine-tuning is wired into the code base for both **LoRA/QLoRA** and the **butterfly rotation** experiments. It focuses on (1) where the additional parameters are inserted and (2) what exactly gets trained versus frozen in each mode.

## LoRA and QLoRA

The core logic lives in `src/train.py` inside `build_model` (see `src/train.py:95-142`). The flow is:

1. Load the base decoder-only model via `AutoModelForCausalLM.from_pretrained`.
2. If `--mode qlora`, run `prepare_model_for_kbit_training` so layer norms and gradients are configured for 4-bit weights.
3. Construct a `LoraConfig` with the CLI flags (`--lora_r`, `--lora_alpha`, `--lora_dropout`, and `--target_modules`).
4. Call `get_peft_model(model, lora_config)`, which walks through every module whose name matches any entry in `target_modules` (default `["c_attn", "c_proj"]`) and injects learnable low-rank adapters.

### Where the LoRA matrices live

PEFT replaces each matched projection (e.g., GPT-2’s `c_attn` key/query/value projection) with a wrapper that keeps the frozen weight `W` and augments it with a low rank update:

```
output = x @ (W + scale * (B @ A))
```

- `A` has shape `[in_dim, r]` and `B` has `[r, out_dim]` (with `r=--lora_r`).
- The scaling factor is `--lora_alpha / r`, matching the original paper.
- Because PEFT handles `Conv1D` and `Linear` uniformly, the only requirement is the module name filter.
- When `--merge_lora` is provided at save time, the adapters are merged back into the frozen weights.

For QLoRA the same wrappers are added, but the base weights remain quantized during training while the LoRA matrices stay in full precision. This is why `prepare_model_for_kbit_training` must run before `get_peft_model`.

## Butterfly rotation adapters

The butterfly path is selected with `--mode butterfly`. Instead of PEFT, `build_model` calls `apply_butterfly_rotation` (see `src/butterfly_rotation_adapter.py:11-114`) which recursively scans the module hierarchy and swaps any matched projection for a rotation-aware wrapper.

### Rotation module

`ButterflyRotation` (`src/butterfly_rotation_adapter.py:25-66`) stores a stack of learnable angles, one per pair of coordinates per stage. For a given input dimension `d`:

- Stage `s` mixes entries separated by `2^s` positions. `_stage_pairs` precomputes all `(i, j)` pairs so the implementation works for arbitrary `d`, not just powers of two.
- During `forward`, the tensor is flattened to `[batch*dims, d]`, the relevant columns are gathered, and a 2×2 rotation is applied using `cos(θ)` and `sin(θ)` per pair.
- Only the angle tensors are trainable. The base projection weights remain untouched and frozen.

### Linear vs Conv1D wrappers

Two thin adapters wrap the actual projections:

1. `ButterflyLinearRotationAdapter` (`src/butterfly_rotation_adapter.py:69-80`) holds a frozen `nn.Linear` plus a `ButterflyRotation` that acts on its input features. Forward path = reshape input → rotate → reshape back → apply original linear layer.
2. `ButterflyConv1DRotationAdapter` (`src/butterfly_rotation_adapter.py:83-98`) does the same for GPT-style `Conv1D` projections (used for `c_attn` and `c_proj`). The wrapper inspects `conv.weight.size(0)` to know the input width, rotates activations, then calls the frozen Conv1D.

`apply_butterfly_rotation` (`src/butterfly_rotation_adapter.py:101-114`) ensures each named child module is replaced with the appropriate wrapper. Immediately after insertion, `build_model` (in `src/train.py:130-141`) sets `requires_grad=False` for all parameters, then flips it back to `True` only for modules that are instances of `ButterflyRotation`. This keeps the optimizer focused on the rotation angles (~O(d log d) parameters) while every base projection stays frozen.

### Parameter intuition

- With `butterfly_stages = 6` on GPT-2’s 768-wide projections, each stage introduces ~768 trainable angles (slightly fewer near the tail), so the total trainable count is roughly `stages * d`. This is why the CLI prints ~1e5 trainable parameters compared to 8e7 total.
- Increasing stages deepens the rotation (more expressivity) but adds parameters linearly and extra trig ops per forward pass.

### Blockwise butterfly math

When `--mode butterfly_block` is active, the feature dimension \(d\) is partitioned into contiguous blocks of size \(b\) (default 256). Let \(x \in \mathbb{R}^d\) and write
\[
x = \begin{bmatrix} x^{(0)} \\ x^{(1)} \\ \vdots \\ x^{(m-1)} \end{bmatrix}, \quad x^{(k)} \in \mathbb{R}^{b_k},
\]
where \(b_k = \min(b, d - k b)\). Each block receives its own butterfly rotation \(R^{(k)} \in \mathbb{R}^{b_k \times b_k}\), so the full transform is block diagonal:
\[
R = \mathrm{diag}(R^{(0)}, R^{(1)}, \dots, R^{(m-1)}).
\]

#### Stage structure

Within a block of length \(L\), stage \(s\) (zero-indexed) pairs indices
\[
\mathcal{P}_s = \left\{ (i,\, j) \,\big|\, i = \ell + r,\; j = i + 2^s,\; \ell \in \{0, 2^{s+1}, 2 \cdot 2^{s+1}, \dots\},\; 0 \le r < 2^s,\; j < L \right\}.
\]
Each pair \((i, j)\) has a learnable angle \(\theta_{s,(i,j)}\). Applying the stage multiplies the vector by the block-diagonal matrix whose 2×2 blocks are
\[
Q(\theta) =
\begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix}.
\]
Orthogonality holds because \(Q(\theta)^\top Q(\theta) = I_2\), so the stage matrix \(R_s\) satisfies \(R_s^\top R_s = I\). The full block rotation is \(R^{(k)} = R_{S-1} \cdots R_1 R_0\), a product of orthogonal matrices, hence itself orthogonal.

#### Cross-block intuition

Even though \(R\) is block diagonal, the frozen projection \(W\) that follows is dense. The effective weight becomes \(W R\), so channels from different blocks still interact through \(W\); the blockwise butterfly simply re-bases each subspace independently while keeping log-depth pairing and neat power-of-two strides within each block.

## Practical tips

- Use `--target_modules` to widen or narrow the set of projections receiving adapters. For LoRA, PEFT matches any module whose name is equal to or ends with the token. For butterfly, the same name matching happens inside `apply_butterfly_rotation`.
- For quick verification, run `python -m src.train ... --max_train_samples 512 --max_eval_samples 64` to make sure adapters are inserted correctly before a long training job.
- Keep an eye on the `Parameters total=... trainable=...` log line; it lets you confirm that LoRA (~1–2M trainables) or butterfly (~1e5 trainables) are behaving as expected.
