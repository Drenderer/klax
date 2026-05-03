---
title: Multi-layer perceptrons
---

::: klax.nn.MLP
    options:
        members:
            - __init__
            - __call__

---

# Input Convex Neural Networks

Input Convex Neural Networks (ICNNs) are a family of neural network architectures introduced by
[Amos et al. (2017)](https://arxiv.org/abs/1609.07152) that are designed to produce outputs that
are provably convex with respect to (some of) their inputs. Klax provides two variants:
the **Fully Input Convex Neural Network** (`FICNN`) and the
**Partially Input Convex Neural Network** (`PICNN`).

## Fully Input Convex Neural Network (FICNN)

An `FICNN` represents a function $f : x \mapsto y$ where every element of the output $y$ is a
convex function of the input $x$.

### Layer structure

![FICNN Layer](../../assets/icnn_ficnn_layer.png#only-light)
![FICNN Layer](../../assets/icnn_ficnn_layer_dark.png#only-dark)

Each `FICNNLayer` computes:

$$y_{i+1} = \sigma\!\left(U_i y_i + W_i x + b_i\right)$$

where:

- $y_i$ is the hidden state from the previous layer (initialised as $y^0 = x$),
- $x$ is the original network input, passed directly to every layer via the *passthrough connection*,
- $U_i$, $W_i$, $_i$ are the weight matrices and bias of layer $i$,
- $\sigma$ is an activation function that must be **convex and non-decreasing** (default: `softplus`).

### Convexity guarantee

Convexity is maintained by enforcing two constraints:

1. **Non-negative weights on the convex path.** The weight matrix $U_i$ acting on the
   previous hidden state $y_i$ is constrained to be element-wise non-negative (via
   [`NonNegative`][klax.NonNegative]) for all layers except the first. This ensures that the
   composition of affine maps and a convex activation remains convex.
2. **Convex, non-decreasing activation.** The activation $\sigma$ applied to each layer's
   pre-activation must itself be convex and non-decreasing.

The passthrough weight $W_i$ has no sign constraint because $x$ enters each layer as a
fixed affine term, which does not break convexity.

### Optional: non-decreasing output

It is possible to additionally constrain $U_0$ (the first-layer weight) and all
passthrough weights $W_i$ to be non-negative. This ensures the output is not only convex but
also element-wise **non-decreasing** in $x$. This is needed, for example, when the FICNN is
composed with another convex function $x(z)$, and convexity in $z$ must be preserved through
the chain rule.

### Usage

::: klax.nn.FICNN
    options:
        members:
            - __init__
            - __call__

---

## Partially Input Convex Neural Network (PICNN)

A `PICNN` represents a function $f : (x, p) \mapsto y$ where every element of the output $y$ is
a convex function of $x$, but can have an **arbitrary** relationship to the conditioning input $p$.
Intuitively, a PICNN is a FICNN whose weights and biases are themselves functions of $p$.

### Layer structure

![FICNN Layer](../../assets/icnn_picnn_layer.png#only-light)
![FICNN Layer](../../assets/icnn_picnn_layer_dark.png#only-dark)

Each `PICNNLayer` maintains two parallel hidden states:

- **Convex path** $y_i$: carries information about $x$ and $p$; convexity in $x$ is enforced here.
- **Arbitrary path** $u_i$: an unconstrained MLP processing the input $p$ with no convexity constraints applied.

The two paths interact via **interconnection sublayers** that modulate the weights of the convex
path using the current state $u_i$ of the arbitrary path:

$$
\begin{align}
    y^{i+1} &= \sigma^y\Bigl(U_i(u_i) y_i \;+W_i(u_i) x \;+b_i(u_i)\Bigr)\\
    u^{i+1} &= \sigma^u\!\left(W_i^u\, u_i + b_i^u\right)
\end{align}
$$

where:

- the weight $U_i(u_i) = \tilde U_i \cdot \texttt{diag}(\sigma^{yu}(W^{yu}u_i+b^{yu}))$ is computed from a constant weight matrix $\tilde U_i$, that is column-wise modulated by $u_i$ via a single layer MLP,
- the weight $W(u_i) = \tilde W_i \cdot \texttt{diag}(\sigma^{xu}(W^{xu}u_i+b^{xu}))$ is similarly computed from a constant weight matrix $\tilde W_i$, that is column-wise modulated by $u_i$ via a single layer MLP (only when `use_passthrough=True`),
- the bias $b(u_i) = W^{bu}u_i+b^{bu}$ is provided through an additional affine projection of $u_i$.
- $\sigma_{yu}$ must be **non-negative** (default: `softplus`) to preserve convexity,
- $\sigma_{xu}$ has no sign constraint by default (identity), but must be non-negative when
  `non_decreasing=True`.

Note that information from $x$ **never** enters the arbitrary path $u$, preserving the
convexity guarantee in $x$.

### Convexity guarantee

The convexity argument mirrors the FICNN:

1. **Non-negative weights on the convex path.** The weight matrix $U_i(u_i)$ acting on the
   previous hidden state $y_i$ is constrained to be element-wise non-negative for all layers except the first. For this, $\tilde U_i$ is constrained to be non-negative (via [`NonNegative`][klax.NonNegative]). Additionally, $\sigma^{yu}$ *must* be non-negative (e.g. `softplus`), ensuring the constraint is maintained regardless of the value of $u_i$. This ensures that the composition of affine maps and a convex activation remains convex.
2. **Convex, non-decreasing activation.** The activation $\sigma_y$ must be convex and non-decreasing.

### Optional: non-decreasing output

Analogously to the FICNN case, it's possible to ensure the output $y$ is non-decreasing in $x$ by extending the non-negativity constraints to the first-layer weight and to all passthrough weights (setting `non_decreasing=True`).

### Usage

::: klax.nn.PICNN
    options:
        members:
            - __init__
            - __call__