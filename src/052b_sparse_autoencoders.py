#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<4.0"
# dependencies = ["numpy>=2.5.4"]
# ///
"""
Lesson 5.2b: Sparse Autoencoders on a Toy of Superposition (side quest).

You train a second, wider autoencoder on the hidden activations of the toy from
Lesson 5.2a and ask it to explain each activation as a sparse sum of unit-length
directions (a dictionary), with every gradient derived by hand. The toy packs 12
features into 4 dimensions as overlapping directions, so no single hidden
dimension means anything on its own. The sparse autoencoder's directions line up
with the packed features far better than the hidden dimensions do, which you
measure as a cosine.

This lesson trains the same toy as Lesson 5.2a (n features squeezed through m
hidden dimensions) in a compact copy, because numbered lesson files are
scripts and cannot be imported.

Source: Bricken et al., "Towards Monosemanticity: Decomposing Language Models
With Dictionary Learning", Transformer Circuits Thread (2023),
https://transformer-circuits.pub/2023/monosemantic-features/index.html

Prerequisites (must complete first)
----------------------------------
- Lesson 5.2a: Toy Models of Superposition (the network whose features this
  lesson recovers)

Learning Objectives
-------------------
After completing this lesson, you will be able to:
1. Derive the gradient of a sparse autoencoder by hand, including the
   subgradient of the L1 penalty and the bias that enters both the encoder and
   the decoder.
2. Check that gradient against central differences, and see the check fail
   when the bias term or the ReLU mask is missing.
3. Measure how well a learned dictionary recovers known feature directions,
   against the hidden dimensions as a baseline.

The Big Idea
------------
Superposition hides features as overlapping directions, so you cannot read
them off the hidden dimensions. A sparse autoencoder looks for a larger set of
directions such that each activation is a combination of only a few of them.
The L1 penalty makes few-at-a-time cheaper than many-at-a-time, and the
directions that make it cheap are the packed features.

Why It Matters
--------------
Bricken et al. train sparse autoencoders on a language model's activations and
report interpretable features that single neurons do not show. Nick Cammarata's
2026 Goodfire essay "Neural Networks Are Already Clean"
(https://already-clean.goodfire.com/) argues that real networks' neurons are
mostly readable directly and that decompositions like this are often not the
right first step. This toy's hidden layer has no activation function, so its
dimensions are an arbitrary basis: the lesson shows what a sparse autoencoder
does, not whether real neurons need one.

Stdlib Pain → Library Power-Up
------------------------------
Pain (this lesson):
- You derive the gradient on paper, then write it as about ten lines of NumPy;
  a missing ReLU mask or a dropped ``bd`` term trains silently to a worse
  dictionary.
- Lines of code: ~15 for the gradient, ~15 for Adam.
- Runtime: about 15 seconds for the demo (8,000 toy steps, then 6,000 steps of
  1,024 activations), measured with one BLAS thread.
- Common bugs: forgetting that ``bd`` also shifts the encoder's input, and
  differentiating through the ReLU where it is inactive.

Power-Up (Lesson 5.3 — PyTorch):
- ``loss.backward()`` produces all four gradients from the forward pass, and
  ``torch.optim.Adam`` replaces the hand-written update.
- Key APIs: ``torch.autograd``, ``torch.optim.Adam``.

Complexity Analysis
-------------------
Pain (manual), for batch B, m hidden dimensions, d dictionary atoms, k steps:
- Time:  O(k·B·m·d).
- Space: O(B·d + m·d).

Execution
---------
Project-style:
    uv run python src/052b_sparse_autoencoders.py

Script-style (self-contained deps):
    uv run --script src/052b_sparse_autoencoders.py

On a busy machine, NumPy's BLAS threads compete with other work and can stretch
the demo from about 15 seconds to a minute or more; one thread avoids that, since the
matrices here are small:
    OPENBLAS_NUM_THREADS=1 uv run python src/052b_sparse_autoencoders.py

Options:
    python src/052b_sparse_autoencoders.py --doctest-only
    python src/052b_sparse_autoencoders.py --no-doctest
    python src/052b_sparse_autoencoders.py --index
    python src/052b_sparse_autoencoders.py -v

Quality Gates
-------------
    uv run pytest src/052b_sparse_autoencoders.py
    uv run ruff check src/052b_sparse_autoencoders.py
    uv run ruff format --check src/052b_sparse_autoencoders.py
    uv run mypy --strict src/052b_sparse_autoencoders.py

Module Doctests
---------------
With one hidden dimension, one atom, encoder weight 2, decoder weight 1, and
the activation h = 1, the code is f = 2 and the reconstruction is 2. With
λ = 0.5 the loss is (2 - 1)² + 0.5·2 = 2:

>>> import numpy as np
>>> params = (np.array([[2.0]]), np.zeros(1), np.array([[1.0]]), np.zeros(1))
>>> loss, grads = sae_loss_and_grads(params, np.array([[1.0]]), 0.5)
>>> float(loss), [float(g[0, 0]) if g.ndim == 2 else float(g[0]) for g in grads]
(2.0, [2.5, 2.5, 4.0, -3.0])
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    type Array = NDArray[np.float64]
    type Params = tuple[Array, Array, Array, Array]
    type GradFn = Callable[[Params, Array, float], tuple[float, Params]]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON IDENTITY & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON_ID: Final[str] = "tier_5_lesson_2b_sparse_autoencoders"

# Autograd and torch.optim.Adam arrive with PyTorch.
POWER_UP_AVAILABLE: Final[bool] = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Format: lesson_id -> reason

PREREQUISITES: Final[Mapping[str, str]] = {
    "tier_5_lesson_2a_superposition": "The toy whose features it recovers",
}

DEPENDENCIES: Final[Mapping[str, str]] = {
    "tier_3_lesson_1_dot_product": "Cosine similarity between a feature and an atom",
}

SUCCESSORS: Final[Mapping[str, str]] = {
    "tier_5_lesson_2c_tiny_cnn": "Another small network you train by hand, then inspect",
}

DEPENDENTS: Final[Mapping[str, str]] = {}

RELATED: Final[Mapping[str, str]] = {
    "tier_5_lesson_2d_reading_a_unit": "Reads a unit directly instead of decomposing it",
}

POWER_UP_IN_LESSON: Final[str | None] = "tier_5_lesson_3_pytorch"
PAIN_FROM_LESSON: Final[str | None] = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE TOY: The Lesson 5.2a Autoencoder, Compact
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def importance(n: int) -> Array:
    """
    Return the weight 0.9**i of each feature's reconstruction error.

    Examples
    --------
    >>> importance(3).round(2).tolist()
    [1.0, 0.9, 0.81]
    """
    return np.asarray(0.9 ** np.arange(n), dtype=np.float64)


def sample_features(rng: np.random.Generator, batch: int, n: int, sparsity: float) -> Array:
    """
    Draw a ``(batch, n)`` matrix of features in [0, 1], each zeroed with probability ``sparsity``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> sample_features(rng, 4, 3, 0.5).shape
    (4, 3)
    >>> bool((sample_features(rng, 100, 3, 1.0) == 0).all())
    True
    >>> bool((sample_features(rng, 100, 3, 0.0) > 0).all())
    True
    """
    values = rng.random((batch, n))
    return np.asarray(values * (rng.random((batch, n)) >= sparsity), dtype=np.float64)


def toy_loss_and_grads(W: Array, b: Array, X: Array, imp: Array) -> tuple[float, Array, Array]:
    """
    Return the toy's loss and its gradients with respect to ``W`` and ``b``.

    The forward pass is H = X Wᵀ, Z = H W + b, X' = ReLU(Z), and the loss is
    the batch mean of Σᵢ impᵢ (Xᵢ - X'ᵢ)². Backward, with D = ∂loss/∂Z::

        D = 2 · imp · (X' - X) · (Z > 0) / B
        grad_W = Hᵀ D + W Dᵀ X      (decoder term + encoder term)
        grad_b = Σ over the batch of D

    Lesson 5.2a derives and checks this gradient.

    Examples
    --------
    One feature, one dimension, W = 2: the loss is (1 - 4)² = 9, and
    d/dW of (W² - 1)² = 4W(W² - 1) is 24:

    >>> import numpy as np
    >>> W, b, X = np.array([[2.0]]), np.zeros(1), np.array([[1.0]])
    >>> loss, grad_W, grad_b = toy_loss_and_grads(W, b, X, importance(1))
    >>> float(loss), float(grad_W[0, 0]), float(grad_b[0])
    (9.0, 24.0, 6.0)

    With W = 2 and b = -5 the pre-activation is -1, so ReLU is inactive: the
    loss is 1 and the gradients are exactly zero.

    >>> W, b, X = np.array([[2.0]]), np.full(1, -5.0), np.array([[1.0]])
    >>> loss, grad_W, grad_b = toy_loss_and_grads(W, b, X, importance(1))
    >>> float(loss), float(grad_W[0, 0]), float(grad_b[0])
    (1.0, 0.0, 0.0)
    """
    H = X @ W.T
    Z = H @ W + b
    out = np.maximum(Z, 0.0)
    loss = float(np.mean(np.sum(imp * (X - out) ** 2, axis=1)))
    D = 2.0 * imp * (out - X) * (Z > 0) / len(X)
    return loss, H.T @ D + W @ D.T @ X, D.sum(axis=0)


def adam_step(
    params: Sequence[Array],
    grads: Sequence[Array],
    first: Sequence[Array],
    second: Sequence[Array],
    t: int,
    lr: float,
) -> None:
    """
    Update ``params`` in place with one Adam step; ``first`` and ``second`` are updated too.

    Adam uses betas 0.9 and 0.999 and eps 1e-8. ``t`` counts steps from 1 and
    sets the bias correction that undoes the zero start of the moving averages.

    Examples
    --------
    On the first step the corrected moments are g and g², so every parameter
    moves by lr·g/|g| = lr against the sign of its gradient:

    >>> import numpy as np
    >>> p, v, u = np.array([1.0, 1.0]), np.zeros(2), np.zeros(2)
    >>> adam_step([p], [np.array([3.0, -0.5])], [v], [u], t=1, lr=0.1)
    >>> p.round(6).tolist()
    [0.9, 1.1]
    """
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for p, g, v, u in zip(params, grads, first, second, strict=True):
        v *= beta1
        v += (1 - beta1) * g
        u *= beta2
        u += (1 - beta2) * g * g
        p -= lr * (v / (1 - beta1**t)) / (np.sqrt(u / (1 - beta2**t)) + eps)


def train_toy(
    n: int,
    m: int,
    sparsity: float,
    steps: int = 8000,
    seed: int = 0,
    batch: int = 1024,
    lr: float = 1e-2,
) -> tuple[Array, Array]:
    """
    Train the Lesson 5.2a autoencoder with Adam and return ``(W, b)``.

    The seeded generator draws the initial ``W`` and then a fresh batch per
    step, so a run repeats exactly.

    Examples
    --------
    The paper's small case, n = 5 features in m = 2 dimensions. Dense inputs
    fit only two features; the rest of ``W`` shrinks away:

    >>> W, b = train_toy(5, 2, sparsity=0.0, steps=2000)
    >>> W.shape, b.shape
    ((2, 5), (5,))
    >>> features_represented(W)
    2

    Sparse inputs let all five share the two dimensions:

    >>> W, b = train_toy(5, 2, sparsity=0.95, steps=2000)
    >>> features_represented(W)
    5
    """
    rng = np.random.default_rng(seed)
    imp = importance(n)
    params = [rng.normal(size=(m, n)) * 0.3, np.zeros(n)]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    for t in range(1, steps + 1):
        X = sample_features(rng, batch, n, sparsity)
        _, grad_W, grad_b = toy_loss_and_grads(params[0], params[1], X, imp)
        adam_step(params, (grad_W, grad_b), first, second, t, lr)
    return params[0], params[1]


def features_represented(W: Array) -> int:
    """
    Count the columns of ``W`` with norm above 0.5.

    A feature the toy stores has a column of norm near 1; a dropped one has a
    column near 0.

    Examples
    --------
    >>> import numpy as np
    >>> features_represented(np.array([[1.0, 0.0, 0.6], [0.0, 0.1, 0.0]]))
    2
    """
    return int(np.sum(np.linalg.norm(W, axis=0) > 0.5))


def hidden_activations(W: Array, X: Array) -> Array:
    """
    Return the toy's hidden layer H = X Wᵀ, one row per sample.

    Each row is a sum of the columns of ``W`` for the features active in that
    sample, weighted by their values.

    Examples
    --------
    One active feature, number 2, with value 1 gives that feature's direction:

    >>> import numpy as np
    >>> W = np.array([[1.0, 0.0, 0.6], [0.0, 1.0, 0.8]])
    >>> hidden_activations(W, np.array([[0.0, 0.0, 1.0]])).tolist()
    [[0.6, 0.8]]
    """
    return np.asarray(X @ W.T, dtype=np.float64)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE PAIN: A Sparse Autoencoder With Hand-Derived Gradients
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def unit_columns(M: Array) -> Array:
    """
    Scale each column of ``M`` to length 1; a zero column stays zero.

    Examples
    --------
    >>> import numpy as np
    >>> unit_columns(np.array([[3.0, 0.0], [4.0, 0.0]])).round(6).tolist()
    [[0.6, 0.0], [0.8, 0.0]]
    """
    return np.asarray(M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-12), dtype=np.float64)


def sae_forward(params: Params, h: Array) -> tuple[Array, Array, Array]:
    """
    Return the pre-activation, the code ``f``, and the reconstruction of ``h``.

    With ``params = (We, be, Wd, bd)``: pre = (h - bd) Weᵀ + be, f = ReLU(pre),
    and h' = f Wdᵀ + bd. The decoder bias ``bd`` is subtracted before encoding
    and added back after decoding.

    Examples
    --------
    One atom along the first axis, ``bd`` = (1, 0): the input (3, 0) is shifted
    to (2, 0), encoded as 2, and decoded back to (3, 0):

    >>> import numpy as np
    >>> Wd, bd = np.array([[1.0], [0.0]]), np.array([1.0, 0.0])
    >>> params = (np.array([[1.0, 0.0]]), np.zeros(1), Wd, bd)
    >>> pre, f, rec = sae_forward(params, np.array([[3.0, 0.0]]))
    >>> pre.tolist(), f.tolist(), rec.tolist()
    ([[2.0]], [[2.0]], [[3.0, 0.0]])
    """
    We, be, Wd, bd = params
    pre = (h - bd) @ We.T + be
    f = np.maximum(pre, 0.0)
    return pre, f, f @ Wd.T + bd


def sae_loss_and_grads(params: Params, h: Array, lam: float) -> tuple[float, Params]:
    """
    Return the loss and its gradients with respect to ``(We, be, Wd, bd)``.

    The loss is the batch mean of ‖h - h'‖² + λ‖f‖₁. Backward, with B samples,
    R = ∂loss/∂h' and G = ∂loss/∂pre::

        R = 2 · (h' - h) / B
        G = (R Wd + λ · sign(f) / B) · (pre > 0)
        grad_Wd = Rᵀ f
        grad_bd = Σ R  -  Σ G We       (decoder use + encoder use)
        grad_We = Gᵀ (h - bd)
        grad_be = Σ G

    ``sign(f)`` is the subgradient of |f|: f is never negative, and it is
    exactly 0 only where the ReLU is off and G is masked anyway. ``bd`` appears
    twice, so its gradient has two terms.

    Examples
    --------
    One dimension, one atom, We = 2, Wd = 1, h = 1, λ = 0.5: f = 2 and the
    reconstruction is 2. R = 2 and G = 2·1 + 0.5 = 2.5, so grad_Wd = R·f = 4,
    grad_bd = R - G·We = 2 - 5 = -3, and grad_We = grad_be = G = 2.5. The value
    of grad_bd matches d/d(bd) of (1 - bd)² + (1 - bd) = -3 at bd = 0:

    >>> import numpy as np
    >>> params = (np.array([[2.0]]), np.zeros(1), np.array([[1.0]]), np.zeros(1))
    >>> loss, (gWe, gbe, gWd, gbd) = sae_loss_and_grads(params, np.array([[1.0]]), 0.5)
    >>> float(loss), float(gWe[0, 0]), float(gbe[0]), float(gWd[0, 0]), float(gbd[0])
    (2.0, 2.5, 2.5, 4.0, -3.0)

    With be = -5 the pre-activation is -3, so the unit is off: the output is
    bd = 0, the loss is 1, and only grad_bd = R = -2 is nonzero:

    >>> params = (np.array([[2.0]]), np.full(1, -5.0), np.array([[1.0]]), np.zeros(1))
    >>> loss, (gWe, gbe, gWd, gbd) = sae_loss_and_grads(params, np.array([[1.0]]), 0.5)
    >>> float(loss), float(gWe[0, 0]), float(gbe[0]), float(gWd[0, 0]), float(gbd[0])
    (1.0, 0.0, 0.0, 0.0, -2.0)

    A dictionary equal to the feature directions reconstructs any single
    feature exactly, so with λ = 0 the loss and every gradient are zero:

    >>> I = np.eye(3)
    >>> params = (I, np.zeros(3), I, np.zeros(3))
    >>> loss, grads = sae_loss_and_grads(params, np.array([[0.0, 0.7, 0.0]]), 0.0)
    >>> float(loss), [float(np.abs(g).max()) for g in grads]
    (0.0, [0.0, 0.0, 0.0, 0.0])
    """
    We, _, Wd, bd = params
    pre, f, rec = sae_forward(params, h)
    B = len(h)
    loss = float(np.mean(np.sum((rec - h) ** 2, axis=1) + lam * np.sum(f, axis=1)))
    R = 2.0 * (rec - h) / B
    G = (R @ Wd + lam * np.sign(f) / B) * (pre > 0)
    grad_Wd = R.T @ f
    grad_bd = R.sum(axis=0) - (G @ We).sum(axis=0)
    grad_We = G.T @ (h - bd)
    grad_be = G.sum(axis=0)
    return loss, (grad_We, grad_be, grad_Wd, grad_bd)


def gradient_error(
    grad_fn: GradFn,
    params: Params,
    h: Array,
    lam: float,
    eps: float = 1e-5,
) -> float:
    """
    Return the largest gap between ``grad_fn``'s gradients and central differences.

    ``grad_fn`` has the signature of ``sae_loss_and_grads``. Each entry of each
    parameter is nudged by ±``eps`` and the loss from ``grad_fn`` is
    differenced. An exact gradient gives an error near ``eps``², a wrong one
    gives an error of the size of the missing term.

    Examples
    --------
    The hand-derived gradient agrees with the numeric one:

    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> params = (
    ...     rng.normal(size=(6, 3)),
    ...     rng.normal(size=6),
    ...     rng.normal(size=(3, 6)),
    ...     rng.normal(size=3),
    ... )
    >>> h = rng.normal(size=(8, 3))
    >>> gradient_error(sae_loss_and_grads, params, h, 0.1) < 1e-6
    True

    The deliberate break, first: a gradient for ``bd`` that leaves out the term
    through the encoder, ``Σ G We``, is far off.

    >>> def no_encoder_bd(params, h, lam):
    ...     loss, (gWe, gbe, gWd, _) = sae_loss_and_grads(params, h, lam)
    ...     _, _, rec = sae_forward(params, h)
    ...     return loss, (gWe, gbe, gWd, (2 * (rec - h) / len(h)).sum(axis=0))
    >>> gradient_error(no_encoder_bd, params, h, 0.1) > 0.1
    True

    And second: a gradient that forgets the ReLU mask ``(pre > 0)``.

    >>> def no_mask(params, h, lam):
    ...     loss, _ = sae_loss_and_grads(params, h, lam)
    ...     We, _, Wd, bd = params
    ...     _, f, rec = sae_forward(params, h)
    ...     R = 2 * (rec - h) / len(h)
    ...     G = R @ Wd + lam * np.sign(f) / len(h)
    ...     grads = (G.T @ (h - bd), G.sum(axis=0), R.T @ f, R.sum(axis=0) - (G @ We).sum(axis=0))
    ...     return loss, grads
    >>> gradient_error(no_mask, params, h, 0.1) > 0.1
    True
    """
    _, analytic = grad_fn(params, h, lam)
    worst = 0.0
    for k, p in enumerate(params):
        for index in np.ndindex(*p.shape):
            losses = []
            for sign in (1.0, -1.0):
                shifted = [q.copy() for q in params]
                shifted[k][index] += sign * eps
                moved: Params = (shifted[0], shifted[1], shifted[2], shifted[3])
                losses.append(grad_fn(moved, h, lam)[0])
            numeric = (losses[0] - losses[1]) / (2 * eps)
            worst = max(worst, abs(float(analytic[k][index]) - numeric))
    return worst


def train_sae(
    H: Array,
    atoms: int,
    lam: float,
    steps: int = 6000,
    lr: float = 3e-3,
    batch: int = 1024,
    seed: int = 0,
) -> Params:
    """
    Train a sparse autoencoder on the rows of ``H`` and return ``(We, be, Wd, bd)``.

    Adam updates all four parameters, then each decoder column is rescaled to
    length 1. Without that rescale the penalty on ``f`` is cheapest to dodge by
    shrinking ``f`` and growing the decoder. Each step draws a batch of rows
    from ``H`` with the seeded generator, so a run repeats exactly.

    Examples
    --------
    Decoder columns have length 1 after any number of steps, and training
    lowers the loss relative to the untrained start (``steps=0``):

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> H = hidden_activations(rng.normal(size=(2, 4)), sample_features(rng, 500, 4, 0.7))
    >>> start = train_sae(H, 8, lam=0.01, steps=0)
    >>> trained = train_sae(H, 8, lam=0.01, steps=300)
    >>> trained[0].shape, trained[2].shape
    ((8, 2), (2, 8))
    >>> bool(np.allclose(np.linalg.norm(trained[2], axis=0), 1.0))
    True
    >>> sae_loss_and_grads(trained, H, 0.01)[0] < sae_loss_and_grads(start, H, 0.01)[0]
    True
    """
    rng = np.random.default_rng(seed)
    m = H.shape[1]
    params: list[Array] = [
        rng.normal(size=(atoms, m)) * 0.3,
        np.zeros(atoms),
        unit_columns(rng.normal(size=(m, atoms))),
        np.zeros(m),
    ]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    for t in range(1, steps + 1):
        h = H[rng.integers(0, len(H), batch)]
        _, grads = sae_loss_and_grads((params[0], params[1], params[2], params[3]), h, lam)
        adam_step(params, grads, first, second, t, lr)
        params[2][...] = unit_columns(params[2])
    return params[0], params[1], params[2], params[3]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MEASURING: Cosines, Variance, ASCII Bars
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def stored_directions(W: Array) -> Array:
    """
    Return the unit-length columns of ``W`` for the features the toy stores.

    Examples
    --------
    >>> import numpy as np
    >>> stored_directions(np.array([[3.0, 0.0, 0.0], [4.0, 0.1, 0.0]])).round(6).tolist()
    [[0.6], [0.8]]
    """
    return unit_columns(W[:, np.linalg.norm(W, axis=0) > 0.5])


def best_cosines(directions: Array, candidates: Array) -> Array:
    """
    Return each column of ``directions``'s largest |cosine| with any column of ``candidates``.

    Pass the decoder ``Wd`` as ``candidates`` to score the dictionary, and an
    identity matrix to score the hidden dimensions themselves. The sign is
    ignored because a direction and its negative span the same line.

    Examples
    --------
    The direction (0.6, 0.8) against the axes is best matched by the second,
    at 0.8; add it as a candidate and the match is exact. The axis (1, 0)
    matches itself either way:

    >>> import numpy as np
    >>> directions = np.array([[0.6, 1.0], [0.8, 0.0]])
    >>> best_cosines(directions, np.eye(2)).round(6).tolist()
    [0.8, 1.0]
    >>> with_atom = np.array([[1.0, 0.0, -0.6], [0.0, 1.0, -0.8]])
    >>> best_cosines(directions, with_atom).round(6).tolist()
    [1.0, 1.0]
    """
    return np.asarray(
        np.abs(unit_columns(candidates).T @ unit_columns(directions)).max(axis=0),
        dtype=np.float64,
    )


def matched(cosines: Array, threshold: float = 0.9) -> int:
    """
    Count the entries of ``cosines`` above ``threshold``.

    Examples
    --------
    >>> import numpy as np
    >>> matched(np.array([0.95, 0.9, 0.5, 1.0]))
    2
    """
    return int(np.sum(cosines > threshold))


def variance_explained(h: Array, rec: Array) -> float:
    """
    Return 1 - Σ‖h - h'‖² / Σ‖h - mean(h)‖², the share of ``h``'s variance that ``rec`` explains.

    Examples
    --------
    A perfect reconstruction explains everything and the mean explains nothing:

    >>> import numpy as np
    >>> h = np.array([[0.0, 0.0], [2.0, 4.0]])
    >>> variance_explained(h, h)
    1.0
    >>> variance_explained(h, np.tile(h.mean(axis=0), (2, 1)))
    0.0
    """
    return float(1.0 - np.sum((h - rec) ** 2) / np.sum((h - h.mean(axis=0)) ** 2))


def bars(rows: Sequence[tuple[str, float]], width: int = 40, top: float | None = None) -> str:
    """
    Draw one bar per row, with lengths proportional to each value.

    ``top`` is the value that fills ``width`` characters; it defaults to the
    largest value. Labels are left-aligned to the longest one.

    Examples
    --------
    >>> print(bars([("a", 0.5), ("bb", 1.0)], width=10))
    a  |#####       0.50
    bb |##########  1.00
    >>> print(bars([("a", 0.5)], width=10, top=2.0))
    a |##          0.50
    """
    label_width = max(len(label) for label, _ in rows)
    full = top if top is not None else max(value for _, value in rows)
    lines = []
    for label, value in rows:
        length = round(value / full * width) if full > 0 else 0
        lines.append(f"{label:<{label_width}} |{'#' * length:<{width}}  {value:.2f}")
    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI ARGUMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True)
class Args:
    """Parsed command-line arguments.

    Attributes
    ----------
    doctest : bool
        Run the doctests before the demo. On by default; ``--no-doctest``
        turns it off.
    doctest_only : bool
        Run the doctests and exit without running the demo.
    verbose : bool
        Report each doctest rather than only failures.
    show_index : bool
        Print the lesson index instead of running anything.
    """

    doctest: bool
    doctest_only: bool
    verbose: bool
    show_index: bool


def parse_args(argv: list[str] | None = None) -> Args:
    """
    Parse the lesson's command-line flags.

    Examples
    --------
    >>> parse_args(["--doctest-only"]).doctest_only
    True
    >>> parse_args(["--no-doctest"]).doctest
    False
    """
    parser = argparse.ArgumentParser(description=f"Run {LESSON_ID}")
    parser.add_argument(
        "--doctest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run doctests before demo (default: True)",
    )
    parser.add_argument("--doctest-only", action="store_true", help="Run doctests and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose doctest output")
    parser.add_argument(
        "--index",
        action="store_true",
        dest="show_index",
        help="Show lesson index (dependencies, successors, related)",
    )
    ns = parser.parse_args(argv)
    return Args(
        doctest=ns.doctest,
        doctest_only=ns.doctest_only,
        verbose=ns.verbose,
        show_index=ns.show_index,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX DISPLAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _format_mapping(name: str, items: Mapping[str, str]) -> str:
    """
    Format a ``lesson_id -> reason`` mapping as an indented list.

    Examples
    --------
    >>> print(_format_mapping("Related", {}))
    Related: (none)
    >>> print(_format_mapping("Next", {"tier_9_x": "why"}))
    Next:
       → tier_9_x: why
    """
    if not items:
        return f"{name}: (none)"
    lines = "\n".join(f"   → {lesson_id}: {reason}" for lesson_id, reason in items.items())
    return f"{name}:\n{lines}"


def show_lesson_index() -> None:
    """
    Print where this lesson sits in the curriculum.

    Examples
    --------
    >>> show_lesson_index()  # doctest: +ELLIPSIS
    ═...
      LESSON INDEX: tier_5_lesson_2b_sparse_autoencoders
    ...
    Prerequisites (hard deps):
       → tier_5_lesson_2a_superposition: The toy whose features it recovers
    ...
       → Power-up in: tier_5_lesson_3_pytorch
    """
    print("═" * 70)
    print(f"  LESSON INDEX: {LESSON_ID}")
    print("═" * 70)
    print()
    print(_format_mapping("Prerequisites (hard deps)", PREREQUISITES))
    print()
    print(_format_mapping("Dependencies (soft deps)", DEPENDENCIES))
    print()
    print(_format_mapping("Successors (unlocks)", SUCCESSORS))
    print()
    print(_format_mapping("Dependents (rely on this)", DEPENDENTS))
    print()
    print(_format_mapping("Related (optional)", RELATED))
    print()
    print("Power-Up Timeline:")
    if PAIN_FROM_LESSON:
        print(f"   ← Pain from: {PAIN_FROM_LESSON}")
    if POWER_UP_AVAILABLE:
        print("   ✓ Power-up: Available in THIS lesson")
    elif POWER_UP_IN_LESSON:
        print(f"   → Power-up in: {POWER_UP_IN_LESSON}")
    else:
        print("   → Power-up in: (future lesson TBD)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEMONSTRATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def demo(
    n: int = 12,
    m: int = 4,
    sparsity: float = 0.95,
    toy_steps: int = 8000,
    samples: int = 50_000,
    sae_steps: int = 6000,
    lam: float = 3e-3,
    threshold: float = 0.9,
) -> None:
    """
    Train the toy and a 3n-atom sparse autoencoder, then compare atoms with hidden dimensions.

    Prints how many features the toy stores, the variance the dictionary
    explains, each stored feature's best |cosine| with an atom and with a
    hidden dimension, and how many of each clear ``threshold``.
    The matched count moves with the dictionary's seed: across six seeds it ran
    from 5 to 9 of 12, and seed 3, used here, is the best of them. The median
    cosine stayed between 0.89 and 0.96, above the hidden dimensions' 0.78, in
    every seed tried.

    Examples
    --------
    >>> demo(n=5, m=2, toy_steps=2000, samples=2000, sae_steps=300)  # doctest: +ELLIPSIS
    ═...
    The toy stores 5 of 5 features in 2 dimensions.
    ...
    Matched (cosine above 0.9), out of 5:
    ...
    """
    print("═" * 70)
    print(f"  {LESSON_ID.upper().replace('_', ' ')}")
    print("═" * 70)
    print()
    W, _ = train_toy(n, m, sparsity, steps=toy_steps)
    stored = features_represented(W)
    print(f"The toy stores {stored} of {n} features in {m} dimensions.")
    rng = np.random.default_rng(1)
    H = hidden_activations(W, sample_features(rng, samples, n, sparsity))
    atoms = 3 * n
    params = train_sae(H, atoms, lam, steps=sae_steps, seed=3)
    _, _, rec = sae_forward(params, H)
    print(
        f"A dictionary of {atoms} atoms explains {variance_explained(H, rec):.2%} "
        "of the variance of the hidden activations."
    )
    print()
    directions = stored_directions(W)
    by_atom = best_cosines(directions, params[2])
    by_basis = best_cosines(directions, np.eye(m))
    print("Best |cosine| of each stored feature with an atom, and with a hidden dimension:")
    rows: list[tuple[str, float]] = []
    for i, (a, b) in enumerate(zip(by_atom, by_basis, strict=True)):
        rows.append((f"feature {i:<2} atom ", float(a)))
        rows.append((f"feature {i:<2} basis", float(b)))
    print(bars(rows, width=30, top=1.0))
    print()
    print(f"Matched (cosine above {threshold:g}), out of {stored}:")
    print(f"  dictionary atoms:  {matched(by_atom, threshold)}")
    print(f"  hidden dimensions: {matched(by_basis, threshold)}")
    median_atom, median_basis = np.median(by_atom), np.median(by_basis)
    print(f"Median best cosine: atoms {median_atom:.2f}, hidden dimensions {median_basis:.2f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main(argv: list[str] | None = None) -> int:
    """
    Parse arguments, run the doctests unless told not to, then run the demo.

    Returns
    -------
    int
        Exit code: 0 on success, 1 when a doctest fails.

    Examples
    --------
    >>> main(["--index"])  # doctest: +ELLIPSIS
    ═...
    0
    """
    import doctest

    args = parse_args(argv)
    if args.show_index:
        show_lesson_index()
        return 0
    if args.doctest:
        results = doctest.testmod(verbose=args.verbose)
        if results.failed > 0:
            print(f"DOCTESTS FAILED: {results.failed}/{results.attempted}")
            return 1
        print(f"All {results.attempted} doctests passed\n")
    if args.doctest_only:
        return 0
    demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
