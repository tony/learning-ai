#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<4.0"
# dependencies = ["numpy>=2.5.4"]
# ///
"""
Lesson 7.3a: How Attention Scores Keys (side quest).

Attention is a softmax over scores, one score per key, and the weights it
produces average the values. You measure two choices behind the scores. First,
why scaled dot-product attention divides by sqrt(d_k): for unit-variance
queries and keys the variance of q.k equals d_k, so an unscaled softmax over
64 keys collapses onto one key as d_k grows and its gradient vanishes, while
the scaled softmax keeps the same spread at every d_k. Second, how additive
(Bahdanau) scoring compares with dot-product (Luong, Vaswani) scoring on a toy
alignment task: a query is a fixed hidden linear map of one of T keys plus
noise, and the model must put its attention on that key. Both scores have
gradients derived by hand and checked against central differences, and both
train with a hand-written Adam.

Sources:

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017, arXiv:1706.03762.
  Scaled dot-product attention: "the dot products grow large in magnitude,
  pushing the softmax function into regions where it has extremely small
  gradients".
- Bahdanau, Cho, Bengio, "Neural Machine Translation by Jointly Learning to
  Align and Translate", ICLR 2015, arXiv:1409.0473. Additive attention: the
  alignment model is a small feedforward network.
- Luong, Pham, Manning, "Effective Approaches to Attention-based Neural
  Machine Translation", EMNLP 2015, arXiv:1508.04025. The dot and general
  scores.
- Olah and Carter, "Attention and Augmented Recurrent Neural Networks",
  Distill (2016), https://distill.pub/2016/augmented-rnns/.

Prerequisites (must complete first)
----------------------------------
- Lesson 7.3: Attention with NumPy (scaled dot-product attention in NumPy)

Learning Objectives
-------------------
After completing this lesson, you will be able to:
1. Measure how the softmax over unscaled dot products saturates as d_k grows,
   and show that dividing by sqrt(d_k) removes the effect.
2. Write the dot-product and additive scores with hand-derived gradients, and
   check each against central finite differences.
3. Train both on an alignment task and read the results, including why
   additive scoring without its bias stays at chance.

The Big Idea
------------
A score function answers "how well does this key match this query", and the
softmax turns the answers into weights. Dot-product scoring is a bilinear
form, so it forms the query-key interaction directly; additive scoring feeds
the sum of the two through a tanh layer, and needs the bias to make the
interaction. Dividing by sqrt(d_k) keeps the scores at unit spread so the
softmax stays in the region where it has a gradient.

The bias matters because tanh is odd. With the bias at zero, the additive
score satisfies s(-q, -k) = -s(q, k). The task needs the target key to score
highest. Negating the query and every key gives another valid example, since
-q is the hidden map of the negated target, but it negates every score, so the
target would have to rank first and last at once. The best the model can do is
score every key alike, and the loss stays at ln(T). With the bias, tanh(Wa q + Ua k
+ b) is no longer odd, and the same network learns the interaction.

Why It Matters
--------------
Every transformer scores keys this way, and the 1/sqrt(d_k) factor is the
difference between a model that trains at d_k = 64 and one whose attention
rows are one-hot from step one. The same softmax-over-scores shape recurs
wherever a model selects among candidates: contrastive learning, retrieval,
and pointer networks.

Stdlib Pain → Library Power-Up
------------------------------
Pain (this lesson):
- Both gradients are written by hand over (batch, keys, hidden) tensors, and
  the finite-difference check loops over every parameter.
- Lines of code: ~90 for the scores, gradients, and optimizer.
- Runtime: about 15 seconds for the demo with one BLAS thread; one training
  run of 1500 steps takes 1 to 2 seconds.
- Common bugs: forgetting the tanh derivative, dropping the 1/sqrt(d) in
  only one of forward and backward, and fixing the additive bias at zero.

Power-Up (Lesson 7.4 — PyTorch):
- ``torch.nn.functional.scaled_dot_product_attention`` applies the scaled
  score, the softmax, and the value average in one call, and autograd
  replaces the hand-derived gradients.
- Trade-off: you give up the visible derivation; you gain masks, batching
  across heads, and fused kernels.

Complexity Analysis
-------------------
Pain (manual), for a batch of n queries, T keys, input dimension d, and
hidden dimension h:
- Time:  O(n·T·d·h) per step for either score. Additive adds an O(n·T·h)
  tanh and costs about 1.5 to 2x as much per step as dot here.
- Space: O(n·T·h) for the projected keys and the tanh activations.

Execution
---------
Project-style:
    uv run python src/073a_attention_scoring.py

Script-style (self-contained deps):
    uv run --script src/073a_attention_scoring.py

On a busy machine, NumPy's BLAS threads compete with other work; one thread
avoids that, since the matrices here are small:
    OPENBLAS_NUM_THREADS=1 uv run python src/073a_attention_scoring.py

Options:
    python src/073a_attention_scoring.py --doctest-only
    python src/073a_attention_scoring.py --no-doctest
    python src/073a_attention_scoring.py --index
    python src/073a_attention_scoring.py -v

Quality Gates
-------------
    uv run pytest src/073a_attention_scoring.py
    uv run ruff check src/073a_attention_scoring.py
    uv run ruff format --check src/073a_attention_scoring.py
    uv run mypy --strict src/073a_attention_scoring.py

Module Doctests
---------------
Equal scores give equal weights, and the output averages the values:

>>> import numpy as np
>>> attend(np.zeros(2), np.array([[1.0, 0.0], [0.0, 1.0]])).tolist()
[0.5, 0.5]

The variance of q.k for unit-variance q and k grows as d_k:

>>> round(dot_variance(16, samples=20_000))
16
"""

from __future__ import annotations

import argparse
import doctest
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

    from numpy.typing import NDArray

    type Array = NDArray[np.float64]
    type Labels = NDArray[np.int64]
    type Params = dict[str, Array]
    type GradFn = Callable[[str, Params, Array, Array, Labels], Params]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON IDENTITY & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON_ID: Final[str] = "tier_7_lesson_3a_attention_scoring"

# The library form is PyTorch's scaled_dot_product_attention, in Lesson 7.4.
POWER_UP_AVAILABLE: Final[bool] = False

KINDS: Final[tuple[str, ...]] = ("dot", "additive", "additive0")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Format: lesson_id -> reason

PREREQUISITES: Final[Mapping[str, str]] = {
    "tier_7_lesson_3_attention_numpy": "Scaled dot-product attention in NumPy",
}

DEPENDENCIES: Final[Mapping[str, str]] = {
    "tier_2_lesson_3_softmax": "Each row of scores becomes a distribution",
}

SUCCESSORS: Final[Mapping[str, str]] = {
    "tier_7_lesson_4_multi_head_attention": "Runs many scaled dot-product heads in parallel",
}

DEPENDENTS: Final[Mapping[str, str]] = {}

RELATED: Final[Mapping[str, str]] = {
    "tier_5_lesson_2f_contrastive_learning": "Another softmax over dot products",
}

POWER_UP_IN_LESSON: Final[str | None] = "tier_7_lesson_4_multi_head_attention"
PAIN_FROM_LESSON: Final[str | None] = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOFTMAX AND ATTEND
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def softmax(scores: Array) -> Array:
    """
    Return the softmax over the last axis, with the row maximum subtracted first.

    Examples
    --------
    >>> import numpy as np
    >>> softmax(np.array([0.0, 0.0, 0.0, 0.0])).tolist()
    [0.25, 0.25, 0.25, 0.25]

    Subtracting the maximum keeps large scores from overflowing:

    >>> softmax(np.array([1000.0, 1000.0])).tolist()
    [0.5, 0.5]
    """
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return np.asarray(e / e.sum(axis=-1, keepdims=True), dtype=np.float64)


def attend(scores: Array, values: Array) -> Array:
    """
    Return the softmax-weighted sum of ``values``, one output per row of ``scores``.

    ``scores`` has shape ``(..., T)`` and ``values`` has shape ``(..., T, d_v)``.
    The scores decide how much of each value reaches the output; they do not
    appear in it otherwise.

    Complexity
    ----------
    Time
        O(T·d_v) per query.
    Space
        O(T) for the weights.

    Examples
    --------
    A score far above the rest copies that key's value:

    >>> import numpy as np
    >>> V = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    >>> attend(np.array([0.0, 30.0, 0.0]), V).round(3).tolist()
    [0.0, 1.0]

    Two equal scores average their values, and a batch gets one output per row:

    >>> attend(np.array([[0.0, 0.0, -50.0]]), V[None]).round(3).tolist()
    [[0.5, 0.5]]
    """
    return np.asarray(np.einsum("...t,...td->...d", softmax(scores), values), dtype=np.float64)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART A: WHY DIVIDE BY sqrt(d_k)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def dot_variance(d_k: int, samples: int = 100_000, seed: int = 0) -> float:
    """
    Return the sample variance of ``q . k`` for unit-variance Gaussian ``q`` and ``k``.

    Each product ``q_i k_i`` has variance 1 and the ``d_k`` products are
    independent, so the sum has variance ``d_k``.

    Examples
    --------
    >>> round(dot_variance(4, samples=40_000))
    4
    >>> round(dot_variance(64, samples=20_000) / 64, 1)
    1.0
    """
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((samples, d_k))
    k = rng.standard_normal((samples, d_k))
    return float(np.var((q * k).sum(axis=1)))


def softmax_weights(
    d_k: int,
    scale: bool,
    n_keys: int = 64,
    trials: int = 2000,
    seed: int = 0,
) -> Array:
    """
    Return softmax weights of shape ``(trials, n_keys)`` over random ``q . k`` scores.

    With ``scale`` the scores are divided by ``sqrt(d_k)``. ``trials`` is
    reduced so the key tensor stays under about 4 million floats.

    Examples
    --------
    >>> w = softmax_weights(8, scale=True, n_keys=5, trials=10)
    >>> w.shape
    (10, 5)
    >>> w.sum(axis=1).round(6).tolist()
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    """
    rng = np.random.default_rng(seed)
    trials = max(1, min(trials, 4_000_000 // (n_keys * d_k)))
    q = rng.standard_normal((trials, d_k))
    keys = rng.standard_normal((trials, n_keys, d_k))
    scores = np.einsum("nd,ntd->nt", q, keys)
    if scale:
        scores = scores / np.sqrt(d_k)
    return softmax(np.asarray(scores, dtype=np.float64))


def saturation(
    d_k: int,
    scale: bool,
    n_keys: int = 64,
    trials: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Return the mean max weight and mean entropy (nats) of softmax over random ``q . k``.

    Uniform weights over ``n_keys`` keys have max weight ``1/n_keys`` and
    entropy ``ln(n_keys)``; a one-hot row has max weight 1 and entropy 0.

    Examples
    --------
    Without scaling, the weights concentrate as ``d_k`` grows:

    >>> small, _ = saturation(4, scale=False, trials=300)
    >>> large, _ = saturation(256, scale=False, trials=300)
    >>> small < 0.4 < 0.8 < large
    True

    With scaling the spread stays about the same:

    >>> small, _ = saturation(4, scale=True, trials=300)
    >>> large, _ = saturation(256, scale=True, trials=300)
    >>> abs(large - small) < 0.03
    True
    """
    p = softmax_weights(d_k, scale, n_keys, trials, seed)
    entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(axis=1)
    return float(p.max(axis=1).mean()), float(entropy.mean())


def jacobian_norm(p: Array) -> float:
    """
    Return the mean Frobenius norm of the softmax Jacobian ``diag(p) - p p^T`` over rows of ``p``.

    This is the size of the gradient the softmax passes back to its scores. A
    one-hot row has a zero Jacobian: moving any score changes nothing.

    Examples
    --------
    >>> import numpy as np
    >>> jacobian_norm(np.array([[1.0, 0.0, 0.0]]))
    0.0
    >>> jacobian_norm(np.array([[0.5, 0.5]]))
    0.5
    """
    s2 = (p**2).sum(axis=1)
    s3 = (p**3).sum(axis=1)
    return float(np.sqrt(np.maximum(s2 - 2 * s3 + s2**2, 0.0)).mean())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART B: TWO SCORE FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Shapes: q (n, d), K (n, T, d), scores (n, T), target (n,). Both scores
# project into h dimensions first.


def init_params(kind: str, d: int, h: int, rng: np.random.Generator) -> Params:
    """
    Return initial parameters for ``dot``, ``additive``, or ``additive0``.

    ``dot`` has the projections ``Wq`` and ``Wk``. ``additive`` adds the output
    vector ``v`` and a bias ``b``; ``additive0`` is ``additive`` with ``b``
    held at zero by ``grads``.

    Examples
    --------
    >>> import numpy as np
    >>> sorted(init_params("dot", 4, 3, np.random.default_rng(0)))
    ['Wk', 'Wq']
    >>> sorted(init_params("additive", 4, 3, np.random.default_rng(0)))
    ['Ua', 'Wa', 'b', 'v']
    >>> init_params("additive0", 4, 3, np.random.default_rng(0))["b"].tolist()
    [0.0, 0.0, 0.0]
    >>> init_params("bilinear", 4, 3, np.random.default_rng(0))
    Traceback (most recent call last):
    ...
    ValueError: unknown kind 'bilinear'
    """
    if kind not in KINDS:
        msg = f"unknown kind {kind!r}"
        raise ValueError(msg)
    first = rng.standard_normal((h, d)) / np.sqrt(d)
    second = rng.standard_normal((h, d)) / np.sqrt(d)
    if kind == "dot":
        return {"Wq": first, "Wk": second}
    bias = rng.standard_normal(h) * 0.5 if kind == "additive" else np.zeros(h)
    return {"Wa": first, "Ua": second, "v": rng.standard_normal(h) / np.sqrt(h), "b": bias}


def score_dot(w: Params, q: Array, K: Array) -> Array:
    """
    Return the dot-product scores ``(Wq q) . (Wk k_j) / sqrt(h)``, shape ``(n, T)``.

    The score is bilinear in ``q`` and ``k_j``, so it holds the query-key
    interaction directly. ``h`` is the projected dimension, the lesson's
    ``d_k``.

    Complexity
    ----------
    Time
        O(n·T·d·h).
    Space
        O(n·T·h).

    Examples
    --------
    With identity projections the score is ``q . k / sqrt(h)``:

    >>> import numpy as np
    >>> w = {"Wq": np.eye(2), "Wk": np.eye(2)}
    >>> score_dot(w, np.array([[3.0, 4.0]]), np.array([[[1.0, 0.0], [0.0, 2.0]]])).round(4).tolist()
    [[2.1213, 5.6569]]
    """
    a = q @ w["Wq"].T
    b = K @ w["Wk"].T
    return np.asarray(np.einsum("nh,nth->nt", a, b) / np.sqrt(a.shape[1]), dtype=np.float64)


def score_additive(w: Params, q: Array, K: Array) -> Array:
    """
    Return the additive scores ``v . tanh(Wa q + Ua k_j + b)``, shape ``(n, T)``.

    A one-hidden-layer network scores the pair. With ``b = 0`` the score is odd
    under ``(q, k) -> (-q, -k)``, which is why the bias matters below.

    Complexity
    ----------
    Time
        O(n·T·d·h).
    Space
        O(n·T·h).

    Examples
    --------
    >>> import numpy as np
    >>> w = {"Wa": np.eye(2), "Ua": np.eye(2), "v": np.ones(2), "b": np.zeros(2)}
    >>> q, K = np.array([[1.0, 0.0]]), np.array([[[0.0, 0.0], [1.0, 1.0]]])
    >>> score_additive(w, q, K).round(4).tolist()
    [[0.7616, 1.7256]]
    >>> bool(np.allclose(score_additive(w, -q, -K), -score_additive(w, q, K)))
    True
    """
    z = np.tanh((q @ w["Wa"].T)[:, None, :] + K @ w["Ua"].T + w["b"])
    return np.asarray(z @ w["v"], dtype=np.float64)


def scores(kind: str, w: Params, q: Array, K: Array) -> Array:
    """
    Return the scores of the model named by ``kind``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> w = init_params("dot", 3, 4, rng)
    >>> q, K = rng.standard_normal((2, 3)), rng.standard_normal((2, 5, 3))
    >>> scores("dot", w, q, K).shape
    (2, 5)
    """
    return score_dot(w, q, K) if kind == "dot" else score_additive(w, q, K)


def loss(kind: str, w: Params, q: Array, K: Array, target: Labels) -> float:
    """
    Return the mean softmax cross-entropy of putting the attention on ``target``.

    Examples
    --------
    Zero scores spread the attention evenly, so the loss is ``ln(T)``:

    >>> import numpy as np
    >>> w = {"Wq": np.zeros((2, 2)), "Wk": np.zeros((2, 2))}
    >>> L = loss("dot", w, np.ones((3, 2)), np.ones((3, 4, 2)), np.array([0, 1, 2]))
    >>> round(L, 4) == round(float(np.log(4)), 4)
    True
    """
    p = softmax(scores(kind, w, q, K))
    return float(-np.log(p[np.arange(len(q)), target]).mean())


def grads(kind: str, w: Params, q: Array, K: Array, target: Labels) -> Params:
    """
    Return the hand-derived gradient of ``loss`` for every parameter.

    The softmax cross-entropy gradient with respect to the scores is
    ``ds = (p - onehot(target)) / n``. Then, with ``a = Wq q``, ``b = Wk k_j``
    and ``r = 1/sqrt(h)``, the dot score ``s = r a.b`` gives::

        da = r sum_j ds_j b_j        dWq = da^T q
        db_j = r ds_j a              dWk = sum_j db_j^T k_j

    The additive score ``s = v.z`` with ``z = tanh(pre)`` and
    ``pre = Wa q + Ua k_j + b`` gives, using ``tanh' = 1 - z^2``::

        dv = sum ds_j z_j            dpre_j = ds_j v * (1 - z_j^2)
        dWa = (sum_j dpre_j)^T q     dUa = sum_j dpre_j^T k_j
        db = sum dpre_j

    For ``additive0`` the gradient of ``b`` is zero, so ``b`` stays at 0.

    Complexity
    ----------
    Time
        O(n·T·d·h).
    Space
        O(n·T·h).

    Examples
    --------
    Every parameter gets a gradient of its own shape, and the zero-bias model
    has no gradient on its bias:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> q, K, target = make_task(5, 4, 3, rng, np.eye(3), 0.1)
    >>> g = grads("additive0", init_params("additive0", 3, 2, rng), q, K, target)
    >>> {name: arr.shape for name, arr in g.items()}
    {'Wa': (2, 3), 'Ua': (2, 3), 'v': (2,), 'b': (2,)}
    >>> g["b"].tolist()
    [0.0, 0.0]
    """
    n = len(q)
    p = softmax(scores(kind, w, q, K))
    ds = p.copy()
    ds[np.arange(n), target] -= 1.0
    ds /= n
    if kind == "dot":
        a = q @ w["Wq"].T
        b = K @ w["Wk"].T
        r = 1.0 / np.sqrt(a.shape[1])
        da = np.einsum("nt,nth->nh", ds, b) * r
        db = ds[:, :, None] * a[:, None, :] * r
        return {"Wq": da.T @ q, "Wk": np.einsum("nth,ntd->hd", db, K)}
    z = np.tanh((q @ w["Wa"].T)[:, None, :] + K @ w["Ua"].T + w["b"])
    dpre = ds[:, :, None] * w["v"] * (1.0 - z * z)
    return {
        "Wa": dpre.sum(axis=1).T @ q,
        "Ua": np.einsum("nth,ntd->hd", dpre, K),
        "v": np.einsum("nt,nth->h", ds, z),
        "b": dpre.sum(axis=(0, 1)) if kind == "additive" else np.zeros_like(w["b"]),
    }


def gradient_error(
    kind: str,
    w: Params,
    q: Array,
    K: Array,
    target: Labels,
    grad_fn: GradFn = grads,
    eps: float = 1e-6,
) -> float:
    """
    Return the largest gap between ``grad_fn``'s gradient and central differences of ``loss``.

    Each parameter entry is nudged by +/-``eps`` and the change in the loss is
    differenced. An exact gradient gives an error far below ``eps``; a wrong
    one gives an error of the size of the missing term. For ``additive0`` the
    bias is held fixed, so its numeric gradient is zero by construction.

    Examples
    --------
    Both hand-derived gradients agree with the numeric ones:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> q, K, target = make_task(6, 5, 4, rng, np.eye(4), 0.1)
    >>> w_dot = init_params("dot", 4, 3, rng)
    >>> w_add = init_params("additive", 4, 3, rng)
    >>> gradient_error("dot", w_dot, q, K, target) < 1e-7
    True
    >>> gradient_error("additive", w_add, q, K, target) < 1e-7
    True

    The deliberate breaks: a dot gradient that forgets the ``1/sqrt(h)`` and an
    additive gradient that drops the ``tanh`` derivative are far off:

    >>> def no_scale(kind, w, q, K, target):
    ...     g = grads(kind, w, q, K, target)
    ...     return {name: arr * np.sqrt(3.0) for name, arr in g.items()}
    >>> gradient_error("dot", w_dot, q, K, target, no_scale) > 1e-3
    True
    >>> def no_tanh_derivative(kind, w, q, K, target):
    ...     z = np.tanh((q @ w["Wa"].T)[:, None, :] + K @ w["Ua"].T + w["b"])
    ...     ds = softmax(scores(kind, w, q, K))
    ...     ds[np.arange(len(q)), target] -= 1.0
    ...     dpre = (ds / len(q))[:, :, None] * w["v"]
    ...     return {
    ...         "Wa": dpre.sum(axis=1).T @ q,
    ...         "Ua": np.einsum("nth,ntd->hd", dpre, K),
    ...         "v": np.einsum("nt,nth->h", ds / len(q), z),
    ...         "b": dpre.sum(axis=(0, 1)),
    ...     }
    >>> gradient_error("additive", w_add, q, K, target, no_tanh_derivative) > 1e-3
    True
    """
    analytic = grad_fn(kind, w, q, K, target)
    worst = 0.0
    for name, arr in w.items():
        for index in np.ndindex(*arr.shape):
            old = arr[index]
            values = []
            for step in (eps, -eps):
                arr[index] = old + step
                values.append(loss(kind, w, q, K, target))
            arr[index] = old
            numeric = (values[0] - values[1]) / (2 * eps)
            worst = max(worst, abs(numeric - float(analytic[name][index])))
    return worst


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART B: THE ALIGNMENT TASK AND ITS TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def hidden_map(d: int, seed: int = 99) -> Array:
    """
    Return the fixed ``d x d`` matrix the queries are made from, scaled by ``1/sqrt(d)``.

    Examples
    --------
    >>> hidden_map(4).shape
    (4, 4)
    >>> bool((hidden_map(4) == hidden_map(4)).all())
    True
    """
    return np.asarray(
        np.random.default_rng(seed).standard_normal((d, d)) / np.sqrt(d), dtype=np.float64
    )


def make_task(
    n: int,
    t: int,
    d: int,
    rng: np.random.Generator,
    m: Array,
    noise: float,
) -> tuple[Array, Array, Labels]:
    """
    Return queries, key sets, and target indices; each query is ``m @ k_target + noise``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> q, K, target = make_task(7, 5, 3, rng, np.eye(3), 0.0)
    >>> q.shape, K.shape, target.shape
    ((7, 3), (7, 5, 3), (7,))

    With an identity map and no noise the query equals its target key:

    >>> bool(np.allclose(q, K[np.arange(7), target]))
    True
    """
    K = rng.standard_normal((n, t, d))
    target = rng.integers(0, t, size=n)
    q = K[np.arange(n), target] @ m.T + noise * rng.standard_normal((n, d))
    return np.asarray(q, dtype=np.float64), K, target.astype(np.int64)


def adam_step(
    w: Params,
    g: Params,
    first: Params,
    second: Params,
    t: int,
    lr: float = 0.01,
) -> None:
    """
    Update ``w`` in place with one Adam step; ``first`` and ``second`` are updated too.

    Adam uses betas 0.9 and 0.999 and eps 1e-8. ``t`` counts steps from 1 and
    corrects the bias of the zero-initialised moments.

    Examples
    --------
    The first step moves every entry by about ``lr`` against its gradient's sign:

    >>> import numpy as np
    >>> w = {"x": np.array([1.0, 1.0])}
    >>> zeros = {"x": np.zeros(2)}
    >>> adam_step(w, {"x": np.array([3.0, -0.5])}, zeros, {"x": np.zeros(2)}, t=1, lr=0.1)
    >>> w["x"].round(3).tolist()
    [0.9, 1.1]
    """
    for name in w:
        first[name] = 0.9 * first[name] + 0.1 * g[name]
        second[name] = 0.999 * second[name] + 0.001 * g[name] ** 2
        mean = first[name] / (1 - 0.9**t)
        var = second[name] / (1 - 0.999**t)
        w[name] -= lr * mean / (np.sqrt(var) + 1e-8)


@dataclass(frozen=True, slots=True)
class Trained:
    """A trained model and how it did on held-out queries.

    Attributes
    ----------
    params : dict[str, Array]
        Trained parameters.
    accuracy : float
        Share of held-out queries whose top-scoring key is the target.
    loss : float
        Mean held-out cross-entropy; ``ln(T)`` is chance.
    cosine : float
        Mean cosine similarity between ``attend(scores, K)`` and the target key.
    seconds : float
        Wall time spent in the training loop.
    """

    params: Params
    accuracy: float
    loss: float
    cosine: float
    seconds: float


def train(
    kind: str,
    seed: int,
    steps: int,
    d: int = 16,
    h: int = 16,
    t: int = 12,
    batch: int = 64,
    test_n: int = 2000,
) -> Trained:
    """
    Train ``kind`` with Adam on fresh batches and score it on held-out queries.

    Each step draws ``batch`` new queries, so there is no train/test overlap
    to speak of; the held-out set uses a separate generator.

    Complexity
    ----------
    Time
        O(steps·batch·T·d·h).
    Space
        O(batch·T·h).

    Examples
    --------
    The dot model learns the alignment well above the chance rate of 1/6:

    >>> r = train("dot", seed=0, steps=300, d=6, h=8, t=6, test_n=400)
    >>> bool(r.accuracy > 0.6 > 2 / 6)
    True
    >>> bool(r.loss < 1.4 < 1.7)
    True

    The additive model needs more steps; the bias lets it learn too:

    >>> r = train("additive", seed=0, steps=300, d=6, h=8, t=6, test_n=400)
    >>> bool(r.accuracy > 0.6)
    True

    The deliberate break: with the bias fixed at zero, tanh's oddness leaves
    the additive model at chance, and the loss stays near ``ln(6) = 1.79``:

    >>> r = train("additive0", seed=0, steps=300, d=6, h=8, t=6, test_n=400)
    >>> bool(r.accuracy < 0.3)
    True
    >>> round(r.loss, 1)
    1.8
    """
    rng = np.random.default_rng(seed)
    m = hidden_map(d)
    w = init_params(kind, d, h, rng)
    first = {name: np.zeros_like(arr) for name, arr in w.items()}
    second = {name: np.zeros_like(arr) for name, arr in w.items()}
    start = time.perf_counter()
    for step in range(1, steps + 1):
        q, K, target = make_task(batch, t, d, rng, m, 0.1)
        adam_step(w, grads(kind, w, q, K, target), first, second, step)
    seconds = time.perf_counter() - start
    q, K, target = make_task(test_n, t, d, np.random.default_rng(12345), m, 0.1)
    s = scores(kind, w, q, K)
    picked = attend(s, K)
    rows = np.arange(test_n)
    truth = K[rows, target]
    cosine = (picked * truth).sum(axis=1) / (
        np.linalg.norm(picked, axis=1) * np.linalg.norm(truth, axis=1)
    )
    return Trained(
        params=w,
        accuracy=float((s.argmax(axis=1) == target).mean()),
        loss=loss(kind, w, q, K, target),
        cosine=float(cosine.mean()),
        seconds=seconds,
    )


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
      LESSON INDEX: tier_7_lesson_3a_attention_scoring
    ...
    Prerequisites (hard deps):
       → tier_7_lesson_3_attention_numpy: Scaled dot-product attention in NumPy
    ...
       → Power-up in: tier_7_lesson_4_multi_head_attention
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


def _heading(text: str) -> None:
    """
    Print a blank line, the heading, and a rule.

    Examples
    --------
    >>> _heading("Title")
    <BLANKLINE>
    Title
    ──────────────────────────────────────────────────────────────────────
    """
    print()
    print(text)
    print("─" * 70)


def demo(steps: int = 1500, seeds: int = 3, trials: int = 2000) -> None:
    """
    Print the measurements for both parts.

    ``steps`` is the Adam steps per training run, ``seeds`` the number of runs
    per model, and ``trials`` the random softmax rows per row of the table.

    Examples
    --------
    >>> demo(steps=40, seeds=1, trials=20)  # doctest: +ELLIPSIS
    ═...
    1. WHY DIVIDE BY sqrt(d_k)
    ...
    2. TWO WAYS TO SCORE A KEY
    ...
    3. WHAT THE SCORES ARE FOR
    ...
    """
    print("═" * 70)
    print(f"  {LESSON_ID.upper().replace('_', ' ')}")
    print("═" * 70)
    n_keys = 64

    _heading("1. WHY DIVIDE BY sqrt(d_k)")
    print("Unit-variance q and k; softmax over 64 keys.")
    print(f"Uniform weights: max weight {1 / n_keys:.4f}, entropy {np.log(n_keys):.3f} nats.")
    print("jacobian = size of the gradient the softmax passes back to the scores.")
    print()
    for d_k in (4, 16, 64, 256, 1024):
        print(f"  d_k {d_k:>4}: var(q.k) {dot_variance(d_k, 10 * trials):8.1f}")
        for scale in (False, True):
            p = softmax_weights(d_k, scale, n_keys, trials)
            top, entropy = saturation(d_k, scale, n_keys, trials)
            tag = "scaled  " if scale else "unscaled"
            print(
                f"           {tag}: max weight {top:.3f}  entropy {entropy:.3f}"
                f"  jacobian {jacobian_norm(p):.3f}"
            )

    _heading("2. TWO WAYS TO SCORE A KEY")
    rng = np.random.default_rng(0)
    d, h, t = 16, 16, 12
    q, K, target = make_task(6, t, 4, rng, hidden_map(4), 0.1)
    print("Gradient check, largest gap from central differences:")
    for kind in KINDS[:2]:
        w = init_params(kind, 4, 5, rng)
        print(f"  {kind:>9}: {gradient_error(kind, w, q, K, target):.2e}")
    print()
    print(f"Alignment: the query is a hidden linear map of one of {t} keys plus noise.")
    print(f"d={d}, hidden {h}, {steps} Adam steps, {seeds} seeds; chance is {1 / t:.3f}.")
    print()
    for kind in KINDS:
        runs = [train(kind, seed, steps, d, h, t) for seed in range(seeds)]
        accs = [r.accuracy for r in runs]
        print(
            f"  {kind:>9}: accuracy {np.mean(accs):.3f} (min {min(accs):.3f}, max {max(accs):.3f})"
            f"  loss {np.mean([r.loss for r in runs]):.3f}"
            f"  time per run {np.mean([r.seconds for r in runs]):.2f} s"
        )

    _heading("3. WHAT THE SCORES ARE FOR")
    print("attend(scores, keys) mixes the keys by their weights. Mean cosine between")
    print("that mix and the target key, on held-out queries (1.0 is a perfect copy):")
    print()
    for kind in KINDS:
        r = train(kind, 0, steps, d, h, t)
        print(f"  {kind:>9}: cosine {r.cosine:.3f}")
    print()
    print("═" * 70)
    print("  LESSON COMPLETE")
    print("═" * 70)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_doctests(module: ModuleType | None = None, verbose: bool = False) -> doctest.TestResults:
    r"""
    Run a module's doctests with the option flags pytest uses.

    ``pyproject.toml`` gives pytest ``ELLIPSIS`` and ``NORMALIZE_WHITESPACE``;
    passing the same flags here keeps an example's verdict the same under both.

    Parameters
    ----------
    module : ModuleType or None
        Module to test; ``None`` tests ``__main__``, as ``doctest.testmod`` does.
    verbose : bool
        Report each example rather than only failures.

    Returns
    -------
    doctest.TestResults
        Counts of failed and attempted examples.

    Examples
    --------
    Output that differs only in spacing, or is elided with ``...``, passes:

    >>> import types
    >>> doc = ">>> print('a  b')\na b\n>>> print('abc')\na...\n"
    >>> results = run_doctests(types.ModuleType("flags_demo", doc))
    >>> results.failed, results.attempted
    (0, 2)
    """
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    return doctest.testmod(module, verbose=verbose, optionflags=flags)


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
    args = parse_args(argv)
    if args.show_index:
        show_lesson_index()
        return 0
    if args.doctest:
        results = run_doctests(verbose=args.verbose)
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
