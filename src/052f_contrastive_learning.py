#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<4.0"
# dependencies = ["numpy>=2.5.4"]
# ///
"""
Lesson 5.2f: Contrastive Learning, CLIP's Mechanism at Toy Scale (side quest).

You train two encoders, one for an "image" view and one for a "text" view of the
same hidden concepts, so that matching pairs score high on a dot product and
mismatched pairs score low. The loss is InfoNCE: a softmax over each row of the
batch's similarity matrix, with every pair that shares a concept counting as a
positive. You derive both encoders' gradients by hand through the ReLU and check
them against central differences. Afterwards you ask two questions of held-out
pairs: does an image retrieve a text of its concept, and do the same embedding
units prefer the same concept in both views?

This toy keeps CLIP's contrastive mechanism and the cross-view units it
produces. It does not have CLIP's scale, its image and text towers, or its real
neurons.

Source: Goh et al., "Multimodal Neurons in Artificial Neural Networks", Distill
(2021), https://distill.pub/2021/multimodal-neurons/ and Radford et al.,
"Learning Transferable Visual Models From Natural Language Supervision" (CLIP,
2021), https://arxiv.org/abs/2103.00020

Prerequisites (must complete first)
----------------------------------
- Lesson 3.1: The Dot Product (similarity is a dot product of embeddings)
- Lesson 2.3: Softmax (each row of similarities becomes a distribution)
- Lesson 5.2a: Toy Models of Superposition (hand-derived gradients checked by
  finite differences)

Learning Objectives
-------------------
After completing this lesson, you will be able to:
1. Write the symmetric InfoNCE loss over a batch's similarity matrix, with soft
   targets when several pairs share a concept.
2. Derive its gradient for both encoders through the ReLU, and check it against
   central differences, seeing the check fail when one half of the loss or the
   ReLU mask is missing.
3. Measure cross-view retrieval and cross-view units against an untrained
   control, so you know what the training added.

The Big Idea
------------
You never tell the encoders what the concepts are. You tell them only which
image and text belong together. To make matching pairs score high and every
other pair score low, both encoders must send a concept to the same region of
a shared embedding space, from two different-looking views. Units that fire for
one concept in both views are what that agreement looks like when you read the
embedding one unit at a time.

Why It Matters
--------------
CLIP trains an image encoder and a text encoder this way on hundreds of
millions of captioned images. Goh et al. read the trained network's units and
find single units that respond to the same concept as a photo, a drawing, and a
written word. Embedding models for search and retrieval are trained with the
same loss (Lesson 10.4). This lesson keeps only the mechanism, so you can see
the loss, its gradient, and an untrained control side by side.

Stdlib Pain → Library Power-Up
------------------------------
Pain (this lesson):
- You write the loss over a B by B matrix, derive its gradient by hand for two
  encoders, and apply the ReLU mask on both sides. Dropping one half of the
  symmetric loss or the mask trains silently to a worse model.
- Lines of code: ~20 for the loss and gradient, ~15 for Adam.
- Runtime: about 35 seconds of CPU time for the demo (3,000 steps of batch
  256), measured with one BLAS thread; 90 seconds of wall time on a busy
  machine.
- Common bugs: transposing the second half of the loss, normalizing the
  targets by column instead of by row, and differentiating through the ReLU
  where it is inactive.

Power-Up (Lesson 5.3 — PyTorch):
- ``loss.backward()`` produces both encoders' gradients from the forward pass,
  and ``torch.optim.AdamW`` replaces the hand-written update.
- Key APIs: ``torch.autograd``, ``torch.nn.functional.cross_entropy``,
  ``torch.optim.AdamW``.

Complexity Analysis
-------------------
Pain (manual), for batch B, input dimension d, embedding size e, k steps:
- Time:  O(k·(B·d·e + B²·e)).
- Space: O(B² + B·e + d·e).

Execution
---------
Project-style:
    uv run python src/052f_contrastive_learning.py

Script-style (self-contained deps):
    uv run --script src/052f_contrastive_learning.py

On a busy machine, NumPy's BLAS threads compete with other work and can stretch
the demo; one thread avoids that, since the matrices here are small:
    OPENBLAS_NUM_THREADS=1 uv run python src/052f_contrastive_learning.py

Options:
    python src/052f_contrastive_learning.py --doctest-only
    python src/052f_contrastive_learning.py --no-doctest
    python src/052f_contrastive_learning.py --index
    python src/052f_contrastive_learning.py -v

Quality Gates
-------------
    uv run pytest src/052f_contrastive_learning.py
    uv run ruff check src/052f_contrastive_learning.py
    uv run ruff format --check src/052f_contrastive_learning.py
    uv run mypy --strict src/052f_contrastive_learning.py

Module Doctests
---------------
When every embedding is the same, every similarity is the same, each row's
softmax is uniform, and with one positive among four pairs the loss is
ln 4 ≈ 1.386. The row sums of the gradient's similarity term are zero, so
nothing moves:

>>> import numpy as np
>>> ones = np.ones((4, 1))
>>> loss, grads = infonce_loss_and_grads((ones[:1], ones[:1]), ones, ones, np.arange(4), 0.2)
>>> bool(np.isclose(loss, np.log(4))), [float(np.abs(g).max()) for g in grads]
(True, [0.0, 0.0])
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
    type Labels = NDArray[np.int64]
    type Params = tuple[Array, Array]
    type GradFn = Callable[[Params, Array, Array, Labels, float], tuple[float, Params]]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON IDENTITY & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON_ID: Final[str] = "tier_5_lesson_2f_contrastive_learning"

# Autograd and torch.optim.AdamW arrive with PyTorch.
POWER_UP_AVAILABLE: Final[bool] = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Format: lesson_id -> reason

PREREQUISITES: Final[Mapping[str, str]] = {
    "tier_3_lesson_1_dot_product": "Similarity between embeddings is a dot product",
    "tier_2_lesson_3_softmax": "Each row of similarities becomes a distribution",
    "tier_5_lesson_2a_superposition": "Hand-derived gradients checked by finite differences",
}

DEPENDENCIES: Final[Mapping[str, str]] = {
    "tier_2_lesson_4_cross_entropy": "InfoNCE is cross-entropy over a row of similarities",
}

SUCCESSORS: Final[Mapping[str, str]] = {
    "tier_10_lesson_4_embeddings": "Embedding models are trained with this loss",
}

DEPENDENTS: Final[Mapping[str, str]] = {}

RELATED: Final[Mapping[str, str]] = {
    "tier_5_lesson_2d_reading_a_unit": "Reads a unit's preferred input directly",
}

POWER_UP_IN_LESSON: Final[str | None] = "tier_5_lesson_3_pytorch"
PAIN_FROM_LESSON: Final[str | None] = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE TOY: Two Views of Eight Hidden Concepts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True)
class Views:
    """
    The fixed random maps that turn a hidden concept into an image and a text view.

    Attributes
    ----------
    proto_img : numpy.ndarray
        ``(k, dim)`` prototype of each of the k concepts in the image view.
    proto_txt : numpy.ndarray
        ``(k, dim)`` prototypes in the text view; unrelated to ``proto_img``.
    mix_img : numpy.ndarray
        ``(dim, dim)`` mixing matrix applied to the image view.
    mix_txt : numpy.ndarray
        ``(dim, dim)`` mixing matrix applied to the text view.

    Examples
    --------
    >>> import numpy as np
    >>> v = Views(np.zeros((8, 32)), np.ones((8, 32)), np.eye(32), np.eye(32))
    >>> v.proto_img.shape, float(v.proto_txt[0, 0])
    ((8, 32), 1.0)
    """

    proto_img: Array
    proto_txt: Array
    mix_img: Array
    mix_txt: Array


def make_views(rng: np.random.Generator, k: int = 8, dim: int = 32) -> Views:
    """
    Draw prototypes and mixing matrices for ``k`` concepts in ``dim`` dimensions.

    Each view has its own standard-normal prototypes and its own mixing matrix
    scaled by 1/√dim, so the two views of one concept share no coordinates. The
    draw order is image prototypes, text prototypes, image mix, text mix.

    Examples
    --------
    >>> import numpy as np
    >>> a = make_views(np.random.default_rng(0), k=4, dim=6)
    >>> b = make_views(np.random.default_rng(0), k=4, dim=6)
    >>> a.proto_img.shape, a.mix_txt.shape
    ((4, 6), (6, 6))
    >>> bool(np.array_equal(a.proto_img, b.proto_img))
    True
    >>> bool(np.array_equal(a.proto_img, a.proto_txt))
    False
    """
    proto_img = rng.normal(size=(k, dim))
    proto_txt = rng.normal(size=(k, dim))
    mix_img = rng.normal(size=(dim, dim)) / np.sqrt(dim)
    mix_txt = rng.normal(size=(dim, dim)) / np.sqrt(dim)
    return Views(proto_img, proto_txt, mix_img, mix_txt)


def draw_pairs(
    views: Views,
    rng: np.random.Generator,
    concepts: Labels,
    noise: float = 0.5,
) -> tuple[Array, Array]:
    """
    Draw one image view and one text view for each concept in ``concepts``.

    A view is tanh((prototype + ``noise`` · N(0, 1)) · mix), with fresh noise
    for every sample and every view, so two samples of one concept differ.

    Examples
    --------
    Without noise, two samples of one concept are identical in each view, and
    every value lies strictly between -1 and 1:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> views = make_views(rng, k=4, dim=6)
    >>> img, txt = draw_pairs(views, rng, np.array([3, 1, 3]), noise=0.0)
    >>> img.shape, txt.shape
    ((3, 6), (3, 6))
    >>> bool(np.array_equal(img[0], img[2])), bool(np.array_equal(img[0], img[1]))
    (True, False)
    >>> bool(np.abs(img).max() < 1.0)
    True
    """
    n, dim = len(concepts), views.proto_img.shape[1]
    X_img = views.proto_img[concepts] + noise * rng.normal(size=(n, dim))
    X_txt = views.proto_txt[concepts] + noise * rng.normal(size=(n, dim))
    return np.tanh(X_img @ views.mix_img), np.tanh(X_txt @ views.mix_txt)


def encode(X: Array, W: Array) -> Array:
    """
    Return the embedding ReLU(X W), one row per sample.

    Examples
    --------
    The pre-activation (1, -2) becomes (1, 0):

    >>> import numpy as np
    >>> encode(np.array([[1.0, -1.0]]), np.array([[1.0, -1.0], [0.0, 1.0]])).tolist()
    [[1.0, 0.0]]
    """
    return np.asarray(np.maximum(X @ W, 0.0), dtype=np.float64)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE PAIN: Symmetric InfoNCE With Hand-Derived Gradients
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def softmax_rows(L: Array) -> Array:
    """
    Return the softmax of each row of ``L``.

    The row maximum is subtracted first, so large entries do not overflow.

    Examples
    --------
    >>> import numpy as np
    >>> softmax_rows(np.zeros((2, 2))).tolist()
    [[0.5, 0.5], [0.5, 0.5]]
    >>> softmax_rows(np.array([[1000.0, 1000.0]])).tolist()
    [[0.5, 0.5]]
    """
    exps = np.exp(L - L.max(axis=1, keepdims=True))
    return np.asarray(exps / exps.sum(axis=1, keepdims=True), dtype=np.float64)


def same_concept_targets(concepts: Labels) -> Array:
    """
    Return the matrix whose row i spreads probability 1 over the samples sharing sample i's concept.

    Entry (i, j) is 1/n_i when ``concepts[i] == concepts[j]`` and n_i is the
    number of such j, so each row sums to 1. The matrix is the soft target for
    a row of the similarity matrix.

    Examples
    --------
    >>> import numpy as np
    >>> same_concept_targets(np.array([0, 1, 0])).tolist()
    [[0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]
    """
    same = (concepts[:, None] == concepts[None, :]).astype(np.float64)
    return np.asarray(same / same.sum(axis=1, keepdims=True), dtype=np.float64)


def infonce_loss_and_grads(
    params: Params,
    X_img: Array,
    X_txt: Array,
    concepts: Labels,
    tau: float,
) -> tuple[float, Params]:
    r"""
    Return the symmetric InfoNCE loss and its gradients for ``(W_img, W_txt)``.

    With embeddings E = ReLU(X W) and similarities S = E_img E_txtᵀ / τ, let T
    be ``same_concept_targets(concepts)``. The loss is the mean over rows of
    the cross-entropy between softmax(S) and T (image to text), plus the same
    over rows of Sᵀ (text to image), halved. Backward, with B samples::

        D  = ((softmax(S) - T) + (softmax(Sᵀ) - T)ᵀ) / (2B)      = ∂loss/∂S
        ∂loss/∂E_img = D E_txt / τ        ∂loss/∂E_txt = Dᵀ E_img / τ
        grad_W = Xᵀ (∂loss/∂E · (X W > 0))                        for each view

    The image-to-text half contributes (softmax(S) - T) and the text-to-image
    half contributes its transpose; dropping either leaves half the gradient.

    Examples
    --------
    Two samples, two dimensions, identity weights and τ = 1: S = I, the
    softmax of each row is (e, 1)/(1 + e) in turn, and the loss is
    ln(1 + e⁻¹) in both halves. With q = 1/(1 + e), D = ½ [[-q, q], [q, -q]],
    and only the diagonal of the ReLU mask is open, so grad_W = diag(-q/2):

    >>> import numpy as np
    >>> I = np.eye(2)
    >>> loss, (g_img, g_txt) = infonce_loss_and_grads((I, I), I, I, np.array([0, 1]), 1.0)
    >>> q = 1 / (1 + np.e)
    >>> bool(np.isclose(loss, np.log1p(np.exp(-1.0))))
    True
    >>> bool(np.allclose(g_img, np.diag([-q / 2, -q / 2]))), bool(np.allclose(g_img, g_txt))
    (True, True)

    When every sample shares one concept the targets are uniform. Identical
    embeddings then match the targets exactly: the loss is ln B, the entropy of
    a uniform row, and the gradient is zero:

    >>> ones, same = np.ones((3, 1)), np.zeros(3, int)
    >>> loss, grads = infonce_loss_and_grads((ones[:1], ones[:1]), ones, ones, same, 0.5)
    >>> bool(np.isclose(loss, np.log(3))), [float(np.abs(g).max()) for g in grads]
    (True, [0.0, 0.0])

    A weight that turns every unit off gives zero embeddings, so the ReLU mask
    blocks every gradient while the loss stays ln B:

    >>> neg, X = -np.ones((1, 2)), np.ones((4, 1))
    >>> loss, grads = infonce_loss_and_grads((neg, neg), X, X, np.arange(4), 0.2)
    >>> bool(np.isclose(loss, np.log(4))), [float(np.abs(g).max()) for g in grads]
    (True, [0.0, 0.0])
    """
    W_img, W_txt = params
    H_img, H_txt = X_img @ W_img, X_txt @ W_txt
    E_img, E_txt = np.maximum(H_img, 0.0), np.maximum(H_txt, 0.0)
    S = E_img @ E_txt.T / tau
    T = same_concept_targets(concepts)
    P_img, P_txt = softmax_rows(S), softmax_rows(S.T)
    B = len(concepts)
    ce_img = -np.sum(T * np.log(P_img + 1e-300)) / B
    ce_txt = -np.sum(T * np.log(P_txt + 1e-300)) / B
    D = ((P_img - T) + (P_txt - T).T) / (2 * B)
    grad_img = X_img.T @ (D @ E_txt / tau * (H_img > 0))
    grad_txt = X_txt.T @ (D.T @ E_img / tau * (H_txt > 0))
    return float((ce_img + ce_txt) / 2), (grad_img, grad_txt)


def gradient_error(
    grad_fn: GradFn,
    params: Params,
    X_img: Array,
    X_txt: Array,
    concepts: Labels,
    tau: float,
    eps: float = 1e-5,
) -> float:
    """
    Return the largest gap between ``grad_fn``'s gradients and central differences.

    ``grad_fn`` has the signature of ``infonce_loss_and_grads``. Each entry of
    each weight matrix is nudged by ±``eps`` and the loss from ``grad_fn`` is
    differenced. An exact gradient gives an error near ``eps``², a wrong one
    gives an error of the size of the missing term.

    Examples
    --------
    The hand-derived gradient agrees with the numeric one, on a batch where
    concepts repeat so the targets are soft:

    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> params = (rng.normal(size=(5, 4)), rng.normal(size=(5, 4)))
    >>> X_img, X_txt = rng.normal(size=(6, 5)), rng.normal(size=(6, 5))
    >>> concepts = np.array([0, 1, 2, 0, 1, 2])
    >>> gradient_error(infonce_loss_and_grads, params, X_img, X_txt, concepts, 0.5) < 1e-6
    True

    The deliberate break, first: a gradient that keeps only the image-to-text
    half of the symmetric loss is far off.

    >>> def one_half(params, X_img, X_txt, concepts, tau):
    ...     loss, _ = infonce_loss_and_grads(params, X_img, X_txt, concepts, tau)
    ...     H_img, H_txt = X_img @ params[0], X_txt @ params[1]
    ...     E_img, E_txt = np.maximum(H_img, 0), np.maximum(H_txt, 0)
    ...     S = E_img @ E_txt.T / tau
    ...     D = (softmax_rows(S) - same_concept_targets(concepts)) / (2 * len(concepts))
    ...     return loss, (
    ...         X_img.T @ (D @ E_txt / tau * (H_img > 0)),
    ...         X_txt.T @ (D.T @ E_img / tau * (H_txt > 0)),
    ...     )
    >>> gradient_error(one_half, params, X_img, X_txt, concepts, 0.5) > 0.01
    True

    And second: a gradient that forgets the ReLU mask ``(X W > 0)``.

    >>> def no_mask(params, X_img, X_txt, concepts, tau):
    ...     loss, _ = infonce_loss_and_grads(params, X_img, X_txt, concepts, tau)
    ...     E_img, E_txt = np.maximum(X_img @ params[0], 0), np.maximum(X_txt @ params[1], 0)
    ...     S, T = E_img @ E_txt.T / tau, same_concept_targets(concepts)
    ...     D = ((softmax_rows(S) - T) + (softmax_rows(S.T) - T).T) / (2 * len(concepts))
    ...     return loss, (X_img.T @ (D @ E_txt / tau), X_txt.T @ (D.T @ E_img / tau))
    >>> gradient_error(no_mask, params, X_img, X_txt, concepts, 0.5) > 0.01
    True
    """
    _, analytic = grad_fn(params, X_img, X_txt, concepts, tau)
    worst = 0.0
    for k, p in enumerate(params):
        for index in np.ndindex(*p.shape):
            losses = []
            for sign in (1.0, -1.0):
                shifted = [q.copy() for q in params]
                shifted[k][index] += sign * eps
                moved: Params = (shifted[0], shifted[1])
                losses.append(grad_fn(moved, X_img, X_txt, concepts, tau)[0])
            numeric = (losses[0] - losses[1]) / (2 * eps)
            worst = max(worst, abs(float(analytic[k][index]) - numeric))
    return worst


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


def train(
    views: Views,
    rng: np.random.Generator,
    emb: int = 16,
    steps: int = 3000,
    batch: int = 256,
    lr: float = 3e-3,
    tau: float = 0.2,
    decay: float = 1e-4,
) -> tuple[Params, Params]:
    """
    Train both encoders with Adam and return ``(initial, trained)`` weights.

    ``rng`` draws the initial weights (normal, scale 0.1), then a fresh batch
    of concepts and pairs per step, so a run repeats exactly. Weight decay adds
    ``decay`` · W to each gradient. The initial weights come back as a copy: they
    are the untrained control.

    Examples
    --------
    With no steps, the trained weights are the initial weights. With some
    steps, the loss on a fresh batch falls:

    >>> import numpy as np
    >>> def fresh_loss(weights):
    ...     rng = np.random.default_rng(7)
    ...     views = make_views(np.random.default_rng(0), k=4, dim=8)
    ...     c = rng.integers(0, 4, 64)
    ...     X_img, X_txt = draw_pairs(views, rng, c)
    ...     return infonce_loss_and_grads(weights, X_img, X_txt, c, 0.2)[0]
    >>> rng = np.random.default_rng(0)
    >>> views = make_views(rng, k=4, dim=8)
    >>> start, same = train(views, rng, emb=8, steps=0)
    >>> bool(np.array_equal(start[0], same[0]))
    True
    >>> rng = np.random.default_rng(0)
    >>> views = make_views(rng, k=4, dim=8)
    >>> start, trained = train(views, rng, emb=8, steps=150, batch=64)
    >>> trained[0].shape, trained[1].shape
    ((8, 8), (8, 8))
    >>> fresh_loss(trained) < fresh_loss(start)
    True
    """
    k, dim = views.proto_img.shape
    params = [rng.normal(0.0, 0.1, (dim, emb)), rng.normal(0.0, 0.1, (dim, emb))]
    start: Params = (params[0].copy(), params[1].copy())
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    for t in range(1, steps + 1):
        concepts = rng.integers(0, k, batch)
        X_img, X_txt = draw_pairs(views, rng, concepts)
        _, grads = infonce_loss_and_grads((params[0], params[1]), X_img, X_txt, concepts, tau)
        decayed = [g + decay * p for g, p in zip(grads, params, strict=True)]
        adam_step(params, decayed, first, second, t, lr)
    return start, (params[0], params[1])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MEASURING: Retrieval, Cross-View Units, ASCII Bars
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def retrieval_accuracy(
    E_img: Array,
    E_txt: Array,
    concepts: Labels,
    queries: int | None = None,
) -> float:
    """
    Return the share of image queries whose best-scoring text has the query's concept.

    Each of the first ``queries`` images (all by default) is scored against
    every text by dot product. A hit is a best-scoring text of the same
    concept; the text need not be the paired one.

    Examples
    --------
    One-hot embeddings that agree across views retrieve perfectly; swapping the
    text rows of two concepts retrieves none of those two:

    >>> import numpy as np
    >>> E = np.eye(3)
    >>> concepts = np.array([0, 1, 2])
    >>> retrieval_accuracy(E, E, concepts)
    1.0
    >>> retrieval_accuracy(E, E[[1, 0, 2]], concepts)
    0.3333333333333333
    >>> retrieval_accuracy(E, E, concepts, queries=1)
    1.0
    """
    n = len(concepts) if queries is None else queries
    best = (E_img[:n] @ E_txt.T).argmax(axis=1)
    return float(np.mean(concepts[best] == concepts[:n]))


def concept_means(E: Array, concepts: Labels, k: int) -> Array:
    """
    Return the ``(k, units)`` mean activation of each unit over the samples of each concept.

    A concept with no samples gets a row of zeros.

    Examples
    --------
    >>> import numpy as np
    >>> E = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
    >>> concept_means(E, np.array([0, 0, 1]), 3).tolist()
    [[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]
    """
    onehot = (concepts[:, None] == np.arange(k)[None, :]).astype(np.float64)
    counts = np.maximum(onehot.sum(axis=0), 1.0)
    return np.asarray(onehot.T @ E / counts[:, None], dtype=np.float64)


def cross_view_units(
    means_img: Array,
    means_txt: Array,
    threshold: float = 1e-3,
) -> tuple[list[int], int]:
    """
    Return the units that prefer the same concept in both views, and the count of active units.

    A unit is active when its largest mean activation exceeds ``threshold`` in
    both views. A unit prefers the concept with the largest mean in a view. The
    arguments are ``(k, units)`` matrices from ``concept_means``.

    Examples
    --------
    Unit 0 prefers concept 0 in both views. Unit 1 never fires in the text
    view and unit 2 in neither, so both are inactive. Unit 3 is active but
    prefers concept 0 in one view and concept 1 in the other:

    >>> import numpy as np
    >>> img = np.array([[1.0, 0.0, 0.0, 3.0], [0.0, 2.0, 0.0, 0.0]])
    >>> txt = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]])
    >>> cross_view_units(img, txt)
    ([0], 2)
    """
    active = (means_img.max(axis=0) > threshold) & (means_txt.max(axis=0) > threshold)
    agree = means_img.argmax(axis=0) == means_txt.argmax(axis=0)
    return [int(u) for u in np.flatnonzero(agree & active)], int(active.sum())


def unit_chart(means_img: Array, means_txt: Array, unit: int, ramp: str = " .:-=+*#%@") -> str:
    """
    Draw one unit's mean activation per concept in each view as a row of shade characters.

    Both rows share one scale, the unit's largest mean in either view, so a
    row that is blank means the unit is quiet in that view. Position i is
    concept i.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.array([[0.0], [10.0], [3.0]])
    >>> txt = np.array([[10.0], [0.0], [0.0]])
    >>> print(unit_chart(img, txt, 0))
    unit 0  img | @-|
            txt |@  |
    """
    top = max(means_img[:, unit].max(), means_txt[:, unit].max())

    def row(means: Array) -> str:
        levels = np.round(means[:, unit] / top * (len(ramp) - 1)).astype(int)
        return "".join(ramp[i] for i in levels)

    return f"unit {unit:<2} img |{row(means_img)}|\n{'':8}txt |{row(means_txt)}|"


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
      LESSON INDEX: tier_5_lesson_2f_contrastive_learning
    ...
    Prerequisites (hard deps):
       → tier_3_lesson_1_dot_product: Similarity between embeddings is a dot product
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
    k: int = 8,
    dim: int = 32,
    emb: int = 16,
    steps: int = 3000,
    held_out: int = 2000,
    queries: int = 500,
    shown: int = 3,
    seed: int = 0,
) -> None:
    """
    Train both encoders, then compare them with the untrained start on held-out pairs.

    Prints held-out image-to-text retrieval accuracy and the share of active
    units that prefer the same concept in both views, for the trained encoders
    and for the untrained control, as numbers and bars. Then it draws the
    per-concept mean activation of the ``shown`` most active cross-view units.
    The untrained encoders are random ReLU projections, so they retrieve above
    chance and share a few preferences by accident; the trained encoders
    retrieve every held-out concept correctly in the measured run, and more
    units agree.

    Examples
    --------
    >>> demo(k=4, dim=8, emb=8, steps=60, held_out=200, queries=50, shown=1)  # doctest: +ELLIPSIS
    ═...
    Held-out image-to-text retrieval accuracy (chance 0.25):
    ...
    Active units preferring the same concept in both views:
    ...
    """
    print("═" * 70)
    print(f"  {LESSON_ID.upper().replace('_', ' ')}")
    print("═" * 70)
    print()
    rng = np.random.default_rng(seed)
    views = make_views(rng, k, dim)
    start, trained = train(views, rng, emb=emb, steps=steps)
    concepts = rng.integers(0, k, held_out)
    X_img, X_txt = draw_pairs(views, rng, concepts)
    results = {}
    for name, (W_img, W_txt) in (("trained", trained), ("untrained", start)):
        E_img, E_txt = encode(X_img, W_img), encode(X_txt, W_txt)
        means_img, means_txt = concept_means(E_img, concepts, k), concept_means(E_txt, concepts, k)
        agreeing, active = cross_view_units(means_img, means_txt)
        results[name] = (
            retrieval_accuracy(E_img, E_txt, concepts, queries),
            agreeing,
            active,
            means_img,
            means_txt,
        )
    print(f"Trained {steps} steps; held-out pairs: {held_out}; embedding units: {emb}.")
    print()
    print(f"Held-out image-to-text retrieval accuracy (chance {1 / k:.2f}):")
    print(bars([(name, r[0]) for name, r in results.items()], width=30, top=1.0))
    print()
    print("Active units preferring the same concept in both views:")
    for name, (_, agreeing, active, _, _) in results.items():
        print(f"  {name:<9} {len(agreeing)} of {active} active units")
    share = [(name, len(r[1]) / max(r[2], 1)) for name, r in results.items()]
    print(bars(share, width=30, top=1.0))
    print()
    _, agreeing, _, means_img, means_txt = results["trained"]
    strength = means_img.max(axis=0) + means_txt.max(axis=0)
    most_active = sorted(agreeing, key=lambda u: -float(strength[u]))
    print(f"Mean activation of the {min(shown, len(most_active))} most active cross-view units,")
    print(f"one character per concept (0 to {k - 1}), image row over text row:")
    for unit in most_active[:shown]:
        print(unit_chart(means_img, means_txt, unit))


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
