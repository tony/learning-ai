#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<4.0"
# dependencies = ["numpy>=2.5.4"]
# ///
"""
Lesson 5.2c: A Tiny Convolutional Network (side quest).

You train a convolutional network with one layer of 5x5 kernels, with every
gradient derived by hand, to tell lines at four angles apart. Nothing in the
loss mentions edges or orientation, yet the kernels it ends up with are line
detectors: each one responds strongly to gratings at some angles and weakly to
others, which you measure.

Source: Olah et al., "An Overview of Early Vision in InceptionV1", Distill
(2020), https://distill.pub/2020/circuits/early-vision/ (trained networks'
first layers learn oriented edge and line detectors; this lesson watches that
happen in a tiny network).

Prerequisites (must complete first)
----------------------------------
- Lesson 3.4a: Convolution as a Sliding Dot Product (the operation each kernel runs)
- Lesson 5.2a: Toy Models of Superposition (hand-derived gradients checked by
  finite differences)

Learning Objectives
-------------------
After completing this lesson, you will be able to:
1. Turn a convolution into one matrix product by listing every window as a row.
2. Derive the gradients of convolution, ReLU, average pooling, and softmax
   cross-entropy by hand, and see a central-difference check fail when one is
   wrong.
3. Measure how orientation-selective each learned kernel is.

The Big Idea
------------
A convolutional layer is a linear layer whose weights are shared across
positions, so its gradient is the sum of the gradients from every position.
Trained to separate line angles, the shared kernels become detectors for those
angles.

Why It Matters
--------------
InceptionV1's first layer, the subject of Distill's early-vision article,
contains oriented edge and line detectors that no one designed. Watching 8
kernels do the same on 16x16 images shows where those detectors come from: the
task rewards them.

Stdlib Pain → Library Power-Up
------------------------------
Pain (this lesson):
- You derive the gradient through the shared kernels on paper, then write it as
  about ten lines of NumPy; a missing ReLU mask or a pooled mean treated as a
  sum trains silently to a worse answer.
- Lines of code: ~15 for the gradient, ~15 for Adam.
- Runtime: about 15 seconds for the demo (1,500 steps of 128 images), measured with
  one BLAS thread.
- Common bugs: forgetting that each kernel weight collects a gradient from
  every position, and forgetting the 1/positions of the average pool.

Power-Up (Lesson 5.3 — PyTorch):
- ``nn.Conv2d`` holds the kernels and ``loss.backward()`` produces every
  gradient above from the forward pass.
- Key APIs: ``torch.nn.Conv2d``, ``torch.autograd``.

Complexity Analysis
-------------------
Pain (manual), for batch B, P window positions, k x k kernels, C channels,
k_steps steps:
- Time:  O(k_steps·B·P·k²·C).
- Space: O(B·P·(k² + C)) for the windows and activations of one batch.

Execution
---------
Project-style:
    uv run python src/052c_tiny_cnn.py

Script-style (self-contained deps):
    uv run --script src/052c_tiny_cnn.py

On a busy machine, NumPy's BLAS threads compete with other work and can stretch
the demo from about 15 seconds to a minute or more; one thread avoids that, since the
matrices here are small:
    OPENBLAS_NUM_THREADS=1 uv run python src/052c_tiny_cnn.py

Options:
    python src/052c_tiny_cnn.py --doctest-only
    python src/052c_tiny_cnn.py --no-doctest
    python src/052c_tiny_cnn.py --index
    python src/052c_tiny_cnn.py -v

Quality Gates
-------------
    uv run pytest src/052c_tiny_cnn.py
    uv run ruff check src/052c_tiny_cnn.py
    uv run ruff format --check src/052c_tiny_cnn.py
    uv run mypy --strict src/052c_tiny_cnn.py

Module Doctests
---------------
One 1x1 image with pixel 2, a 1x1 kernel of weight 1, and zero everywhere else:
the logits are all 0, so the loss is ln 4 and the softmax error on the true
class 0 is 1/4 - 1 = -3/4. The pooled activation is 2, so the gradient of the
linear weights is 2 times that error:

>>> import numpy as np
>>> params = (np.ones((1, 1)), np.zeros(1), np.zeros((1, 4)), np.zeros(4))
>>> loss, grads = loss_and_grads(params, np.full((1, 1, 1), 2.0), np.array([0]))
>>> round(float(loss), 4), grads[2].tolist()
(1.3863, [[-1.5, 0.5, 0.5, 0.5]])
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
    type Params = tuple[Array, Array, Array, Array]
    type GradFn = Callable[[Params, Array, Labels], tuple[float, Params]]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON IDENTITY & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON_ID: Final[str] = "tier_5_lesson_2c_tiny_cnn"

# nn.Conv2d and autograd arrive with PyTorch.
POWER_UP_AVAILABLE: Final[bool] = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Format: lesson_id -> reason

PREREQUISITES: Final[Mapping[str, str]] = {
    "tier_3_lesson_4a_convolution": "The sliding dot product each kernel computes",
    "tier_5_lesson_2a_superposition": "Hand-derived gradients checked by finite differences",
}

DEPENDENCIES: Final[Mapping[str, str]] = {
    "tier_2_lesson_4_cross_entropy": "The softmax cross-entropy loss",
}

SUCCESSORS: Final[Mapping[str, str]] = {
    "tier_5_lesson_2d_reading_a_unit": "Reads this network's units",
}

DEPENDENTS: Final[Mapping[str, str]] = {}

RELATED: Final[Mapping[str, str]] = {
    "tier_4_lesson_2a_momentum": "Adam builds on momentum",
}

POWER_UP_IN_LESSON: Final[str | None] = "tier_5_lesson_3_pytorch"
PAIN_FROM_LESSON: Final[str | None] = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE PAIN: A Convolutional Network With Hand-Derived Gradients
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANGLES: Final[tuple[float, ...]] = (0.0, 45.0, 90.0, 135.0)
SIZE: Final[int] = 16


def line_dataset(n: int, seed: int = 0, noise: float = 0.1) -> tuple[Array, Labels]:
    """
    Return ``n`` images of a soft line segment, and the index into ``ANGLES`` of each line's angle.

    Each image is ``SIZE`` x ``SIZE``. The line has a Gaussian profile across
    it, a random centre and a random half-length, plus Gaussian pixel noise of
    standard deviation ``noise``. The seed fixes the whole dataset.

    Examples
    --------
    >>> images, labels = line_dataset(8, seed=1)
    >>> images.shape, labels.shape
    ((8, 16, 16), (8,))
    >>> bool(((labels >= 0) & (labels < 4)).all())
    True
    >>> again, _ = line_dataset(8, seed=1)
    >>> bool((images == again).all())
    True

    Without noise a pixel is at most the line's peak of 1 and at least 0:

    >>> clean, _ = line_dataset(50, seed=2, noise=0.0)
    >>> float(clean.min()), float(clean.max()) <= 1.0
    (0.0, True)
    """
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(SIZE, dtype=np.float64), np.arange(SIZE, dtype=np.float64))
    labels = rng.integers(0, len(ANGLES), n)
    images = np.empty((n, SIZE, SIZE))
    for i, label in enumerate(labels):
        theta = np.deg2rad(ANGLES[label])
        cy, cx = rng.uniform(5, 11, 2)
        along = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
        across = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
        images[i] = np.exp(-(across**2)) * (np.abs(along) < rng.uniform(3, 7))
    images += rng.normal(0.0, noise, images.shape)
    return np.asarray(images, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def patches(images: Array, k: int = 5) -> Array:
    """
    Return every ``k`` x ``k`` window of each image as a row, shape ``(n, positions, k*k)``.

    Windows run left to right, then top to bottom, and each is flattened row by
    row. A convolution with several kernels becomes one product of this array
    with a ``(k*k, channels)`` matrix.

    Examples
    --------
    On a 3x3 grid counting 1 to 9, the four 2x2 windows are:

    >>> import numpy as np
    >>> grid = np.arange(1.0, 10.0).reshape(1, 3, 3)
    >>> patches(grid, k=2)[0].tolist()
    [[1.0, 2.0, 4.0, 5.0], [2.0, 3.0, 5.0, 6.0], [4.0, 5.0, 7.0, 8.0], [5.0, 6.0, 8.0, 9.0]]
    >>> patches(np.zeros((2, 16, 16))).shape
    (2, 144, 25)
    """
    windows = np.lib.stride_tricks.sliding_window_view(images, (k, k), axis=(1, 2))
    return np.asarray(windows.reshape(len(images), -1, k * k), dtype=np.float64)


def init_params(seed: int = 0, channels: int = 8, k: int = 5, classes: int = 4) -> Params:
    """
    Return ``(Wc, bc, W2, b2)``: kernels, kernel biases, linear weights, linear biases.

    ``Wc`` has shape ``(k*k, channels)``, one flattened kernel per column, and
    ``W2`` has shape ``(channels, classes)``. Weights are Gaussian with standard
    deviation 0.2 and 0.3; biases start at zero.

    Examples
    --------
    >>> Wc, bc, W2, b2 = init_params(0, channels=3, k=2, classes=4)
    >>> Wc.shape, bc.shape, W2.shape, b2.shape
    ((4, 3), (3,), (3, 4), (4,))
    >>> float(abs(bc).sum()), float(abs(b2).sum())
    (0.0, 0.0)
    """
    rng = np.random.default_rng(seed)
    return (
        rng.normal(0.0, 0.2, (k * k, channels)),
        np.zeros(channels),
        rng.normal(0.0, 0.3, (channels, classes)),
        np.zeros(classes),
    )


def forward(params: Params, P: Array) -> tuple[Array, Array, Array, Array]:
    """
    Return ``(Z, A, F, logits)`` for the windows ``P``.

    ``Z = P @ Wc + bc`` is each kernel's response at each position, ``A`` is
    ``ReLU(Z)``, ``F`` is the mean of ``A`` over positions (global average
    pooling, one number per kernel), and ``logits = F @ W2 + b2``.

    Examples
    --------
    With all-zero weights the logits are the linear biases, whatever the image:

    >>> import numpy as np
    >>> params = (np.zeros((4, 2)), np.zeros(2), np.zeros((2, 3)), np.array([0.0, 1.0, 2.0]))
    >>> Z, A, F, logits = forward(params, np.ones((5, 9, 4)))
    >>> Z.shape, F.shape, logits.shape
    ((5, 9, 2), (5, 2), (5, 3))
    >>> logits[0].tolist()
    [0.0, 1.0, 2.0]

    A kernel of all ones sums its window, and a negative response is clipped:

    >>> params = (np.array([[1.0], [1.0]]), np.array([-1.0]), np.zeros((1, 3)), np.zeros(3))
    >>> windows = np.array([[[1.0, 2.0], [0.0, 0.5]]])
    >>> Z, A, F, _ = forward(params, windows)
    >>> Z[0, :, 0].tolist(), A[0, :, 0].tolist(), F.tolist()
    ([2.0, -0.5], [2.0, 0.0], [[1.0]])
    """
    Wc, bc, W2, b2 = params
    Z = P @ Wc + bc
    A = np.maximum(Z, 0.0)
    F = A.mean(axis=1)
    return Z, A, F, F @ W2 + b2


def loss_and_grads(params: Params, P: Array, y: Labels) -> tuple[float, Params]:
    """
    Return the mean softmax cross-entropy and its gradient for every parameter.

    Backward, with ``S`` the softmax of the logits and ``p`` the positions::

        dL = (S - onehot(y)) / B
        dF = dL @ W2ᵀ
        dZ = dF / p · (Z > 0)            (average pool, then ReLU mask)
        grad_Wc = Σ over images and positions of  windowᵀ · dZ
        grad_bc = Σ over images and positions of  dZ
        grad_W2 = Fᵀ @ dL
        grad_b2 = Σ over the batch of dL

    Every position reuses the same kernel, so ``grad_Wc`` adds up the
    contribution of all of them.

    Examples
    --------
    With zero weights the logits are 0, so the loss is ln 4. The error on a
    wrong class is 1/4 and on the true class 1/4 - 1, so ``b2`` moves by
    [-3/4, 1/4, 1/4, 1/4]. The kernels get no gradient, because ``W2`` is 0:

    >>> import numpy as np
    >>> params = (np.ones((1, 1)), np.zeros(1), np.zeros((1, 4)), np.zeros(4))
    >>> loss, (gWc, gbc, gW2, gb2) = loss_and_grads(params, np.full((1, 1, 1), 2.0), np.array([0]))
    >>> round(float(loss), 4), gb2.tolist(), gWc.tolist()
    (1.3863, [-0.75, 0.25, 0.25, 0.25], [[0.0]])

    A kernel whose response is negative is inactive, so ReLU blocks its
    gradient, even though ``W2`` would pass it on. Nothing reaches ``W2`` either,
    because the pooled activation is 0:

    >>> params = (np.ones((1, 1)), np.full(1, -5.0), np.array([[1.0, 0, 0, 0]]), np.zeros(4))
    >>> loss, (gWc, gbc, gW2, gb2) = loss_and_grads(params, np.full((1, 1, 1), 2.0), np.array([0]))
    >>> gWc.tolist(), gbc.tolist(), bool(gW2.any())
    ([[0.0]], [0.0], False)

    Active, the kernel passes the gradient ``dF`` on to its bias, and to its
    weight multiplied by the pixel 2. Here ``dF`` is also the error on class 0
    that ``W2`` carries, which is ``gW2``'s first entry because ``F`` is 1:

    >>> params = (np.ones((1, 1)), np.full(1, -1.0), np.array([[1.0, 0, 0, 0]]), np.zeros(4))
    >>> loss, (gWc, gbc, gW2, gb2) = loss_and_grads(params, np.full((1, 1, 1), 2.0), np.array([1]))
    >>> bool(gbc[0] == gW2[0, 0] > 0), bool(gWc[0, 0] == 2 * gbc[0])
    (True, True)
    """
    Wc, _, W2, _ = params
    Z, _, F, logits = forward(params, P)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(shifted).sum(axis=1))
    loss = float(np.mean(log_norm - shifted[np.arange(len(y)), y]))
    dL = np.exp(shifted - log_norm[:, None])
    dL[np.arange(len(y)), y] -= 1.0
    dL /= len(y)
    dF = dL @ W2.T
    dZ = (dF[:, None, :] / Z.shape[1]) * (Z > 0)
    grads = (
        P.reshape(-1, Wc.shape[0]).T @ dZ.reshape(-1, Wc.shape[1]),
        dZ.sum(axis=(0, 1)),
        F.T @ dL,
        dL.sum(axis=0),
    )
    return loss, grads


def gradient_error(
    grad_fn: GradFn,
    params: Params,
    P: Array,
    y: Labels,
    which: int = 0,
    eps: float = 1e-5,
) -> float:
    """
    Return the largest gap between ``grad_fn``'s gradient and central differences.

    ``grad_fn`` has the signature of ``loss_and_grads``. Each entry of
    ``params[which]`` is nudged by ±``eps`` and the loss from ``grad_fn`` is
    differenced. An exact gradient gives an error near ``eps``², a wrong one
    gives an error of the size of the missing or mis-scaled term.

    Examples
    --------
    The hand-derived gradient agrees with the numeric one for all four parameters:

    >>> import numpy as np
    >>> images, labels = line_dataset(6, seed=3)
    >>> P = patches(images)
    >>> params = init_params(1, channels=3)
    >>> [gradient_error(loss_and_grads, params, P, labels, which=i) < 1e-6 for i in range(4)]
    [True, True, True, True]

    The deliberate break: a gradient that treats the pooled mean as a sum
    scales the kernel gradient by the number of positions and is far off.

    >>> def broken(params, P, y):
    ...     loss, grads = loss_and_grads(params, P, y)
    ...     return loss, (grads[0] * P.shape[1], *grads[1:])
    >>> gradient_error(broken, params, P, labels, which=0) > 0.1
    True
    """
    analytic = grad_fn(params, P, y)[1][which]
    numeric = np.zeros_like(analytic)
    for index in np.ndindex(*analytic.shape):
        losses = []
        for step in (eps, -eps):
            moved = params[which].copy()
            moved[index] += step
            parts = list(params)
            parts[which] = moved
            losses.append(grad_fn((parts[0], parts[1], parts[2], parts[3]), P, y)[0])
        numeric[index] = (losses[0] - losses[1]) / (2 * eps)
    return float(np.max(np.abs(analytic - numeric)))


def accuracy(params: Params, P: Array, y: Labels) -> float:
    """
    Return the fraction of images whose largest logit is the true class.

    Examples
    --------
    With zero weights the logits are the biases, so every image gets class 2:

    >>> import numpy as np
    >>> params = (np.zeros((1, 1)), np.zeros(1), np.zeros((1, 3)), np.array([0.0, 0.0, 1.0]))
    >>> accuracy(params, np.zeros((4, 1, 1)), np.array([2, 2, 0, 1]))
    0.5
    """
    logits = forward(params, P)[3]
    return float(np.mean(logits.argmax(axis=1) == y))


def train(
    P: Array,
    y: Labels,
    steps: int = 1500,
    batch: int = 128,
    lr: float = 1e-2,
    seed: int = 0,
    channels: int = 8,
) -> tuple[Params, float]:
    """
    Train with hand-written Adam and return ``(params, accuracy on P)``.

    Adam uses betas 0.9 and 0.999 and eps 1e-8. The seeded generator draws the
    initial parameters and then a batch of indices per step, so a run repeats
    exactly.

    Examples
    --------
    A small run on 300 noisy lines separates the four angles:

    >>> images, labels = line_dataset(300, seed=0)
    >>> P = patches(images)
    >>> params, acc = train(P, labels, steps=300, batch=64, channels=4)
    >>> params[0].shape, acc > 0.9
    ((25, 4), True)
    >>> _, again = train(P, labels, steps=300, batch=64, channels=4)
    >>> again == acc
    True
    """
    rng = np.random.default_rng(seed)
    current = init_params(seed, channels, round(P.shape[2] ** 0.5))
    params = list(current)
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        idx = rng.integers(0, len(y), batch)
        _, grads = loss_and_grads(current, P[idx], y[idx])
        for p, g, v, u in zip(params, grads, first, second, strict=True):
            v *= beta1
            v += (1 - beta1) * g
            u *= beta2
            u += (1 - beta2) * g * g
            p -= lr * (v / (1 - beta1**t)) / (np.sqrt(u / (1 - beta2**t)) + eps)
    return current, accuracy(current, P, y)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# READING THE KERNELS: Orientation Selectivity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBE_ANGLES: Final[tuple[float, ...]] = tuple(np.arange(0.0, 180.0, 22.5).tolist())


def grating(angle: float, period: float = 6.0) -> Array:
    """
    Return a ``SIZE`` x ``SIZE`` sine grating whose stripes run along ``angle`` degrees.

    Pixel values are ``cos(2π d / period)`` with ``d`` the distance across the
    stripes, so the stripes repeat every ``period`` pixels. Angle 0 gives
    horizontal stripes, which vary only down the rows.

    Examples
    --------
    >>> import numpy as np
    >>> g0, g90 = grating(0.0), grating(90.0)
    >>> bool(np.allclose(g0, g0[:, :1])), bool(np.allclose(g90, g90[:1, :]))
    (True, True)
    >>> float(g0[0, 0]), round(float(g0[3, 0]), 6)
    (1.0, -1.0)
    """
    xx, yy = np.meshgrid(np.arange(SIZE, dtype=np.float64), np.arange(SIZE, dtype=np.float64))
    t = np.deg2rad(angle)
    return np.asarray(np.cos(2 * np.pi * (-xx * np.sin(t) + yy * np.cos(t)) / period))


def tuning(
    params: Params,
    angles: Sequence[float] = PROBE_ANGLES,
    periods: Sequence[float] = (4.0, 6.0, 8.0),
) -> Array:
    """
    Return the pooled activation ``F`` of each kernel for each angle, shape ``(angles, channels)``.

    Each entry averages the network's pooled activation over gratings of
    several ``periods``.

    Examples
    --------
    A kernel with zero weights and bias 1 pools to 1 for every grating:

    >>> import numpy as np
    >>> params = (np.zeros((25, 1)), np.ones(1), np.zeros((1, 4)), np.zeros(4))
    >>> tuning(params, angles=(0.0, 45.0)).round(6).tolist()
    [[1.0], [1.0]]
    """
    return np.array(
        [
            forward(params, patches(np.stack([grating(a, p) for p in periods])))[2].mean(axis=0)
            for a in angles
        ]
    )


def orientation_selectivity(
    params: Params,
    angles: Sequence[float] = PROBE_ANGLES,
) -> Array:
    """
    Return ``(max - min) / (max + min)`` of each kernel's ``tuning`` over ``angles``.

    The score is 0 for a kernel that responds equally to every angle, 1 for one
    that responds to some angle and not at all to another, and 0 for a kernel
    that never responds.

    Examples
    --------
    A flat kernel with a bias responds equally to every angle, and a dead one
    never responds:

    >>> import numpy as np
    >>> flat = (np.zeros((25, 1)), np.ones(1), np.zeros((1, 4)), np.zeros(4))
    >>> orientation_selectivity(flat).tolist()
    [0.0]
    >>> dead = (np.zeros((25, 1)), np.zeros(1), np.zeros((1, 4)), np.zeros(4))
    >>> orientation_selectivity(dead).tolist()
    [0.0]

    A horizontal bar detector has weights 4 on the middle row and -1 on the
    others. They sum to 0, so a grating whose stripes vary only along the
    columns (90 degrees) averages to nothing under it, and ReLU leaves 0:

    >>> bar = np.full((5, 5), -1.0)
    >>> bar[2] = 4.0
    >>> params = (bar.reshape(25, 1), np.zeros(1), np.zeros((1, 4)), np.zeros(4))
    >>> float(orientation_selectivity(params)[0]) > 0.99
    True
    """
    response = tuning(params, angles)
    top, bottom = response.max(axis=0), response.min(axis=0)
    total = top + bottom
    return np.asarray(
        np.divide(top - bottom, total, out=np.zeros_like(total), where=total > 0),
        dtype=np.float64,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEEING IT: ASCII Kernels (stdlib)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAMP: Final[str] = " .:-=+*#%@"
NEG_RAMP: Final[str] = " ,;~ioxOX&"


def show_filters(Wc: Array, per_row: int = 4) -> str:
    """
    Draw each column of ``Wc`` as a square kernel, ``per_row`` kernels per band of text.

    Cells are doubled characters. Positive weights use ``RAMP`` and negative
    weights use ``NEG_RAMP``, both scaled so that the largest ``|weight|`` of
    all the kernels draws darkest.

    Examples
    --------
    One 2x2 kernel with weights 1, 0, -1, -0.05:

    >>> import numpy as np
    >>> print(show_filters(np.array([[1.0], [0.0], [-1.0], [-0.05]])))
    #0
    @@
    &&
    """
    k = round(len(Wc) ** 0.5)
    top = float(np.abs(Wc).max())
    width = 2 * k + 2
    bands = []
    for start in range(0, Wc.shape[1], per_row):
        columns = range(start, min(start + per_row, Wc.shape[1]))
        grids = []
        for c in columns:
            rows = []
            for row in Wc[:, c].reshape(k, k):
                cells = []
                for v in row:
                    level = min(int(abs(v) / top * len(RAMP)), len(RAMP) - 1) if top > 0 else 0
                    cells.append((RAMP if v >= 0 else NEG_RAMP)[level] * 2)
                rows.append("".join(cells))
            grids.append(rows)
        lines = ["".join(f"{'#' + str(c):<{width}}" for c in columns).rstrip()]
        lines += [
            "".join(f"{row:<{width}}" for row in rows).rstrip() for rows in zip(*grids, strict=True)
        ]
        bands.append("\n".join(lines))
    return "\n\n".join(bands)


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
      LESSON INDEX: tier_5_lesson_2c_tiny_cnn
    ...
    Prerequisites (hard deps):
       → tier_3_lesson_4a_convolution: The sliding dot product each kernel computes
       → tier_5_lesson_2a_superposition: Hand-derived gradients checked by finite differences
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
    steps: int = 1500,
    n_train: int = 4000,
    n_test: int = 1000,
    batch: int = 128,
    channels: int = 8,
) -> None:
    """
    Train the network, then print its accuracy, each kernel's selectivity, and the kernels.

    Examples
    --------
    >>> demo(steps=100, n_train=100, n_test=50, channels=2)  # doctest: +ELLIPSIS
    ═...
    train accuracy ...
    test accuracy  ...
    kernel  selectivity  preferred angle
    0       ...
    1       ...
    ...
    """
    print("═" * 70)
    print(f"  {LESSON_ID.upper().replace('_', ' ')}")
    print("═" * 70)
    print()
    train_images, y_train = line_dataset(n_train, seed=0)
    test_images, y_test = line_dataset(n_test, seed=1)
    P_train, P_test = patches(train_images), patches(test_images)
    params, train_acc = train(P_train, y_train, steps=steps, batch=batch, channels=channels)
    print(f"train accuracy {train_acc:.3f}")
    print(f"test accuracy  {accuracy(params, P_test, y_test):.3f}")
    print()
    selectivity = orientation_selectivity(params)
    preferred = tuning(params).argmax(axis=0)
    print("kernel  selectivity  preferred angle")
    for c, (score, angle_index) in enumerate(zip(selectivity, preferred, strict=True)):
        print(f"{c:<8}{score:<13.2f}{PROBE_ANGLES[angle_index]:g} deg")
    print()
    print(f"Learned kernels (positive: '{RAMP.strip()}', negative: '{NEG_RAMP.strip()}'):")
    print()
    print(show_filters(params[0]))


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
