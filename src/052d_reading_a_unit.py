#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<4.0"
# dependencies = ["numpy>=2.5.4"]
# ///
"""
Lesson 5.2d: Reading a Unit (side quest).

You read one channel of a tiny convolutional network with the three tools that
Distill's interpretability articles use together: the dataset examples that
excite it most, its tuning curve over sine gratings at many angles, and a
feature visualization found by gradient ascent on the input image. Each tool
alone can mislead; where they agree, the reading is evidence, and where they
disagree, the gap tells you something about the unit.

The network is the one Lesson 5.2c trains: 8 kernels of 5x5, ReLU, global
average pooling, and a linear layer over 4 line angles. This file carries a
compact copy of its data, windows, forward pass, and training, so the lesson
runs on its own and trains the same network as Lesson 5.2c.

Source: Olah, Mordvintsev, Schubert, "Feature Visualization", Distill (2017),
https://distill.pub/2017/feature-visualization/ , and Cammarata et al., "Curve
Detectors", Distill (2020), https://distill.pub/2020/circuits/curve-detectors/
(which reads each neuron family with feature visualization, dataset examples,
and tuning curves together).

Prerequisites (must complete first)
----------------------------------
- Lesson 5.2c: A Tiny Convolutional Network (the network being read)

Learning Objectives
-------------------
After completing this lesson, you will be able to:
1. Rank a dataset by one channel's activation and read the classes at the top.
2. Measure a tuning curve and find a channel's preferred grating angle.
3. Derive the gradient of a channel's pooled activation with respect to the
   input image, check it by central differences, and climb it to draw what the
   channel prefers.
4. Compare the three readings, and treat a disagreement as information.

The Big Idea
------------
A unit is not what any one tool says it is. Dataset examples show what
excites it on the data, tuning curves show how it responds to a controlled
family of inputs, and feature visualization shows the input it would most
like. When all three name the same thing, you have evidence; when they differ,
the unit responds to something the other tools cannot see.

Why It Matters
--------------
Distill's curve-detector article reads each neuron with all three tools
because each one fails differently: dataset examples reflect what the data
contains, gratings only test what you thought to try, and an optimized image
can drift off the data. Trying the same three on 8 channels you can inspect in
full shows both outcomes: some channels agree on one angle, others only
partly.

Stdlib Pain → Library Power-Up
------------------------------
Pain (this lesson):
- You derive the input gradient by hand: each window position's kernel goes
  back onto the pixels it covers, masked by the ReLU. A forgotten mask or a
  shifted offset still climbs, to the wrong picture, so you check it with
  central differences.
- Lines of code: ~15 for the input gradient, ~15 for the ascent.
- Runtime: about 20 seconds for the demo (training dominates), measured with
  one BLAS thread.
- Common bugs: scattering at the wrong window offset, dropping the ReLU mask,
  and reading one tool's answer as the unit's.

Power-Up (Lesson 5.3 — PyTorch):
- ``requires_grad`` on the image makes ``loss.backward()`` produce the input
  gradient with no hand derivation, on networks far too large to derive by hand.
- Key APIs: ``torch.autograd.grad``.

Complexity Analysis
-------------------
Pain (manual), for k x k kernels on H x W images, N images, s ascent steps:
- Time:  O(s·k²·H·W) for one visualization; O(N·H·W·k²) to rank N images.
- Space: O(H·W) for the image and its gradient.

Execution
---------
Project-style:
    uv run python src/052d_reading_a_unit.py

Script-style (self-contained deps):
    uv run --script src/052d_reading_a_unit.py

On a busy machine, NumPy's BLAS threads compete with other work and can stretch
the demo from about 20 seconds to a minute or more; one thread avoids that, since the
matrices here are small:
    OPENBLAS_NUM_THREADS=1 uv run python src/052d_reading_a_unit.py

Options:
    python src/052d_reading_a_unit.py --doctest-only
    python src/052d_reading_a_unit.py --no-doctest
    python src/052d_reading_a_unit.py --index
    python src/052d_reading_a_unit.py -v

Quality Gates
-------------
    uv run pytest src/052d_reading_a_unit.py
    uv run ruff check src/052d_reading_a_unit.py
    uv run ruff format --check src/052d_reading_a_unit.py
    uv run mypy --strict src/052d_reading_a_unit.py

Module Doctests
---------------
A 1x1 kernel of weight 2 on a 1x1 image of 3 gives the pooled activation 2·3,
so its gradient with respect to the pixel is the weight, 2:

>>> import numpy as np
>>> params = (np.full((1, 1), 2.0), np.zeros(1), np.zeros((1, 4)), np.zeros(4))
>>> input_gradient(params, np.array([[3.0]]), 0).tolist()
[[2.0]]
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
    type GradFn = Callable[[Params, Array, int], Array]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON IDENTITY & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON_ID: Final[str] = "tier_5_lesson_2d_reading_a_unit"

# Autograd gives the input gradient for free with PyTorch.
POWER_UP_AVAILABLE: Final[bool] = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Format: lesson_id -> reason

PREREQUISITES: Final[Mapping[str, str]] = {
    "tier_5_lesson_2c_tiny_cnn": "The network being read",
}

DEPENDENCIES: Final[Mapping[str, str]] = {
    "tier_3_lesson_4a_convolution": "The sliding dot product the input gradient reverses",
}

SUCCESSORS: Final[Mapping[str, str]] = {
    "tier_5_lesson_3_pytorch": "The same tools on real networks",
}

DEPENDENTS: Final[Mapping[str, str]] = {}

RELATED: Final[Mapping[str, str]] = {
    "tier_5_lesson_2a_superposition": "Polysemantic units respond to more than one thing",
}

POWER_UP_IN_LESSON: Final[str | None] = "tier_5_lesson_3_pytorch"
PAIN_FROM_LESSON: Final[str | None] = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE NETWORK: A Compact Copy of Lesson 5.2c
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


def forward(params: Params, P: Array) -> tuple[Array, Array, Array, Array]:
    """
    Return ``(Z, A, F, logits)`` for the windows ``P``.

    ``Z = P @ Wc + bc`` is each kernel's response at each position, ``A`` is
    ``ReLU(Z)``, ``F`` is the mean of ``A`` over positions (global average
    pooling, one number per kernel), and ``logits = F @ W2 + b2``. ``F[:, c]``
    is the number this lesson reads for channel ``c``.

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


def train(
    P: Array,
    y: Labels,
    steps: int = 1500,
    batch: int = 128,
    lr: float = 1e-2,
    seed: int = 0,
    channels: int = 8,
) -> Params:
    """
    Train Lesson 5.2c's network with hand-written Adam and return ``(Wc, bc, W2, b2)``.

    The gradient is the one Lesson 5.2c derives: softmax error ``(S - onehot) / B``,
    back through the linear layer, the average pool (``/ positions``), and the
    ReLU mask, then summed over every position the kernel visits. Adam uses
    betas 0.9 and 0.999 and eps 1e-8, and the seeded generators fix the
    initial weights and the batches, so a run repeats exactly.

    Examples
    --------
    A small run on 300 noisy lines separates the four angles:

    >>> import numpy as np
    >>> images, labels = line_dataset(300, seed=0)
    >>> P = patches(images)
    >>> params = train(P, labels, steps=300, batch=64, channels=4)
    >>> params[0].shape, params[2].shape
    ((25, 4), (4, 4))
    >>> float(np.mean(forward(params, P)[3].argmax(axis=1) == labels)) > 0.9
    True
    >>> again = train(P, labels, steps=300, batch=64, channels=4)
    >>> bool(np.array_equal(params[0], again[0]))
    True
    """
    rng = np.random.default_rng(seed)
    init = np.random.default_rng(seed)
    k = round(P.shape[2] ** 0.5)
    weights = [
        init.normal(0.0, 0.2, (k * k, channels)),
        np.zeros(channels),
        init.normal(0.0, 0.3, (channels, len(ANGLES))),
        np.zeros(len(ANGLES)),
    ]
    current: Params = (weights[0], weights[1], weights[2], weights[3])
    first = [np.zeros_like(w) for w in weights]
    second = [np.zeros_like(w) for w in weights]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        idx = rng.integers(0, len(y), batch)
        Z, _, F, logits = forward(current, P[idx])
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(shifted).sum(axis=1))
        dL = np.exp(shifted - log_norm[:, None])
        dL[np.arange(batch), y[idx]] -= 1.0
        dL /= batch
        dZ = ((dL @ current[2].T)[:, None, :] / Z.shape[1]) * (Z > 0)
        grads = (
            P[idx].reshape(-1, k * k).T @ dZ.reshape(-1, channels),
            dZ.sum(axis=(0, 1)),
            F.T @ dL,
            dL.sum(axis=0),
        )
        for w, g, v, u in zip(weights, grads, first, second, strict=True):
            v *= beta1
            v += (1 - beta1) * g
            u *= beta2
            u += (1 - beta2) * g * g
            w -= lr * (v / (1 - beta1**t)) / (np.sqrt(u / (1 - beta2**t)) + eps)
    return current


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 1 & 2: Dataset Examples and Tuning Curves
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBE_ANGLES: Final[tuple[float, ...]] = tuple(np.arange(0.0, 180.0, 22.5).tolist())
ANGLE_STEP: Final[float] = 22.5


def vertical_bar(k: int = 5) -> Params:
    """
    Return a one-channel network whose kernel is a hand-set vertical line detector.

    The middle column of the ``k`` x ``k`` kernel has weight 4 per pixel, every
    other pixel has weight -1, and the bias is 0. The weights sum to 0 when
    ``k`` is 5, so a flat patch gives no response. The linear layer is zeros;
    only the kernel matters for reading.

    Examples
    --------
    >>> import numpy as np
    >>> Wc, bc, W2, b2 = vertical_bar()
    >>> Wc.shape, float(Wc.sum())
    ((25, 1), 0.0)
    >>> Wc[:, 0].reshape(5, 5)[:, 2].tolist(), Wc[:, 0].reshape(5, 5)[:, 0].tolist()
    ([4.0, 4.0, 4.0, 4.0, 4.0], [-1.0, -1.0, -1.0, -1.0, -1.0])
    """
    kernel = np.full((k, k), -1.0)
    kernel[:, k // 2] = 4.0
    return (
        kernel.reshape(k * k, 1),
        np.zeros(1),
        np.zeros((1, len(ANGLES))),
        np.zeros(len(ANGLES)),
    )


def grating(angle: float, period: float = 6.0, phase: float = 0.0) -> Array:
    """
    Return a ``SIZE`` x ``SIZE`` sine grating whose stripes run along ``angle`` degrees.

    Pixel values are ``cos(2π d / period + phase)`` with ``d`` the distance
    across the stripes, so the stripes repeat every ``period`` pixels. Angle 0
    gives horizontal stripes, which vary only down the rows.

    Examples
    --------
    >>> import numpy as np
    >>> g0, g90 = grating(0.0), grating(90.0)
    >>> bool(np.allclose(g0, g0[:, :1])), bool(np.allclose(g90, g90[:1, :]))
    (True, True)
    >>> float(g0[0, 0]), round(float(g0[3, 0]), 6)
    (1.0, -1.0)

    A phase of π flips the sign of every pixel:

    >>> bool(np.allclose(grating(45.0, phase=np.pi), -grating(45.0)))
    True
    """
    xx, yy = np.meshgrid(np.arange(SIZE, dtype=np.float64), np.arange(SIZE, dtype=np.float64))
    t = np.deg2rad(angle)
    return np.asarray(
        np.cos(2 * np.pi * (-xx * np.sin(t) + yy * np.cos(t)) / period + phase),
        dtype=np.float64,
    )


def pooled(params: Params, image: Array, channel: int) -> float:
    """
    Return channel ``channel``'s pooled activation ``F`` for one ``image``.

    ``image`` is a single 2-D array of any size at least as large as the kernel.

    Examples
    --------
    A 2x2 kernel of ones sums each window, and every window of a 3x3 image of
    ones sums to 4:

    >>> import numpy as np
    >>> params = (np.ones((4, 1)), np.zeros(1), np.zeros((1, 4)), np.zeros(4))
    >>> pooled(params, np.ones((3, 3)), 0)
    4.0
    """
    k = round(len(params[0]) ** 0.5)
    return float(forward(params, patches(image[None], k))[2][0, channel])


def dataset_examples(
    params: Params,
    P: Array,
    labels: Labels,
    channel: int,
    k: int = 100,
) -> tuple[Labels, Array]:
    """
    Return the top ``k`` images by pooled activation, and the class shares among them.

    The indices run from the most to the least activating image. The shares
    have one entry per class in ``ANGLES``, the fraction of those ``k`` images
    that belong to it, so they sum to 1.

    Examples
    --------
    A vertical line detector (class 2, 90 degrees) ranks vertical lines first.
    The first image is the most active, and most of the top 50 are vertical:

    >>> import numpy as np
    >>> images, labels = line_dataset(400, seed=0)
    >>> P = patches(images)
    >>> params = vertical_bar()
    >>> top, shares = dataset_examples(params, P, labels, 0, k=50)
    >>> F = forward(params, P)[2][:, 0]
    >>> bool(F[top[0]] == F.max()), bool(F[top[0]] >= F[top[-1]])
    (True, True)
    >>> round(float(shares.sum()), 6), int(shares.argmax()), bool(shares.max() > 0.8)
    (1.0, 2, True)
    """
    F = forward(params, P)[2][:, channel]
    top = np.argsort(-F, kind="stable")[:k]
    shares = np.bincount(labels[top], minlength=len(ANGLES)) / len(top)
    return np.asarray(top, dtype=np.int64), np.asarray(shares, dtype=np.float64)


def tuning_curve(
    params: Params,
    channel: int,
    angles: Sequence[float] = PROBE_ANGLES,
    periods: Sequence[float] = (4.0, 6.0, 8.0),
) -> tuple[Array, float]:
    """
    Return channel ``channel``'s response to gratings at each angle, and its preferred angle.

    Each response averages the pooled activation over gratings of several
    ``periods``. The preferred angle is the one with the largest response.

    Examples
    --------
    A vertical line detector (90 degrees) gives no response to horizontal
    stripes, whose rows are constant and cancel under the zero-sum kernel, and
    peaks at 90:

    >>> curve, preferred = tuning_curve(vertical_bar(), 0)
    >>> curve.shape, round(float(curve[0]), 6), preferred
    ((8,), 0.0, 90.0)
    >>> bool(curve.argmax() == PROBE_ANGLES.index(90.0))
    True
    """
    k = round(len(params[0]) ** 0.5)
    curve = np.array(
        [
            forward(params, patches(np.stack([grating(a, p) for p in periods]), k))[2][
                :, channel
            ].mean()
            for a in angles
        ]
    )
    return curve, float(angles[int(curve.argmax())])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 3: Feature Visualization by Gradient Ascent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def input_gradient(params: Params, image: Array, channel: int) -> Array:
    """
    Return the gradient of channel ``channel``'s pooled activation with respect to ``image``.

    Backward through ``F = mean(ReLU(Z))``, with ``p`` window positions::

        dZ[position] = (Z[position] > 0) / p          (average pool, then ReLU mask)
        dimage       = Σ over positions of  kernel placed on the pixels that window covers · dZ

    Each pixel collects the kernel weight at its place in every window that
    covers it. The loop runs over the ``k`` x ``k`` kernel offsets and adds
    ``kernel[i, j] · dZ`` to the pixels each offset reaches.

    Examples
    --------
    A 2x2 kernel with weights 1, 2, 3, 4 on a 3x3 image of ones has 4 active
    windows. The centre pixel lies in all four, at the weights 4, 3, 2, 1, so its
    gradient is 10/4. A corner lies in one window, at weight 1, so 1/4. The top
    middle pixel lies in two, at weights 2 and 1, so 3/4. The gradient sums to
    4 · 10/4 = 10, the total weight each window passes back:

    >>> import numpy as np
    >>> weights = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> params = (weights, np.zeros(1), np.zeros((1, 4)), np.zeros(4))
    >>> g = input_gradient(params, np.ones((3, 3)), 0)
    >>> float(g[1, 1]), float(g[0, 0]), float(g[0, 1]), float(g.sum())
    (2.5, 0.25, 0.75, 10.0)

    A window with a negative response is blocked by the ReLU. With a 1x1 kernel
    of weight 2, the negative pixel passes nothing back and each active pixel
    passes 2/4:

    >>> params = (np.full((1, 1), 2.0), np.zeros(1), np.zeros((1, 4)), np.zeros(4))
    >>> input_gradient(params, np.array([[1.0, -1.0], [1.0, 1.0]]), 0).tolist()
    [[0.5, 0.0], [0.5, 0.5]]
    """
    Wc = params[0]
    k = round(len(Wc) ** 0.5)
    Z = forward(params, patches(image[None], k))[0][0, :, channel]
    rows, cols = image.shape[0] - k + 1, image.shape[1] - k + 1
    dZ = (Z > 0).reshape(rows, cols) / Z.size
    grad = np.zeros_like(image)
    for i in range(k):
        for j in range(k):
            grad[i : i + rows, j : j + cols] += Wc[i * k + j, channel] * dZ
    return grad


def input_gradient_error(
    grad_fn: GradFn,
    params: Params,
    image: Array,
    channel: int = 0,
    eps: float = 1e-6,
) -> float:
    """
    Return the largest gap between ``grad_fn``'s gradient and central differences on the image.

    ``grad_fn`` has the signature of ``input_gradient``. Each pixel is nudged
    by ±``eps`` and the change in ``pooled`` is differenced. An exact gradient
    gives an error near ``eps``², a wrong one gives an error of the size of the
    missing or misplaced term.

    Examples
    --------
    On a 6x6 float64 image with a 3x3 kernel, the hand-derived gradient agrees
    with the numeric one:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> image = rng.normal(0.0, 1.0, (6, 6))
    >>> params = (rng.normal(0.0, 1.0, (9, 1)), np.zeros(1), np.zeros((1, 4)), np.zeros(4))
    >>> input_gradient_error(input_gradient, params, image) < 1e-6
    True

    The deliberate break: a gradient that forgets the ReLU mask passes back
    the kernel from windows whose response is negative, and is far off:

    >>> def no_mask(params, image, channel):
    ...     size = image.shape[0] - 2
    ...     g = np.zeros_like(image)
    ...     for i in range(3):
    ...         for j in range(3):
    ...             g[i : i + size, j : j + size] += params[0][3 * i + j, channel] / size**2
    ...     return g
    >>> input_gradient_error(no_mask, params, image) > 0.01
    True
    """
    analytic = grad_fn(params, image, channel)
    numeric = np.zeros_like(analytic)
    for index in np.ndindex(*image.shape):
        values = []
        for step in (eps, -eps):
            moved = image.copy()
            moved[index] += step
            values.append(pooled(params, moved, channel))
        numeric[index] = (values[0] - values[1]) / (2 * eps)
    return float(np.max(np.abs(analytic - numeric)))


def feature_visualization(
    params: Params,
    channel: int,
    steps: int = 200,
    step_size: float = 0.1,
    jitter: int = 2,
    seed: int = 0,
) -> Array:
    """
    Return a ``SIZE`` x ``SIZE`` image that climbs channel ``channel``'s pooled activation.

    The image starts as Gaussian noise of standard deviation 0.01. Each step
    shifts it by a random offset of up to ``jitter`` pixels in each direction,
    so the result has to work wherever it lands, takes ``input_gradient`` there,
    shifts the gradient back, and adds it scaled so its largest entry is
    ``step_size``. Pixels clip to [-1, 1]. A channel that never activates
    has zero gradient, and its image stays noise.

    Examples
    --------
    On a vertical line detector, 100 steps raise the pooled activation far above
    that of the noise it started from, keep every pixel in [-1, 1], and repeat
    exactly for one seed:

    >>> import numpy as np
    >>> params = vertical_bar()
    >>> image = feature_visualization(params, 0, steps=100)
    >>> image.shape, float(np.abs(image).max()) <= 1.0
    ((16, 16), True)
    >>> noise = np.random.default_rng(0).normal(0.0, 0.01, (16, 16))
    >>> bool(pooled(params, image, 0) > 10 * pooled(params, noise, 0))
    True
    >>> bool(np.array_equal(image, feature_visualization(params, 0, steps=100)))
    True

    A dead channel (bias far below zero) never moves off the noise:

    >>> dead = (params[0], np.full(1, -100.0), params[2], params[3])
    >>> float(np.abs(feature_visualization(dead, 0, steps=20)).max()) < 0.1
    True
    """
    rng = np.random.default_rng(seed)
    image = rng.normal(0.0, 0.01, (SIZE, SIZE))
    for _ in range(steps):
        dy, dx = (int(v) for v in rng.integers(-jitter, jitter + 1, 2))
        shifted = np.roll(image, (dy, dx), axis=(0, 1))
        grad = np.roll(input_gradient(params, shifted, channel), (-dy, -dx), axis=(0, 1))
        image = np.clip(image + step_size * grad / (np.abs(grad).max() + 1e-9), -1.0, 1.0)
    return image


def dominant_angle(
    image: Array,
    angles: Sequence[float] = PROBE_ANGLES,
    periods: Sequence[float] = (4.0, 6.0, 8.0),
) -> float:
    """
    Return the angle in ``angles`` whose gratings ``image`` correlates with most.

    The score of an angle sums, over ``periods``, the length of the pair
    ``(image · cosine grating, image · sine grating)``, so a stripe pattern
    scores the same wherever its stripes fall. The image's mean is removed first.

    Examples
    --------
    A grating scores highest at its own angle, whatever its phase:

    >>> dominant_angle(grating(45.0)), dominant_angle(grating(90.0, phase=1.0))
    (45.0, 90.0)

    A vertical line detector's visualization correlates most with vertical stripes:

    >>> dominant_angle(feature_visualization(vertical_bar(), 0, steps=100))
    90.0
    """
    x = image - image.mean()
    scores = [
        sum(
            np.hypot(
                (x * grating(a, p)).sum(),
                (x * grating(a, p, phase=-np.pi / 2)).sum(),
            )
            for p in periods
        )
        for a in angles
    ]
    return float(angles[int(np.argmax(scores))])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPARING THE THREE READINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True)
class Reading:
    """One channel read by three tools.

    Attributes
    ----------
    channel : int
        Channel index.
    preferred : float
        Grating angle with the largest response, in degrees.
    visualized : float
        Grating angle the feature visualization correlates with most.
    top_angle : float
        Angle of the class most common among the top dataset examples.
    top_share : float
        Fraction of the top examples that belong to that class.
    """

    channel: int
    preferred: float
    visualized: float
    top_angle: float
    top_share: float


def angle_gap(a: float, b: float) -> float:
    """
    Return the distance in degrees between two line angles, which repeat every 180.

    Examples
    --------
    >>> angle_gap(90.0, 135.0), angle_gap(0.0, 135.0), angle_gap(45.0, 45.0)
    (45.0, 45.0, 0.0)
    """
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def spread(reading: Reading) -> float:
    """
    Return the largest gap between the three angles of ``reading``.

    A spread of 0 means the three tools name one angle; a spread of at most
    ``ANGLE_STEP`` counts as agreement.

    Examples
    --------
    >>> spread(Reading(0, 90.0, 90.0, 90.0, 0.9))
    0.0
    >>> spread(Reading(1, 90.0, 112.5, 45.0, 0.6))
    67.5
    """
    a, b, c = reading.preferred, reading.visualized, reading.top_angle
    return max(angle_gap(a, b), angle_gap(a, c), angle_gap(b, c))


def compare(
    params: Params,
    P: Array,
    labels: Labels,
    k: int = 100,
    steps: int = 200,
    seed: int = 0,
) -> list[Reading]:
    """
    Return one ``Reading`` per channel: preferred angle, visualized angle, and top-example class.

    Each channel uses ``tuning_curve``, ``feature_visualization`` followed by
    ``dominant_angle``, and ``dataset_examples`` over the windows ``P``.

    Examples
    --------
    A vertical line detector is read as 90 degrees by all three tools, and
    most of its top examples are vertical lines:

    >>> images, labels = line_dataset(400, seed=0)
    >>> readings = compare(vertical_bar(), patches(images), labels, k=50, steps=100)
    >>> len(readings), readings[0].preferred, readings[0].visualized, readings[0].top_angle
    (1, 90.0, 90.0, 90.0)
    >>> bool(readings[0].top_share > 0.8), spread(readings[0])
    (True, 0.0)
    """
    readings = []
    for channel in range(params[0].shape[1]):
        _, preferred = tuning_curve(params, channel)
        visualized = dominant_angle(feature_visualization(params, channel, steps, seed=seed))
        _, shares = dataset_examples(params, P, labels, channel, k)
        top = int(shares.argmax())
        readings.append(Reading(channel, preferred, visualized, ANGLES[top], float(shares[top])))
    return readings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEEING IT: ASCII Images (stdlib)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAMP: Final[str] = " .:-=+*#%@"


def shade(image: Array, top: float | None = None) -> str:
    """
    Draw an image as text, one doubled character per value, darker for larger values.

    Negative values count as 0. ``top`` is the value drawn darkest; it defaults
    to the image's own maximum, and passing one shared ``top`` puts two images
    on the same scale.

    Examples
    --------
    >>> import numpy as np
    >>> shade(np.array([[0.0, 1.0], [0.5, -2.0]])).splitlines()
    ['  @@', '++  ']
    >>> shade(np.array([[0.0, 1.0], [0.5, -2.0]]), top=2.0).splitlines()
    ['  ++', '::  ']
    """
    values = np.maximum(image, 0.0)
    if top is None:
        top = float(values.max())
    rows = []
    for row in values:
        cells = (
            RAMP[min(int(v / top * len(RAMP)), len(RAMP) - 1)] if top > 0 else " " for v in row
        )
        rows.append("".join(c * 2 for c in cells))
    return "\n".join(rows)


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
      LESSON INDEX: tier_5_lesson_2d_reading_a_unit
    ...
    Prerequisites (hard deps):
       → tier_5_lesson_2c_tiny_cnn: The network being read
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
    batch: int = 128,
    channels: int = 8,
    k: int = 100,
    viz_steps: int = 200,
) -> None:
    """
    Train the network, print one row per channel for the three tools, and draw two visualizations.

    Examples
    --------
    >>> demo(steps=100, n_train=200, channels=2, k=20, viz_steps=20)  # doctest: +ELLIPSIS
    ═...
    channel  preferred  visualized  top class  share  spread
    0        ...
    1        ...
    ...
    """
    print("═" * 70)
    print(f"  {LESSON_ID.upper().replace('_', ' ')}")
    print("═" * 70)
    print()
    images, labels = line_dataset(n_train, seed=0)
    P = patches(images)
    params = train(P, labels, steps=steps, batch=batch, channels=channels)
    accuracy = float(np.mean(forward(params, P)[3].argmax(axis=1) == labels))
    print(f"train accuracy {accuracy:.3f}")
    print()
    readings = compare(params, P, labels, k=k, steps=viz_steps)
    print("channel  preferred  visualized  top class  share  spread")
    for r in readings:
        print(
            f"{r.channel:<9}{f'{r.preferred:g} deg':<11}{f'{r.visualized:g} deg':<12}"
            f"{f'{r.top_angle:g} deg':<11}{f'{r.top_share:.0%}':<7}{spread(r):g}"
        )
    print()
    agree = sum(spread(r) <= ANGLE_STEP for r in readings)
    print(
        f"{agree} of {len(readings)} channels: all three tools agree within one angle step"
        f" ({ANGLE_STEP:g} deg)"
    )
    best = min(readings, key=lambda r: (spread(r), -r.top_share))
    worst = max(readings, key=lambda r: (spread(r), -r.top_share))
    for title, r in (("agree best", best), ("agree least", worst)):
        print()
        print(
            f"Channel {r.channel} ({title}): grating {r.preferred:g} deg, "
            f"visualization {r.visualized:g} deg, top {k} examples "
            f"{r.top_share:.0%} from the {r.top_angle:g} deg class"
        )
        print()
        print(shade(feature_visualization(params, r.channel, viz_steps)))


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
