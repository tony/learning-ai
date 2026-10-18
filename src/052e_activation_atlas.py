#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<4.0"
# dependencies = ["numpy>=2.5.4"]
# ///
"""
Lesson 5.2e: An Activation Atlas (side quest).

You build Distill's Activation Atlas in miniature. Reading one unit at a time
(Lesson 5.2d) misses how units combine, so the atlas looks at whole activation
vectors: one 8-channel vector per image position, collected from many images,
laid out in 2-D so that similar vectors land close together, averaged over a
coarse grid, and turned back into an image cell by cell. Each cell then
answers "what does this region of activation space look like?" with a picture.

The network is the one Lesson 5.2c trains: 8 kernels of 5x5, ReLU, global
average pooling, and a linear layer over 4 line angles. This file carries a
compact copy of its data, windows, forward pass, and training, so the lesson
runs on its own and trains the same network as Lesson 5.2c. The layout is an
exact t-SNE written in NumPy, and the turn back into an image is feature
inversion: the input gradient of Lesson 5.2d, generalized from one channel to
a direction in channel space.

Source: Carter, Armstrong, Schubert, Johnson, Olah, "Activation Atlas", Distill
(2019), https://distill.pub/2019/activation-atlas/ (activations from many
images, laid out in 2-D by t-SNE or UMAP, averaged over a grid, and each
cell's average turned back into an image by feature inversion).

Prerequisites (must complete first)
----------------------------------
- Lesson 5.2d: Reading a Unit (feature visualization and the input gradient)
- Lesson 5.2c: A Tiny Convolutional Network (the network being mapped)

Learning Objectives
-------------------
After completing this lesson, you will be able to:
1. Collect activation vectors from a trained network and tag each with its
   image's class.
2. Write t-SNE's divergence and gradient by hand, check the gradient by central
   differences, and run the optimizer with early exaggeration and momentum.
3. Average vectors over a grid and invert each average to an image by gradient
   ascent on the input.
4. Judge a layout honestly: how often a cell's picture matches its majority
   class, and how often neighbouring cells agree.

The Big Idea
------------
A network does not use its channels one at a time. At each position it
produces a point in an 8-dimensional space, and nearby points mean similar
things to the layers above. t-SNE flattens that space to a page, keeping
neighbours together. Averaging the vectors in each cell of the page and
inverting the average shows what the network sees at that spot. The picture
is a hypothesis about the region, not proof: the layout is a lossy flattening,
and the inversion finds one input among many that point the same way.

Why It Matters
--------------
Distill's atlas shows that a vision network's concepts form regions with
neighbours, such as animals near animals, which no list of single units
reveals. Here the same pipeline runs on 4 line angles, a case where you know
the answer. About half the cells show an orientation that matches the majority
angle of the vectors in them, against 25% by chance, and neighbouring cells
share an orientation about twice as often as shuffled cells do. The layout
still mixes classes: about three in four vectors belong to their cell's
majority class, so a cell's picture describes its average, not each vector.

Stdlib Pain → Library Power-Up
------------------------------
Pain (this lesson):
- t-SNE is an exact O(n²) computation here: a perplexity search per point, a
  Student-t kernel over all pairs, and a gradient you write and check yourself.
  A missing kernel factor or factor of 4 still lays points out, to a layout
  that is wrong in a way no picture reveals, so you check it with central
  differences.
- Lines of code: ~35 for t-SNE, ~25 for the grid, ~40 for the inversion.
- Runtime: about 20 seconds for the demo (training dominates), measured with
  one BLAS thread.
- Common bugs: forgetting the ReLU mask in the inversion gradient, binning
  the layout into transposed cells, and reading a picture as a property of
  the whole region when the cell's vectors mix classes.

Power-Up (Lesson 5.3 — PyTorch):
- ``torch.autograd.grad`` gives the inversion gradient for any network and
  any direction with no hand derivation, and library t-SNE or UMAP lay out
  millions of vectors with approximate neighbours.
- Key APIs: ``torch.autograd.grad``.

Complexity Analysis
-------------------
Pain (manual), for n vectors of c channels, s t-SNE steps, g x g cells, s' inversion steps:
- Time:  O(s·n²) for the layout, O(n·c) for the grid, O(g²·s'·k²·c) for the inversions.
- Space: O(n²) for the pairwise kernel, O(g²·c) for the cell averages.

Execution
---------
Project-style:
    uv run python src/052e_activation_atlas.py

Script-style (self-contained deps):
    uv run --script src/052e_activation_atlas.py

On a busy machine, NumPy's BLAS threads compete with other work and can stretch
the demo from about 20 seconds to a minute or more; one thread avoids that, since the
matrices here are small:
    OPENBLAS_NUM_THREADS=1 uv run python src/052e_activation_atlas.py

Options:
    python src/052e_activation_atlas.py --doctest-only
    python src/052e_activation_atlas.py --no-doctest
    python src/052e_activation_atlas.py --index
    python src/052e_activation_atlas.py -v

Quality Gates
-------------
    uv run pytest src/052e_activation_atlas.py
    uv run ruff check src/052e_activation_atlas.py
    uv run ruff format --check src/052e_activation_atlas.py
    uv run mypy --strict src/052e_activation_atlas.py

Module Doctests
---------------
Two points always have t-SNE affinity one half each, whatever their positions:

>>> import numpy as np
>>> joint_probabilities(np.array([[0.0, 0.0], [3.0, 4.0]]), 1.0).tolist()
[[0.0, 0.5], [0.5, 0.0]]
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
    type GradFn = Callable[[Array, Array], Array]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON IDENTITY & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON_ID: Final[str] = "tier_5_lesson_2e_activation_atlas"

# Autograd gives the inversion gradient for free with PyTorch.
POWER_UP_AVAILABLE: Final[bool] = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Format: lesson_id -> reason

PREREQUISITES: Final[Mapping[str, str]] = {
    "tier_5_lesson_2d_reading_a_unit": "Feature visualization and the input gradient",
    "tier_5_lesson_2c_tiny_cnn": "The network being mapped",
}

DEPENDENCIES: Final[Mapping[str, str]] = {
    "tier_3_lesson_4a_convolution": "The sliding dot product the inversion gradient reverses",
}

SUCCESSORS: Final[Mapping[str, str]] = {
    "tier_5_lesson_3_pytorch": "Atlases on real networks, with autograd and library t-SNE",
}

DEPENDENTS: Final[Mapping[str, str]] = {}

RELATED: Final[Mapping[str, str]] = {
    "tier_5_lesson_2b_sparse_autoencoders": "Another way to split activation space into parts",
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
# STEP 1: Activation Vectors
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBE_ANGLES: Final[tuple[float, ...]] = tuple(np.arange(0.0, 180.0, 22.5).tolist())
INVERT_SIZE: Final[int] = 9


def line_detectors(k: int = 5) -> Params:
    """
    Return a two-channel network of hand-set kernels: vertical lines, then horizontal lines.

    Each ``k`` x ``k`` kernel has weight 4 along the middle column (channel 0)
    or middle row (channel 1) and -1 elsewhere, so a flat patch gives no
    response when ``k`` is 5. The bias and the linear layer are zeros; the
    lesson reads only the kernels.

    Examples
    --------
    >>> import numpy as np
    >>> Wc, bc, W2, b2 = line_detectors()
    >>> Wc.shape, Wc.sum(axis=0).tolist()
    ((25, 2), [0.0, 0.0])
    >>> Wc[:, 0].reshape(5, 5)[:, 2].tolist(), Wc[:, 1].reshape(5, 5)[2, :].tolist()
    ([4.0, 4.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 4.0, 4.0])
    >>> bool(np.array_equal(Wc[:, 0].reshape(5, 5).T, Wc[:, 1].reshape(5, 5)))
    True
    """
    vertical = np.full((k, k), -1.0)
    vertical[:, k // 2] = 4.0
    Wc = np.stack([vertical.reshape(-1), vertical.T.reshape(-1)], axis=1)
    return Wc, np.zeros(2), np.zeros((2, len(ANGLES))), np.zeros(len(ANGLES))


def grating(angle: float, period: float = 6.0, phase: float = 0.0, size: int = SIZE) -> Array:
    """
    Return a ``size`` x ``size`` sine grating whose stripes run along ``angle`` degrees.

    Pixel values are ``cos(2π d / period + phase)`` with ``d`` the distance
    across the stripes, so the stripes repeat every ``period`` pixels. Angle 0
    gives horizontal stripes, which vary only down the rows.

    Examples
    --------
    >>> import numpy as np
    >>> g0, g90 = grating(0.0), grating(90.0)
    >>> bool(np.allclose(g0, g0[:, :1])), bool(np.allclose(g90, g90[:1, :]))
    (True, True)
    >>> float(g0[0, 0]), round(float(g0[3, 0]), 6), grating(0.0, size=9).shape
    (1.0, -1.0, (9, 9))

    A phase of π flips the sign of every pixel:

    >>> bool(np.allclose(grating(45.0, phase=np.pi), -grating(45.0)))
    True
    """
    xx, yy = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
    t = np.deg2rad(angle)
    return np.asarray(
        np.cos(2 * np.pi * (-xx * np.sin(t) + yy * np.cos(t)) / period + phase),
        dtype=np.float64,
    )


def dominant_angle(
    image: Array,
    angles: Sequence[float] = PROBE_ANGLES,
    periods: Sequence[float] = (4.0, 6.0, 8.0),
) -> float:
    """
    Return the angle in ``angles`` whose gratings ``image`` correlates with most.

    The score of an angle sums, over ``periods``, the length of the pair
    ``(image · cosine grating, image · sine grating)``, so a stripe pattern
    scores the same wherever its stripes fall. The image's mean is removed
    first, and the gratings take the image's size.

    Examples
    --------
    A grating scores highest at its own angle, whatever its phase or size:

    >>> dominant_angle(grating(45.0)), dominant_angle(grating(90.0, phase=1.0))
    (45.0, 90.0)
    >>> dominant_angle(grating(135.0, size=9), angles=ANGLES)
    135.0

    A diagonal drawn from the top left to the bottom right is the 45 degree class:

    >>> import numpy as np
    >>> dominant_angle(np.eye(9), angles=ANGLES)
    45.0
    """
    x = image - image.mean()
    size = image.shape[0]
    scores = [
        sum(
            np.hypot(
                (x * grating(a, p, size=size)).sum(),
                (x * grating(a, p, phase=-np.pi / 2, size=size)).sum(),
            )
            for p in periods
        )
        for a in angles
    ]
    return float(angles[int(np.argmax(scores))])


def activation_vectors(
    params: Params,
    images: Array,
    labels: Labels,
    n_vectors: int = 300,
    top_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[Array, Labels]:
    """
    Return up to ``n_vectors`` activation vectors and the class of the image each came from.

    A vector is the ReLU'd response of every kernel at one window position, so
    its length is the number of channels. Only vectors that are not all zero
    qualify, and of those only the strongest ``top_fraction`` by length, which
    drops the positions that see background noise. A seeded draw without
    replacement picks ``n_vectors`` of them.

    Examples
    --------
    One image with 1x1 kernels of weights 1 and -1 has two positions, 2 and -3.
    They give the vectors (2, 0) and (0, 3), both from class 3:

    >>> import numpy as np
    >>> params = (np.array([[1.0, -1.0]]), np.zeros(2), np.zeros((2, 4)), np.zeros(4))
    >>> images, labels = np.array([[[2.0, -3.0]]]), np.array([3])
    >>> vectors, tags = activation_vectors(params, images, labels, top_fraction=1.0)
    >>> sorted(vectors.tolist()), tags.tolist()
    ([[0.0, 3.0], [2.0, 0.0]], [3, 3])

    Keeping the strongest half leaves only the longer vector, (0, 3):

    >>> activation_vectors(params, images, labels, top_fraction=0.5)[0].tolist()
    [[0.0, 3.0]]

    A network with no response gives no vectors:

    >>> dead = (params[0], np.full(2, -100.0), params[2], params[3])
    >>> activation_vectors(dead, images, labels)[0].shape
    (0, 2)
    """
    k = round(len(params[0]) ** 0.5)
    A = forward(params, patches(images, k))[1]
    flat = A.reshape(-1, A.shape[2])
    tags = np.repeat(labels, A.shape[1])
    norms = np.linalg.norm(flat, axis=1)
    live = np.flatnonzero(norms > 0)
    if live.size == 0:
        return np.zeros((0, flat.shape[1])), np.zeros(0, dtype=np.int64)
    cutoff = np.quantile(norms[live], 1.0 - top_fraction)
    strong = live[norms[live] >= cutoff]
    pick = np.random.default_rng(seed).choice(strong, min(n_vectors, strong.size), replace=False)
    return np.asarray(flat[pick], dtype=np.float64), np.asarray(tags[pick], dtype=np.int64)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: The Layout, Exact t-SNE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def squared_distances(x: Array) -> Array:
    """
    Return the matrix of squared Euclidean distances between the rows of ``x``.

    Examples
    --------
    A 3-4-5 triangle's legs and hypotenuse give 9, 16, and 25:

    >>> import numpy as np
    >>> d = squared_distances(np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]))
    >>> d.tolist()
    [[0.0, 9.0, 16.0], [9.0, 0.0, 25.0], [16.0, 25.0, 0.0]]
    """
    sq = (x * x).sum(axis=1)
    return np.maximum(sq[:, None] + sq[None, :] - 2.0 * x @ x.T, 0.0)


def joint_probabilities(x: Array, perplexity: float) -> Array:
    """
    Return t-SNE's symmetric affinities between the rows of ``x``, summing to 1.

    Each point ``i`` spreads a Gaussian over the others with a bandwidth found
    by binary search, so that the entropy of its conditional distribution is
    ``log(perplexity)``: the point has about ``perplexity`` effective
    neighbours. The conditionals are then averaged, ``(P + Pᵀ) / 2n``.

    Examples
    --------
    The affinities sum to 1, are symmetric, and have a zero diagonal:

    >>> import numpy as np
    >>> x = np.random.default_rng(0).normal(size=(30, 5))
    >>> p = joint_probabilities(x, perplexity=5.0)
    >>> round(float(p.sum()), 6), bool(np.allclose(p, p.T)), float(np.diag(p).max())
    (1.0, True, 0.0)

    Two clusters 100 apart share no affinity, since each point's neighbours
    are all in its own cluster:

    >>> rng = np.random.default_rng(1)
    >>> two = np.vstack([rng.normal(0.0, 0.1, (10, 2)), rng.normal(100.0, 0.1, (10, 2))])
    >>> bool(joint_probabilities(two, perplexity=3.0)[:10, 10:].sum() < 1e-6)
    True
    """
    d = squared_distances(x)
    n = len(x)
    p = np.zeros((n, n))
    target = np.log(perplexity)
    for i in range(n):
        lo, hi, beta = 0.0, np.inf, 1.0
        di = np.delete(d[i], i)
        for _ in range(64):
            pi = np.exp(-(di - di.min()) * beta)
            pi /= pi.sum()
            entropy = -(pi * np.log(np.maximum(pi, 1e-12))).sum()
            if abs(entropy - target) < 1e-5:
                break
            if entropy > target:
                lo, beta = beta, beta * 2.0 if hi == np.inf else (beta + hi) / 2.0
            else:
                hi, beta = beta, (beta + lo) / 2.0
        p[i, np.arange(n) != i] = pi
    p = (p + p.T) / (2.0 * n)
    return np.asarray(p / p.sum(), dtype=np.float64)


def tsne_kl_and_grad(P: Array, Y: Array) -> tuple[float, Array]:
    """
    Return the divergence KL(P ‖ Q) of the layout ``Y`` from ``P``, and its gradient in ``Y``.

    The layout's affinities use a Student-t kernel,
    ``w[i, j] = 1 / (1 + |y_i - y_j|²)``, normalized to ``Q = w / Σw``. The
    gradient is::

        dKL/dy_i = 4 Σ_j (P[i, j] - Q[i, j]) · w[i, j] · (y_i - y_j)

    Pairs that ``P`` wants nearer than ``Q`` has them pull together, and the
    kernel ``w`` weakens the push and pull of pairs already far apart.

    Examples
    --------
    Two points always have ``Q`` of one half each, which matches ``P`` of one
    half each, so the divergence and the gradient are 0 wherever they sit:

    >>> import numpy as np
    >>> P = np.array([[0.0, 0.5], [0.5, 0.0]])
    >>> kl, grad = tsne_kl_and_grad(P, np.array([[0.0, 0.0], [1.0, 0.0]]))
    >>> kl, float(np.abs(grad).max())
    (0.0, 0.0)

    Three equally likely pairs on an equilateral triangle match as well:

    >>> P3 = (np.ones((3, 3)) - np.eye(3)) / 6.0
    >>> triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    >>> kl, grad = tsne_kl_and_grad(P3, triangle)
    >>> abs(kl) < 1e-12, bool(np.abs(grad).max() < 1e-12)
    (True, True)

    Stretching one point out along a line breaks the match. The divergence is
    positive, and the forces still cancel, since they act in equal and
    opposite pairs:

    >>> line = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    >>> kl, grad = tsne_kl_and_grad(P3, line)
    >>> kl > 0.01, bool(np.abs(grad).max() > 0.01), bool(np.abs(grad.sum(axis=0)).max() < 1e-12)
    (True, True, True)
    """
    w = 1.0 / (1.0 + squared_distances(Y))
    np.fill_diagonal(w, 0.0)
    Q = np.maximum(w / w.sum(), 1e-12)
    seen = P > 0
    kl = float((P[seen] * np.log(P[seen] / Q[seen])).sum())
    forces = (P - Q) * w
    grad = 4.0 * (forces.sum(axis=1)[:, None] * Y - forces @ Y)
    return kl, np.asarray(grad, dtype=np.float64)


def gradient_error(grad_fn: GradFn, P: Array, Y: Array, eps: float = 1e-6) -> float:
    """
    Return the largest gap between ``grad_fn``'s gradient and central differences of the divergence.

    ``grad_fn(P, Y)`` has the shape of ``tsne_kl_and_grad``'s second output.
    Each coordinate of ``Y`` is nudged by ±``eps`` and the change in the
    divergence is differenced. An exact gradient gives an error far below
    ``eps``; a wrong one gives an error of the size of the missing term.

    Examples
    --------
    On 5 float64 points, the hand-derived gradient agrees with the numeric one:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> P = joint_probabilities(rng.normal(size=(5, 3)), perplexity=2.0)
    >>> Y = rng.normal(size=(5, 2))
    >>> def exact(P, Y):
    ...     return tsne_kl_and_grad(P, Y)[1]
    >>> gradient_error(exact, P, Y) < 1e-7
    True

    The deliberate break: a gradient that drops the Student-t kernel factor
    ``w`` pulls and pushes pairs at the wrong strength, and is far off:

    >>> def no_kernel(P, Y):
    ...     w = 1.0 / (1.0 + squared_distances(Y))
    ...     np.fill_diagonal(w, 0.0)
    ...     Q = w / w.sum()
    ...     forces = P - Q
    ...     np.fill_diagonal(forces, 0.0)
    ...     return 4.0 * (forces.sum(axis=1)[:, None] * Y - forces @ Y)
    >>> gradient_error(no_kernel, P, Y) > 1e-3
    True

    So does one that forgets the factor of 4:

    >>> def no_four(P, Y):
    ...     return tsne_kl_and_grad(P, Y)[1] / 4.0
    >>> gradient_error(no_four, P, Y) > 1e-3
    True
    """
    analytic = grad_fn(P, Y)
    numeric = np.zeros_like(analytic)
    for index in np.ndindex(*Y.shape):
        values = []
        for step in (eps, -eps):
            moved = Y.copy()
            moved[index] += step
            values.append(tsne_kl_and_grad(P, moved)[0])
        numeric[index] = (values[0] - values[1]) / (2 * eps)
    return float(np.max(np.abs(analytic - numeric)))


def tsne(
    x: Array,
    perplexity: float = 30.0,
    steps: int = 750,
    seed: int = 0,
    learning_rate: float = 100.0,
) -> Array:
    """
    Return a 2-D layout of the rows of ``x`` by gradient descent on t-SNE's divergence.

    The layout starts as Gaussian noise of standard deviation 1e-4. For the
    first 100 steps the affinities ``P`` are multiplied by 12 (early
    exaggeration), which pulls clusters tight so they can move past each
    other. Momentum is 0.5 for the first 250 steps and 0.8 after. The
    perplexity is capped at a third of the point count.

    Examples
    --------
    Two clusters 20 apart in 5-D come out as two separate blobs in 2-D, a gap
    wider than twice the farthest point from its own centre:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = np.vstack([rng.normal(0.0, 1.0, (15, 5)), rng.normal(20.0, 1.0, (15, 5))])
    >>> y = tsne(x, perplexity=5.0, steps=300)
    >>> a, b = y[:15], y[15:]
    >>> gap = np.linalg.norm(a.mean(axis=0) - b.mean(axis=0))
    >>> reach = max(np.linalg.norm(a - a.mean(axis=0), axis=1).max(),
    ...             np.linalg.norm(b - b.mean(axis=0), axis=1).max())
    >>> y.shape, bool(gap > 2 * reach)
    ((30, 2), True)

    The layout fits ``P`` better than the starting noise, and repeats for one seed:

    >>> P = joint_probabilities(x, 5.0)
    >>> start = np.random.default_rng(0).normal(0.0, 1e-4, (30, 2))
    >>> bool(tsne_kl_and_grad(P, y)[0] < tsne_kl_and_grad(P, start)[0])
    True
    >>> bool(np.array_equal(y, tsne(x, perplexity=5.0, steps=300)))
    True
    """
    rng = np.random.default_rng(seed)
    P = joint_probabilities(x, min(perplexity, (len(x) - 1) / 3))
    Y = rng.normal(0.0, 1e-4, (len(x), 2))
    velocity = np.zeros_like(Y)
    for step in range(steps):
        exaggeration = 12.0 if step < 100 else 1.0
        grad = tsne_kl_and_grad(exaggeration * P, Y)[1]
        velocity = (0.5 if step < 250 else 0.8) * velocity - learning_rate * grad
        Y += velocity
        Y -= Y.mean(axis=0)
    return Y


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: Average Over a Grid
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def grid_average(
    points: Array,
    vectors: Array,
    tags: Labels,
    cells: int = 6,
) -> tuple[Array, Labels, Labels]:
    """
    Return each grid cell's mean vector, point count, and per-class counts.

    The grid is ``cells`` x ``cells`` over the bounding box of ``points``. A
    point at ``(x, y)`` falls in column ``x`` and row ``y`` of the grid, and
    the largest coordinate falls in the last cell. The outputs are indexed
    ``[row, column]``. An empty cell has count 0 and a mean of zeros.

    Examples
    --------
    Three points on a 2x2 grid over (0, 0) to (1, 1). The first two share the
    top left cell, whose mean is (1 + 3) / 2 = 2 in the first channel; the
    third sits alone at the bottom right:

    >>> import numpy as np
    >>> points = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0]])
    >>> vectors = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 5.0]])
    >>> means, counts, by_class = grid_average(points, vectors, np.array([0, 0, 2]), cells=2)
    >>> counts.tolist()
    [[2, 0], [0, 1]]
    >>> means[0, 0].tolist(), means[1, 1].tolist(), means[0, 1].tolist()
    ([2.0, 0.0], [0.0, 5.0], [0.0, 0.0])
    >>> by_class[0, 0].tolist(), by_class[1, 1].tolist()
    ([2, 0, 0, 0], [0, 0, 1, 0])

    A point with a large x and a small y lands in row 0, column 1, so the
    outputs are indexed ``[y, x]``:

    >>> pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> grid_average(pts, np.ones((3, 1)), np.array([0, 1, 2]), 2)[1].tolist()
    [[1, 1], [1, 0]]
    """
    lo = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - lo, 1e-12)
    index = np.minimum(((points - lo) / span * cells).astype(np.int64), cells - 1)
    col, row = index[:, 0], index[:, 1]
    sums = np.zeros((cells, cells, vectors.shape[1]))
    np.add.at(sums, (row, col), vectors)
    counts = np.zeros((cells, cells), dtype=np.int64)
    np.add.at(counts, (row, col), 1)
    by_class = np.zeros((cells, cells, len(ANGLES)), dtype=np.int64)
    np.add.at(by_class, (row, col, tags), 1)
    return sums / np.maximum(counts, 1)[:, :, None], counts, by_class


def majority_angles(by_class: Labels) -> Array:
    """
    Return each cell's majority class as an angle in degrees, or -1 for an empty cell.

    A tie goes to the smaller angle.

    Examples
    --------
    >>> import numpy as np
    >>> by_class = np.array([[[0, 3, 1, 0], [0, 0, 0, 0]], [[0, 0, 2, 2], [4, 0, 0, 0]]])
    >>> majority_angles(by_class).tolist()
    [[45.0, -1.0], [90.0, 0.0]]
    """
    angles = np.asarray(ANGLES)[by_class.argmax(axis=2)]
    return np.asarray(np.where(by_class.sum(axis=2) > 0, angles, -1.0), dtype=np.float64)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: Feature Inversion Toward a Direction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def centre_response(params: Params, image: Array, direction: Array, jitter: int = 0) -> float:
    """
    Return ``direction · ReLU(Z)`` at the centre window position of ``image``, averaged over shifts.

    The centre position is the middle one of the ``(H - k + 1)`` x ``(W - k + 1)``
    grid of windows. With ``jitter`` above 0 the score averages over every
    shift of the image by up to ``jitter`` pixels in each direction, which is
    the quantity ``invert`` climbs.

    Examples
    --------
    With 1x1 kernels of weights 2 and 3 and biases 0 and -5, a pixel of 1 gives
    the responses 2 and -2. The second is clipped, so the direction (1, 1)
    scores 2:

    >>> import numpy as np
    >>> params = (np.array([[2.0, 3.0]]), np.array([0.0, -5.0]), np.zeros((2, 4)), np.zeros(4))
    >>> centre_response(params, np.array([[1.0]]), np.array([1.0, 1.0]))
    2.0

    On a 3x3 image, a 1x1 kernel of weight 1 reads the middle pixel. Shifting
    by one pixel in each direction brings each of the 9 pixels to the middle
    once, so the average is their mean:

    >>> params = (np.ones((1, 1)), np.zeros(1), np.zeros((1, 4)), np.zeros(4))
    >>> image = np.arange(9.0).reshape(3, 3)
    >>> one = np.ones(1)
    >>> centre_response(params, image, one), centre_response(params, image, one, jitter=1)
    (4.0, 4.0)
    >>> round(centre_response(params, np.eye(3), np.ones(1), jitter=1), 6)
    0.333333
    """
    Wc, bc = params[0], params[1]
    k = round(len(Wc) ** 0.5)
    r, c = (image.shape[0] - k + 1) // 2, (image.shape[1] - k + 1) // 2
    shifts = range(-jitter, jitter + 1)
    scores = []
    for dy in shifts:
        for dx in shifts:
            moved = np.roll(image, (dy, dx), axis=(0, 1))
            Z = moved[r : r + k, c : c + k].reshape(-1) @ Wc + bc
            scores.append(float(np.maximum(Z, 0.0) @ direction))
    return float(np.mean(scores))


def activation_gradient(params: Params, image: Array, direction: Array) -> Array:
    """
    Return the gradient of ``centre_response`` with respect to ``image``.

    Lesson 5.2d's input gradient for one channel, generalized to a direction
    ``d`` over all channels::

        dZ = d · (Z > 0)                (ReLU mask on each channel)
        dwindow = Wc @ dZ               (each kernel back onto the pixels it covers)

    Only the ``k`` x ``k`` window at the centre position has a nonzero gradient.

    Examples
    --------
    With 1x1 kernels of weights 2 and 3 and biases 0 and -5, a pixel of 1
    activates only the first channel. The ReLU blocks the second, so the
    gradient is 2, not 2 + 3:

    >>> import numpy as np
    >>> params = (np.array([[2.0, 3.0]]), np.array([0.0, -5.0]), np.zeros((2, 4)), np.zeros(4))
    >>> activation_gradient(params, np.array([[1.0]]), np.array([1.0, 1.0])).tolist()
    [[2.0]]

    A 3x3 kernel counting 1 to 9 on a 5x5 image of ones has a 3x3 grid of
    windows, whose centre is the window at rows and columns 1 to 3. The
    gradient is the kernel placed there, and zero elsewhere:

    >>> weights = np.arange(1.0, 10.0)[:, None]
    >>> params = (weights, np.zeros(1), np.zeros((1, 4)), np.zeros(4))
    >>> g = activation_gradient(params, np.ones((5, 5)), np.array([1.0]))
    >>> g[1:4, 1:4].reshape(-1).tolist()
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    >>> float(g.sum()), float(g[0].max()), float(g[:, 4].max())
    (45.0, 0.0, 0.0)

    A negative response blocks the whole gradient:

    >>> float(np.abs(activation_gradient(params, -np.ones((5, 5)), np.array([1.0]))).max())
    0.0
    """
    Wc, bc = params[0], params[1]
    k = round(len(Wc) ** 0.5)
    r, c = (image.shape[0] - k + 1) // 2, (image.shape[1] - k + 1) // 2
    Z = image[r : r + k, c : c + k].reshape(-1) @ Wc + bc
    grad = np.zeros_like(image)
    grad[r : r + k, c : c + k] = (Wc @ (direction * (Z > 0))).reshape(k, k)
    return grad


def invert(
    params: Params,
    direction: Array,
    steps: int = 150,
    step_size: float = 0.1,
    jitter: int = 2,
    seed: int = 0,
    size: int = INVERT_SIZE,
) -> Array:
    """
    Return a ``size`` x ``size`` image that climbs ``direction · activation`` at its centre.

    The image starts as Gaussian noise of standard deviation 0.01. Each step
    shifts it by a random offset of up to ``jitter`` pixels in each direction,
    so the result has to work wherever it lands, takes ``activation_gradient``
    there, shifts the gradient back, and adds it scaled so its largest entry
    is ``step_size``. Pixels clip to [-1, 1]. Only the direction matters, not
    its length. A direction no kernel responds to has zero gradient, and its
    image stays noise.

    Examples
    --------
    Inverting toward channel 0 of the hand-set detectors, which fires on
    vertical lines, gives an image that correlates most with vertical stripes
    (90 degrees), and channel 1 gives horizontal ones (0 degrees):

    >>> import numpy as np
    >>> params = line_detectors()
    >>> vertical = invert(params, np.array([1.0, 0.0]))
    >>> horizontal = invert(params, np.array([0.0, 1.0]))
    >>> vertical.shape, float(np.abs(vertical).max()) <= 1.0
    ((9, 9), True)
    >>> dominant_angle(vertical, angles=ANGLES), dominant_angle(horizontal, angles=ANGLES)
    (90.0, 0.0)

    The image raises the response far above the noise it started from, and
    repeats exactly for one seed:

    >>> noise = np.random.default_rng(0).normal(0.0, 0.01, (9, 9))
    >>> direction = np.array([1.0, 0.0])
    >>> score = centre_response(params, vertical, direction, jitter=2)
    >>> bool(score > 10 * centre_response(params, noise, direction, jitter=2))
    True
    >>> bool(np.array_equal(vertical, invert(params, direction)))
    True

    A direction of zeros never moves off the noise:

    >>> float(np.abs(invert(params, np.zeros(2), steps=20)).max()) < 0.1
    True
    """
    rng = np.random.default_rng(seed)
    image = rng.normal(0.0, 0.01, (size, size))
    for _ in range(steps):
        dy, dx = (int(v) for v in rng.integers(-jitter, jitter + 1, 2))
        shifted = np.roll(image, (dy, dx), axis=(0, 1))
        grad = np.roll(activation_gradient(params, shifted, direction), (-dy, -dx), axis=(0, 1))
        image = np.clip(image + step_size * grad / (np.abs(grad).max() + 1e-9), -1.0, 1.0)
    return image


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5: The Atlas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True)
class Atlas:
    """One atlas: the layout, the grid, and one inverted image per occupied cell.

    Attributes
    ----------
    layout : numpy.ndarray
        The 2-D t-SNE position of each activation vector, shape ``(n, 2)``.
    tags : numpy.ndarray
        Class index of the image each vector came from, shape ``(n,)``.
    counts : numpy.ndarray
        Vectors per cell, shape ``(cells, cells)``.
    by_class : numpy.ndarray
        Vectors per cell and class, shape ``(cells, cells, 4)``.
    orientation : numpy.ndarray
        Dominant angle of each cell's inverted image in degrees, from
        ``ANGLES``, or -1 for an empty cell.
    images : numpy.ndarray
        Each cell's inverted image, shape ``(cells, cells, size, size)``;
        zeros for an empty cell.
    """

    layout: Array
    tags: Labels
    counts: Labels
    by_class: Labels
    orientation: Array
    images: Array


def atlas(
    params: Params,
    images: Array,
    labels: Labels,
    n_vectors: int = 300,
    cells: int = 6,
    perplexity: float = 30.0,
    tsne_steps: int = 750,
    inversion_steps: int = 150,
    seed: int = 0,
) -> Atlas:
    """
    Return the atlas of ``params`` over ``images``: collect, lay out, average, and invert.

    The steps are ``activation_vectors``, ``tsne``, ``grid_average``, and for
    every occupied cell ``invert`` toward the cell's mean vector followed by
    ``dominant_angle`` over ``ANGLES``. Raises ``ValueError`` when fewer than 4
    activation vectors qualify, too few to lay out.

    Examples
    --------
    On the hand-set detectors, every vector lands in one cell, the empty cells
    are marked -1, and each occupied cell has a picture and an angle:

    >>> import numpy as np
    >>> images, labels = line_dataset(60, seed=1)
    >>> a = atlas(line_detectors(), images, labels, n_vectors=60, cells=3, perplexity=10.0,
    ...           tsne_steps=100, inversion_steps=20)
    >>> a.layout.shape, a.images.shape
    ((60, 2), (3, 3, 9, 9))
    >>> int(a.counts.sum()), bool((a.by_class.sum(axis=2) == a.counts).all())
    (60, True)
    >>> empty = a.counts == 0
    >>> bool((a.orientation[empty] == -1.0).all())
    True
    >>> bool(np.isin(a.orientation[~empty], ANGLES).all())
    True

    A network with no response has no vectors to lay out:

    >>> dead = (np.zeros((25, 2)), np.full(2, -1.0), np.zeros((2, 4)), np.zeros(4))
    >>> atlas(dead, images, labels)
    Traceback (most recent call last):
        ...
    ValueError: fewer than 4 activation vectors qualify
    """
    vectors, tags = activation_vectors(params, images, labels, n_vectors, seed=seed)
    if len(vectors) < 4:
        raise ValueError("fewer than 4 activation vectors qualify")
    layout = tsne(vectors, perplexity, tsne_steps, seed)
    means, counts, by_class = grid_average(layout, vectors, tags, cells)
    pictures = np.zeros((cells, cells, INVERT_SIZE, INVERT_SIZE))
    orientation = np.full((cells, cells), -1.0)
    for row, col in zip(*np.nonzero(counts), strict=True):
        pictures[row, col] = invert(params, means[row, col], inversion_steps, seed=seed)
        orientation[row, col] = dominant_angle(pictures[row, col], angles=ANGLES)
    return Atlas(layout, tags, counts, by_class, orientation, pictures)


def match_rate(orientation: Array, majority: Array) -> float:
    """
    Return the fraction of occupied cells whose picture shows the cell's majority angle.

    A cell is occupied when its ``orientation`` is not -1. Chance for four
    equally likely angles is 0.25. A grid with no occupied cell gives 0.

    Examples
    --------
    Two of three occupied cells match; the -1 cell does not count:

    >>> import numpy as np
    >>> orientation = np.array([[0.0, 45.0], [90.0, -1.0]])
    >>> majority = np.array([[0.0, 90.0], [90.0, -1.0]])
    >>> round(match_rate(orientation, majority), 4)
    0.6667
    >>> match_rate(np.full((2, 2), -1.0), np.full((2, 2), -1.0))
    0.0
    """
    occupied = orientation >= 0
    if not occupied.any():
        return 0.0
    return float(np.mean(orientation[occupied] == majority[occupied]))


def purity(by_class: Labels) -> float:
    """
    Return the fraction of vectors that belong to their cell's majority class.

    A value of 1 means every cell holds one class; four equal classes in every
    cell give 0.25.

    Examples
    --------
    One cell holds 3 of class 0 and 1 of class 1, another holds 2 of class 2:
    5 of the 6 vectors match their cell's majority.

    >>> import numpy as np
    >>> round(purity(np.array([[[3, 1, 0, 0], [0, 0, 2, 0]]])), 4)
    0.8333
    """
    return float(by_class.max(axis=2).sum() / max(int(by_class.sum()), 1))


def neighbour_agreement(grid: Array) -> float:
    """
    Return the fraction of side-by-side occupied cell pairs that hold the same value.

    Cells equal to -1 are empty and form no pair. A grid with no pair gives 0.

    Examples
    --------
    In a 2x2 grid with 0 on top and 90 below, the two pairs across a row agree
    and the two pairs down a column do not:

    >>> import numpy as np
    >>> neighbour_agreement(np.array([[0.0, 0.0], [90.0, 90.0]]))
    0.5

    Empty cells break pairs:

    >>> neighbour_agreement(np.array([[0.0, -1.0], [-1.0, 0.0]]))
    0.0
    """
    pairs = [(grid[:, :-1], grid[:, 1:]), (grid[:-1, :], grid[1:, :])]
    equal = total = 0
    for a, b in pairs:
        both = (a >= 0) & (b >= 0)
        equal += int(((a == b) & both).sum())
        total += int(both.sum())
    return equal / total if total else 0.0


def shuffled_agreement(grid: Array, trials: int = 200, seed: int = 0) -> float:
    """
    Return the mean ``neighbour_agreement`` after shuffling values among the occupied cells.

    The shuffle keeps which cells are occupied and how often each value occurs,
    so the result is what neighbours would agree by luck.

    Examples
    --------
    Four cells, two of each value, give each pair a one in three chance of
    agreeing:

    >>> import numpy as np
    >>> luck = shuffled_agreement(np.array([[0.0, 0.0], [90.0, 90.0]]), trials=2000)
    >>> abs(luck - 1 / 3) < 0.05
    True

    A grid of one value agrees every time:

    >>> shuffled_agreement(np.full((2, 2), 45.0), trials=5)
    1.0
    """
    rng = np.random.default_rng(seed)
    occupied = grid >= 0
    values = grid[occupied]
    scores = []
    for _ in range(trials):
        mixed = np.full_like(grid, -1.0)
        mixed[occupied] = rng.permutation(values)
        scores.append(neighbour_agreement(mixed))
    return float(np.mean(scores))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEEING IT: ASCII Images (stdlib)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAMP: Final[str] = " .:-=+*#%@"
# Characters as they look on screen, where rows run downward: the 45 degree
# class runs from top left to bottom right.
LINE_CHARS: Final[Mapping[float, str]] = {0.0: "-", 45.0: "\\", 90.0: "|", 135.0: "/"}


def draw_grid(angles: Array) -> str:
    r"""
    Draw a grid of angles as line characters, one per cell, with ``.`` for an empty cell.

    The characters are ``-`` for 0, ``\`` for 45, ``|`` for 90, and ``/`` for
    135 degrees, as the lines look on screen. An angle of -1 marks an empty
    cell.

    Examples
    --------
    >>> import numpy as np
    >>> print(draw_grid(np.array([[0.0, 45.0], [90.0, -1.0], [135.0, 0.0]])))
    - \
    | .
    / -
    """
    return "\n".join(" ".join(LINE_CHARS.get(float(a), ".") for a in row) for row in angles)


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
      LESSON INDEX: tier_5_lesson_2e_activation_atlas
    ...
    Prerequisites (hard deps):
       → tier_5_lesson_2d_reading_a_unit: Feature visualization and the input gradient
       → tier_5_lesson_2c_tiny_cnn: The network being mapped
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
    n_images: int = 400,
    n_vectors: int = 300,
    cells: int = 6,
    tsne_steps: int = 750,
    inversion_steps: int = 150,
) -> None:
    """
    Train the network, build its atlas, print both grids and their agreement, and draw two cells.

    Examples
    --------
    >>> demo(steps=200, n_train=300, channels=4, n_images=30, n_vectors=40, cells=3,
    ...      tsne_steps=100, inversion_steps=10)  # doctest: +ELLIPSIS
    ═...
    Inverted image's orientation    Majority class of the cell's vectors
    ...
    """
    print("═" * 70)
    print(f"  {LESSON_ID.upper().replace('_', ' ')}")
    print("═" * 70)
    print()
    train_images, train_labels = line_dataset(n_train, seed=0)
    P = patches(train_images)
    params = train(P, train_labels, steps=steps, batch=batch, channels=channels)
    accuracy = float(np.mean(forward(params, P)[3].argmax(axis=1) == train_labels))
    print(f"train accuracy {accuracy:.3f}")
    print()
    images, labels = line_dataset(n_images, seed=1)
    a = atlas(params, images, labels, n_vectors, cells, 30.0, tsne_steps, inversion_steps)
    majority = majority_angles(a.by_class)
    left = f"Inverted image's orientation{'':4}"
    print(f"{left}Majority class of the cell's vectors")
    for mine, theirs in zip(
        draw_grid(a.orientation).splitlines(), draw_grid(majority).splitlines(), strict=True
    ):
        print(f"{mine:<{len(left)}}{theirs}")
    print()
    print("Characters: '-' 0 deg, '\\' 45 deg, '|' 90 deg, '/' 135 deg, '.' empty cell")
    print()
    occupied = int((a.counts > 0).sum())
    print(f"{len(a.tags)} vectors from {n_images} images fill {occupied} of {cells * cells} cells")
    print(f"purity: {purity(a.by_class):.0%} of vectors belong to their cell's majority class")
    hits = int(np.sum((a.orientation == majority) & (a.orientation >= 0)))
    print(
        f"match: {hits} of {occupied} cells show an inverted orientation equal to the "
        f"majority angle ({match_rate(a.orientation, majority):.0%}; chance is 25%)"
    )
    print(
        f"neighbours: {neighbour_agreement(a.orientation):.0%} of adjacent cell pairs share "
        f"an orientation (shuffled cells: {shuffled_agreement(a.orientation):.0%})"
    )
    densest = np.unravel_index(int(a.counts.argmax()), a.counts.shape)
    other = np.argwhere((a.counts > 0) & (a.orientation != a.orientation[densest]))
    picks = [densest]
    if len(other):
        picks.append(tuple(other[int(a.counts[tuple(other.T)].argmax())]))
    for row, col in picks:
        print()
        print(
            f"Cell (row {row}, column {col}): {a.counts[row, col]} vectors, "
            f"majority {majority[row, col]:g} deg, inverted image {a.orientation[row, col]:g} deg"
        )
        print()
        print(shade(a.images[row, col]))


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
