#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<4.0"
# dependencies = ["numpy>=2.5.4"]
# ///
"""
Lesson 5.2g: How to Read t-SNE Without Being Fooled (side quest).

You run the exact t-SNE of Lesson 5.2e on synthetic data whose true shape you
know, and measure five ways a t-SNE picture misleads. The same two clusters
look joined or split depending on perplexity. A cluster ten times more spread
out in the data comes out the same size as its tight neighbour. Clusters at
gaps of 10 and 50 come out evenly spaced. Pure noise comes out clumpy. And a
run stopped early leaves pinched shapes that look like structure. Each pitfall
is one function returning one number, and an ASCII scatter plot shows the
picture behind it.

Source: Wattenberg, Viegas, Johnson, "How to Use t-SNE Effectively", Distill
(2016), https://distill.pub/2016/misread-tsne/ ; van der Maaten and Hinton,
"Visualizing Data using t-SNE", Journal of Machine Learning Research 9 (2008).

Prerequisites (must complete first)
----------------------------------
- Lesson 5.2e: An Activation Atlas (exact t-SNE written by hand, the method under test)

Learning Objectives
-------------------
After completing this lesson, you will be able to:
1. Sweep perplexity and measure how cluster separation changes with it.
2. Show that a t-SNE layout does not preserve cluster size or the distance
   between clusters, and put a number on each.
3. Show that noise can look clumpy, and that a layout stopped early is not
   a converged one.
4. Check every metric by breaking it: a gate that cannot fail proves nothing.

The Big Idea
------------
t-SNE keeps each point's neighbours, and nothing else. Neighbourhoods are
measured relative to a perplexity, so a dense and a sparse cluster both get
the same room, and gaps wider than a neighbourhood carry no force. A picture
is therefore a statement about who is near whom at one scale, and sizes,
gaps, and shapes read off it are properties of the optimizer as much as of
the data.

Why It Matters
--------------
You will plot embeddings, activations, and sparse-autoencoder features with
t-SNE and be tempted to read cluster sizes, distances, and shapes as
findings. Here the data's true sizes and gaps are known, so you see how far
off such a reading can be: a spread ratio of about 10 in the data comes out
near 1, and a gap ratio of about 5 comes out near 1. The remedy is the
Distill article's: run several perplexities, run to convergence, and compare
against noise before believing a shape.

Stdlib Pain → Library Power-Up
------------------------------
Pain (this lesson):
- The t-SNE core is the O(n^2) NumPy code of Lesson 5.2e, copied here so the
  lesson runs on its own. Each pitfall runs it several times, and the demo
  runs it about twenty times on 100 to 150 points.
- Lines of code: ~60 for the t-SNE core, ~90 for the five experiments and
  the plot.
- Runtime: about 10 seconds for the demo, measured with one BLAS thread.
- Common bugs: reading a layout at one perplexity, measuring a metric on
  the layout without measuring it on the input, and trusting a gradient
  that was never checked against central differences.

Power-Up (no lesson yet):
- scikit-learn's ``sklearn.manifold.TSNE`` runs the same method with
  approximate neighbours and a tuned optimizer, and exposes ``perplexity``
  and ``max_iter`` as the two knobs studied here. No lesson introduces
  scikit-learn, so ``POWER_UP_IN_LESSON`` is ``None``.
- Key APIs: ``sklearn.manifold.TSNE``.

Complexity Analysis
-------------------
Pain (manual), for n points of d dimensions and s optimizer steps:
- Time:  O(n^2·d) for the affinities, O(s·n^2) for the layout.
- Space: O(n^2) for the pairwise kernel.

Execution
---------
Project-style:
    uv run python src/052g_tsne_pitfalls.py

Script-style (self-contained deps):
    uv run --script src/052g_tsne_pitfalls.py

On a busy machine, NumPy's BLAS threads compete with other work; one thread
avoids that, since the matrices here are small:
    OPENBLAS_NUM_THREADS=1 uv run python src/052g_tsne_pitfalls.py

Options:
    python src/052g_tsne_pitfalls.py --doctest-only
    python src/052g_tsne_pitfalls.py --no-doctest
    python src/052g_tsne_pitfalls.py --index
    python src/052g_tsne_pitfalls.py -v

Quality Gates
-------------
    uv run pytest src/052g_tsne_pitfalls.py
    uv run ruff check src/052g_tsne_pitfalls.py
    uv run ruff format --check src/052g_tsne_pitfalls.py
    uv run mypy --strict src/052g_tsne_pitfalls.py

Module Doctests
---------------
Two points always have t-SNE affinity one half each, whatever their positions:

>>> import numpy as np
>>> joint_probabilities(np.array([[0.0, 0.0], [3.0, 4.0]]), 1.0).tolist()
[[0.0, 0.5], [0.5, 0.0]]

Two tight clusters 12 apart in 5-D, laid out by t-SNE, stay far apart:

>>> x, labels = blobs(axis_centres([0.0, 12.0], 5), [20, 20], [1.0, 1.0])
>>> bool(separation(tsne(x, perplexity=10.0, steps=250), labels) > 3.0)
True
"""

from __future__ import annotations

import argparse
import doctest
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence
    from types import ModuleType

    from numpy.typing import NDArray

    type Array = NDArray[np.float64]
    type Labels = NDArray[np.int64]
    type GradFn = Callable[[Array, Array], Array]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON IDENTITY & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON_ID: Final[str] = "tier_5_lesson_2g_tsne_pitfalls"

# The library form is scikit-learn's TSNE, which no lesson introduces yet.
POWER_UP_AVAILABLE: Final[bool] = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Format: lesson_id -> reason

PREREQUISITES: Final[Mapping[str, str]] = {
    "tier_5_lesson_2e_activation_atlas": "Exact t-SNE written by hand, the method under test",
}

DEPENDENCIES: Final[Mapping[str, str]] = {
    "tier_3_lesson_4_numpy": "Array operations behind every metric here",
}

SUCCESSORS: Final[Mapping[str, str]] = {}

DEPENDENTS: Final[Mapping[str, str]] = {}

RELATED: Final[Mapping[str, str]] = {
    "tier_5_lesson_2b_sparse_autoencoders": "Features you would plot with t-SNE",
    "tier_5_lesson_2f_contrastive_learning": "Embeddings whose t-SNE plots invite over-reading",
}

POWER_UP_IN_LESSON: Final[str | None] = None
PAIN_FROM_LESSON: Final[str | None] = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE METHOD UNDER TEST: Exact t-SNE (from Lesson 5.2e)
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
    neighbours. The conditionals are then averaged, ``(P + P.T) / 2n``.
    ``perplexity`` must be below the number of other points.

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
    ``w[i, j] = 1 / (1 + |y_i - y_j|^2)``, normalized to ``Q = w / sum(w)``. The
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
    Each coordinate of ``Y`` is nudged by +/-``eps`` and the change in the
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


def tsne_snapshots(
    x: Array,
    perplexity: float,
    marks: Collection[int],
    seed: int = 0,
    learning_rate: float = 100.0,
) -> dict[int, Array]:
    """
    Run t-SNE on the rows of ``x`` and return the layout after each step count in ``marks``.

    The layout starts as Gaussian noise of standard deviation 1e-4. For the
    first 100 steps the affinities ``P`` are multiplied by 12 (early
    exaggeration), which pulls clusters tight so they can move past each
    other. Momentum is 0.5 for the first 250 steps and 0.8 after. Every mark
    is a step count of at least 1, and the run stops at the largest.

    Examples
    --------
    One run yields every mark, each a copy, and a mark equals a shorter run:

    >>> import numpy as np
    >>> x = np.random.default_rng(0).normal(size=(20, 4))
    >>> snaps = tsne_snapshots(x, 4.0, [30, 60])
    >>> sorted(snaps), snaps[30].shape
    ([30, 60], (20, 2))
    >>> bool(np.array_equal(snaps[30], tsne_snapshots(x, 4.0, [30])[30]))
    True
    >>> bool(np.array_equal(snaps[30], snaps[60]))
    False
    """
    rng = np.random.default_rng(seed)
    P = joint_probabilities(x, perplexity)
    Y = rng.normal(0.0, 1e-4, (len(x), 2))
    velocity = np.zeros_like(Y)
    snaps: dict[int, Array] = {}
    for step in range(max(marks)):
        exaggeration = 12.0 if step < 100 else 1.0
        grad = tsne_kl_and_grad(exaggeration * P, Y)[1]
        velocity = (0.5 if step < 250 else 0.8) * velocity - learning_rate * grad
        Y += velocity
        Y -= Y.mean(axis=0)
        if step + 1 in marks:
            snaps[step + 1] = Y.copy()
    return snaps


def tsne(
    x: Array,
    perplexity: float = 30.0,
    steps: int = 750,
    seed: int = 0,
    learning_rate: float = 100.0,
) -> Array:
    """
    Return a 2-D layout of the rows of ``x`` after ``steps`` steps of gradient descent.

    See ``tsne_snapshots`` for the schedule. ``perplexity`` must be below
    ``len(x) - 1``.

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
    return tsne_snapshots(x, perplexity, [steps], seed, learning_rate)[steps]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA AND METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def axis_centres(positions: Sequence[float], dim: int) -> Array:
    """
    Return one centre per position, each on the first axis of ``dim``-D space.

    Examples
    --------
    >>> axis_centres([0.0, 10.0, 60.0], 2).tolist()
    [[0.0, 0.0], [10.0, 0.0], [60.0, 0.0]]
    """
    centres = np.zeros((len(positions), dim))
    centres[:, 0] = positions
    return centres


def blobs(
    centres: Array, sizes: Sequence[int], sigmas: Sequence[float], seed: int = 0
) -> tuple[Array, Labels]:
    """
    Return Gaussian clusters and their labels.

    Cluster ``k`` has ``sizes[k]`` points drawn around ``centres[k]`` with
    standard deviation ``sigmas[k]`` in every dimension. Labels are the
    cluster indices, in order. A single centre at the origin with sigma 1 is
    isotropic Gaussian noise.

    Examples
    --------
    >>> x, labels = blobs(axis_centres([0.0, 50.0], 3), [4, 2], [1.0, 1.0])
    >>> x.shape, labels.tolist()
    ((6, 3), [0, 0, 0, 0, 1, 1])
    >>> bool(x[labels == 1, 0].min() > 40.0)
    True
    >>> x2, _ = blobs(axis_centres([0.0, 50.0], 3), [4, 2], [1.0, 1.0])
    >>> bool(np.array_equal(x, x2))
    True
    """
    rng = np.random.default_rng(seed)
    parts = [
        centre + sigma * rng.normal(size=(size, len(centre)))
        for centre, size, sigma in zip(centres, sizes, sigmas, strict=True)
    ]
    labels = np.repeat(np.arange(len(sizes)), sizes)
    return np.vstack(parts), np.asarray(labels, dtype=np.int64)


def radius(points: Array) -> float:
    """
    Return the root-mean-square distance of ``points`` from their centroid.

    Examples
    --------
    Four points at distance 1 from their centroid:

    >>> radius(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]))
    1.0
    """
    centred = points - points.mean(axis=0)
    return float(np.sqrt((centred * centred).sum(axis=1).mean()))


def centre_gap(a: Array, b: Array) -> float:
    """
    Return the distance between the centroids of ``a`` and ``b``.

    Examples
    --------
    >>> centre_gap(np.array([[0.0, 0.0], [2.0, 0.0]]), np.array([[4.0, 4.0]]))
    5.0
    """
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def separation(y: Array, labels: Labels) -> float:
    """
    Return the smallest gap between two class centroids over the largest class radius.

    A score above about 3 means every pair of classes sits at least 3 radii
    apart: separate blobs. A score near or below 1 means some pair overlaps.
    Classes are the distinct values of ``labels``.

    Examples
    --------
    Two blobs 10 apart, each of radius about 1, score about 10:

    >>> rng = np.random.default_rng(0)
    >>> y = np.vstack([rng.normal(0.0, 0.7, (30, 2)), rng.normal(0.0, 0.7, (30, 2)) + [10, 0]])
    >>> labels = np.repeat([0, 1], 30)
    >>> round(separation(y, labels))
    10

    The metric can fail: shuffling the labels leaves two overlapping classes,
    and the score drops below 1:

    >>> shuffled = np.random.default_rng(1).permutation(labels)
    >>> bool(separation(y, shuffled) < 1.0)
    True

    With three classes the closest pair sets the score, so a third class
    placed 2 beyond the second overlaps it and pulls the score down:

    >>> y3 = np.vstack([y, rng.normal(0.0, 0.7, (30, 2)) + [12, 0]])
    >>> labels3 = np.repeat([0, 1, 2], 30)
    >>> bool(separation(y3, labels3) < 4.0)
    True
    """
    ids = np.unique(labels)
    widest = max(radius(y[labels == i]) for i in ids)
    gaps = [
        centre_gap(y[labels == a], y[labels == b]) for k, a in enumerate(ids) for b in ids[k + 1 :]
    ]
    return min(gaps) / widest


def clumpiness(points: Array) -> float:
    """
    Return the mean nearest-neighbour distance over the mean 10th-neighbour distance.

    Both means run over all points. Points spread evenly give a ratio near 1.
    Points in tight groups have a near neighbour but a far 10th neighbour,
    so the ratio falls. The score needs at least 11 points.

    Examples
    --------
    A lattice has the same few distances to its neighbours everywhere, so it
    scores higher than the same count of points drawn in tight pairs:

    >>> grid = np.array([[i, j] for i in range(5) for j in range(5)], dtype=float)
    >>> pairs = np.repeat(grid * 3.0, 2, axis=0) + np.tile([[0.0, 0.0], [0.1, 0.0]], (25, 1))
    >>> round(clumpiness(grid), 2), round(clumpiness(pairs), 2)
    (0.44, 0.02)
    """
    d = np.sqrt(squared_distances(points))
    d.sort(axis=1)
    return float(d[:, 1].mean() / d[:, 10].mean())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE FIVE PITFALLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def perplexity_separation(
    x: Array, labels: Labels, perplexities: Sequence[int], steps: int = 500, seed: int = 0
) -> dict[int, float]:
    """
    Return ``separation`` of the layout of ``x`` at each perplexity.

    The data do not change between entries; only the neighbourhood size does.

    Examples
    --------
    Two clusters of 30 points, 12 apart in 5-D. At perplexity 2 each cluster
    breaks into small groups and the classes barely separate; at 15 the two
    clusters are whole:

    >>> x, labels = blobs(axis_centres([0.0, 12.0], 5), [30, 30], [1.0, 1.0])
    >>> scores = perplexity_separation(x, labels, [2, 15], steps=300)
    >>> sorted(scores), bool(scores[2] < 4.0), bool(scores[15] > 3 * scores[2])
    ([2, 15], True, True)
    """
    return {p: separation(tsne(x, float(p), steps, seed), labels) for p in perplexities}


def spread_ratio(
    x: Array, labels: Labels, perplexity: float = 10.0, steps: int = 400, seed: int = 0
) -> tuple[float, float]:
    """
    Return the radius of class 1 over the radius of class 0, in ``x`` and in its layout.

    Radius is the root-mean-square distance from the class centroid. A layout
    that kept sizes would return two equal numbers.

    Examples
    --------
    Class 1 has ten times the standard deviation of class 0. In the data its
    radius is about 10 times larger; in the layout, within a factor of 2:

    >>> x, labels = blobs(axis_centres([0.0, 60.0], 5), [30, 30], [1.0, 10.0])
    >>> before, after = spread_ratio(x, labels)
    >>> bool(before > 7.0), bool(after < 2.0)
    (True, True)
    """
    y = tsne(x, perplexity, steps, seed)
    before = radius(x[labels == 1]) / radius(x[labels == 0])
    after = radius(y[labels == 1]) / radius(y[labels == 0])
    return before, after


def gap_ratio(
    x: Array, labels: Labels, perplexity: float = 10.0, steps: int = 400, seed: int = 0
) -> tuple[float, float]:
    """
    Return the far gap over the near gap between class centroids, in ``x`` and in its layout.

    The far gap runs from class 1 to class 2 and the near gap from class 0 to
    class 1. A layout that kept gaps would return two equal numbers.

    Examples
    --------
    Centres at 0, 10, and 60: the far gap is 5 times the near gap in the
    data, and within a factor of 2 of it in the layout:

    >>> x, labels = blobs(axis_centres([0.0, 10.0, 60.0], 5), [20, 20, 20], [1.0] * 3)
    >>> before, after = gap_ratio(x, labels)
    >>> bool(4.0 < before < 6.0), bool(after < 2.0)
    (True, True)
    """
    y = tsne(x, perplexity, steps, seed)

    def ratio(points: Array) -> float:
        return centre_gap(points[labels == 2], points[labels == 1]) / centre_gap(
            points[labels == 1], points[labels == 0]
        )

    return ratio(x), ratio(y)


def noise_clumpiness(
    dim: int, n: int = 60, perplexity: float = 2.0, steps: int = 300, seed: int = 0
) -> tuple[float, float]:
    """
    Return ``clumpiness`` of ``n`` Gaussian noise points in ``dim`` dimensions and of their layout.

    Noise has no clusters, yet at low perplexity t-SNE builds small groups
    from the chance neighbours each point has.

    Examples
    --------
    High-dimensional noise is spread evenly, so its score is near 1 and its
    layout's is lower:

    >>> before, after = noise_clumpiness(30)
    >>> bool(before > 0.7), bool(after < before - 0.2)
    (True, True)
    """
    x, _ = blobs(axis_centres([0.0], dim), [n], [1.0], seed)
    return clumpiness(x), clumpiness(tsne(x, perplexity, steps, seed))


def iteration_trace(
    x: Array,
    labels: Labels,
    perplexity: float,
    marks: Sequence[int],
    seed: int = 0,
) -> list[tuple[int, float, float]]:
    """
    Return ``(step, KL, separation)`` for the layout after each step count in ``marks``.

    KL is the divergence from the unexaggerated affinities, lower is a better fit.

    Examples
    --------
    Three clusters at 0, 10, and 60. After 10 steps the layout is still
    noise: the divergence is high and the classes overlap. After 300 steps it
    fits better and the classes are apart:

    >>> x, labels = blobs(axis_centres([0.0, 10.0, 60.0], 5), [20, 20, 20], [1.0] * 3)
    >>> trace = iteration_trace(x, labels, 10.0, [10, 300])
    >>> (s0, kl0, sep0), (s1, kl1, sep1) = trace
    >>> s0, s1, bool(kl1 < kl0), bool(sep0 < 1.0), bool(sep1 > 3.0)
    (10, 300, True, True, True)
    """
    P = joint_probabilities(x, perplexity)
    snaps = tsne_snapshots(x, perplexity, marks, seed)
    return [(m, tsne_kl_and_grad(P, snaps[m])[0], separation(snaps[m], labels)) for m in marks]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEEING IT: ASCII Scatter Plot (stdlib)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def scatter(y: Array, labels: Labels, width: int = 38, symbols: str = "ox+*") -> str:
    """
    Draw a 2-D layout on a character grid with one symbol per class.

    Both axes share one scale, and a character cell is taken to be twice as
    tall as wide, so a circle of points stays round and a cluster's size on
    screen is its size in the layout. The plot is at most ``width`` columns
    wide. A cell holds the symbol of the last point that falls in it, and an
    empty cell is ``.``. Class ``k`` uses ``symbols[k % len(symbols)]``.

    Examples
    --------
    Two points at opposite corners of a square, one per class:

    >>> pts = np.array([[0.0, 0.0], [1.0, 1.0]])
    >>> print(scatter(pts, np.array([0, 1]), width=6))
    ·····x
    ······
    o·····
    """
    lo = y.min(axis=0)
    span = y.max(axis=0) - lo
    scale = max(float(span[0]) / (width - 1), float(span[1]) / (2.0 * width), 1e-12)
    cols = ((y[:, 0] - lo[0]) / scale).astype(int)
    rows = ((y[:, 1] - lo[1]) / (2.0 * scale)).astype(int)
    grid = [["·"] * (int(cols.max()) + 1) for _ in range(int(rows.max()) + 1)]
    for r, c, k in zip(rows, cols, labels, strict=True):
        grid[int(r)][int(c)] = symbols[int(k) % len(symbols)]
    return "\n".join("".join(row) for row in reversed(grid))


def beside(left: str, right: str, gap: int = 4) -> str:
    r"""
    Join two multi-line blocks side by side, top aligned.

    Examples
    --------
    >>> print(beside("ab\ncd\nef", "X", gap=2))
    ab  X
    cd
    ef
    """
    a, b = left.splitlines(), right.splitlines()
    wide = max(len(line) for line in a)
    rows = max(len(a), len(b))
    a += [""] * (rows - len(a))
    b += [""] * (rows - len(b))
    return "\n".join(f"{p:<{wide}}{' ' * gap}{q}".rstrip() for p, q in zip(a, b, strict=True))


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
      LESSON INDEX: tier_5_lesson_2g_tsne_pitfalls
    ...
    Prerequisites (hard deps):
       → tier_5_lesson_2e_activation_atlas: Exact t-SNE written by hand, the method under test
    ...
       → Power-up in: (future lesson TBD)
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


def demo(n: int = 50, steps: int = 500, seeds: int = 3) -> None:
    """
    Print each pitfall's numbers and pictures on data of known shape.

    ``n`` is the number of points per cluster, ``steps`` the optimizer steps per
    layout, and ``seeds`` the number of data seeds whose median is reported
    for the ratios in the second and third pitfalls.

    Examples
    --------
    >>> demo(n=14, steps=60, seeds=1)  # doctest: +ELLIPSIS
    ═...
    1. PERPLEXITY CHANGES THE PICTURE
    ...
    5. STOPPING EARLY LEAVES PINCHED SHAPES
    ...
    """
    print("═" * 70)
    print(f"  {LESSON_ID.upper().replace('_', ' ')}")
    print("═" * 70)
    perps = [2, 5, max(6, n // 2), n - 2]
    marks = [10, 50, 100, 150, steps]

    _heading("1. PERPLEXITY CHANGES THE PICTURE")
    print(f"Two clusters of {n} points, 12 apart in 5-D, laid out at four perplexities.")
    print("separation = smallest centroid gap / largest cluster radius; 3+ is two blobs.")
    print()
    x, labels = blobs(axis_centres([0.0, 12.0], 5), [n, n], [1.0, 1.0])
    scores = perplexity_separation(x, labels, perps, steps)
    for p, score in scores.items():
        print(f"  perplexity {p:>3}: separation {score:6.2f}")
    print()
    panels = [f"perplexity {p}\n" + scatter(tsne(x, float(p), steps), labels) for p in perps[::2]]
    print(beside(panels[0], panels[1]))

    _heading("2. CLUSTER SIZES MEAN NOTHING")
    print("Two clusters, spread (standard deviation) 1 and 10, centres 60 apart.")
    print(f"Ratio of the diffuse cluster's radius to the tight one's, median of {seeds} seeds.")
    print()
    ins, outs = [], []
    for seed in range(seeds):
        x, labels = blobs(axis_centres([0.0, 60.0], 5), [n, n], [1.0, 10.0], seed)
        before, after = spread_ratio(x, labels, steps=steps, seed=seed)
        ins.append(before)
        outs.append(after)
    print(f"  in the data: {np.median(ins):5.2f}")
    print(f"  in the layout: {np.median(outs):5.2f}")
    print()
    print(scatter(tsne(x, 10.0, steps, seed=seeds - 1), labels))

    _heading("3. DISTANCES BETWEEN CLUSTERS MAY MEAN NOTHING")
    print("Three clusters with centres at 0, 10, and 60.")
    print(f"Ratio of the far gap to the near gap, median of {seeds} seeds.")
    print()
    ins, outs = [], []
    for seed in range(seeds):
        x, labels = blobs(axis_centres([0.0, 10.0, 60.0], 5), [n, n, n], [1.0] * 3, seed)
        before, after = gap_ratio(x, labels, steps=steps, seed=seed)
        ins.append(before)
        outs.append(after)
    print(f"  in the data: {np.median(ins):5.2f}")
    print(f"  in the layout: {np.median(outs):5.2f}")
    print()
    print(scatter(tsne(x, 10.0, steps, seed=seeds - 1), labels))

    _heading("4. NOISE DOES NOT ALWAYS LOOK RANDOM")
    count = 3 * n
    print(f"{count} points of isotropic Gaussian noise, perplexity 2.")
    print("clumpiness = mean nearest-neighbour distance / mean 10th-neighbour distance.")
    print()
    flat, _ = blobs(axis_centres([0.0], 2), [count], [1.0], seed=3)
    before, after = noise_clumpiness(50, n=count, steps=steps, seed=3)
    print(f"  true 2-D Gaussian: {clumpiness(flat):.3f}")
    print(f"  50-D noise, in the data: {before:.3f}")
    print(f"  50-D noise, in the layout: {after:.3f}")
    print()
    noise, _ = blobs(axis_centres([0.0], 50), [count], [1.0], seed=3)
    zeros = np.zeros(count, dtype=np.int64)
    print(
        beside(
            "true 2-D Gaussian\n" + scatter(flat, zeros),
            "50-D noise, laid out\n" + scatter(tsne(noise, 2.0, steps, seed=3), zeros),
        )
    )

    _heading("5. STOPPING EARLY LEAVES PINCHED SHAPES")
    print("Three clusters at 0, 10, 60; perplexity 10. KL is the divergence from P.")
    print()
    x, labels = blobs(axis_centres([0.0, 10.0, 60.0], 5), [n, n, n], [1.0] * 3, seed=1)
    for step, kl, sep in iteration_trace(x, labels, 10.0, marks):
        print(f"  step {step:>4}: KL {kl:6.3f}  separation {sep:6.2f}")
    snaps = tsne_snapshots(x, 10.0, [marks[1], marks[-1]], seed=0)
    print()
    print(
        beside(
            f"after {marks[1]} steps\n" + scatter(snaps[marks[1]], labels),
            f"after {marks[-1]} steps\n" + scatter(snaps[marks[-1]], labels),
        )
    )
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
