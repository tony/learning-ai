#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<4.0"
# dependencies = ["numpy>=2.5.4"]
# ///
"""
Lesson 8.2a: Grokking on Modular Addition (side quest).

You train a small MLP on (a + b) mod p from half of all pairs (a, b), with
every gradient derived by hand, and watch two curves. Training accuracy reaches
99% within a few hundred steps, because the network memorizes its half of the
table. Test accuracy stays near chance for thousands more steps, then climbs to
99%: the network abandons the memorized table for a rule that works on pairs it
has never seen. That delayed jump is grokking.

Source: Power et al., "Grokking: Generalization Beyond Overfitting on Small
Algorithmic Datasets" (2022), https://arxiv.org/abs/2201.02177

Prerequisites (must complete first)
----------------------------------
- Lesson 8.2: Tiny GPT End-to-End (a full training loop with train/test evaluation)
- Lesson 5.2a: Toy Models of Superposition (hand-derived gradients checked by finite differences)

Learning Objectives
-------------------
After completing this lesson, you will be able to:
1. Derive the gradient of a one-hidden-layer MLP with a shared embedding by hand,
   and check it against central differences.
2. Write AdamW with decoupled weight decay in NumPy.
3. Measure the gap between the step where training accuracy and the step
   where test accuracy first reach 99%.

The Big Idea
------------
Fitting the training set and finding the rule are different events. Once the
training loss is near zero, the gradient no longer pushes the weights toward
the rule, but weight decay keeps shrinking them. The large memorizing solution
costs more weight than the compact rule, so the weights drift toward the rule
long after training accuracy stopped changing.

Why It Matters
--------------
Training accuracy tells you the model fits the data, not that it has
generalized, and a flat test curve is not proof that it never will. Nanda et
al. (2023), https://arxiv.org/abs/2301.05217 , reverse-engineered the trained
network as computing the sum with discrete Fourier transforms, a natural next
step once you can reproduce the curves.

Stdlib Pain → Library Power-Up
------------------------------
Pain (this lesson):
- You derive eight gradients by hand, including the shared embedding that
  receives one term from each input slot, then write AdamW's update yourself.
- Lines of code: ~25 for the gradient, ~15 for AdamW.
- Runtime: about four minutes for the demo run (4,800 full-batch steps, about
  50 ms each on one BLAS thread, on a loaded machine).
- Common bugs: dropping the second embedding term. Training still memorizes
  the table, so only the gradient check catches it.

Power-Up (Lesson 5.3 — PyTorch):
- ``nn.Embedding`` and ``nn.Linear`` hold the parameters, ``loss.backward()``
  produces every gradient, and ``torch.optim.AdamW`` replaces the hand-written
  update.
- Key APIs: ``nn.Embedding``, ``nn.Linear``, ``torch.optim.AdamW``, ``torch.autograd``.

Complexity Analysis
-------------------
Pain (manual), for N training pairs, p tokens, embedding size d, h hidden units,
k steps:
- Time:  O(k·N·(p·d + d·h + h·p)).
- Space: O(N·(p + h) + p·d + d·h + h·p).

Execution
---------
Project-style:
    uv run python src/082a_grokking.py

Script-style (self-contained deps):
    uv run --script src/082a_grokking.py

On a busy machine, NumPy's BLAS threads compete with other work and can stretch
the demo from about four minutes to many more; one thread avoids that, since the
matrices here are small:
    OPENBLAS_NUM_THREADS=1 uv run python src/082a_grokking.py

Options:
    python src/082a_grokking.py --doctest-only
    python src/082a_grokking.py --no-doctest
    python src/082a_grokking.py --index
    python src/082a_grokking.py -v

Quality Gates
-------------
    uv run pytest src/082a_grokking.py
    uv run ruff check src/082a_grokking.py
    uv run ruff format --check src/082a_grokking.py
    uv run mypy --strict src/082a_grokking.py

Module Doctests
---------------
With p = 10 and ``frac`` = 0.5, the 100 pairs split into 50 for training and
50 for testing:

>>> pairs, labels, train_idx, test_idx = make_dataset(10, 0.5, seed=0)
>>> pairs.shape, len(train_idx), len(test_idx)
((100, 2), 50, 50)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    type Array = NDArray[np.floating[Any]]
    type IntArray = NDArray[np.int64]
    type Params = dict[str, Array]
    type GradFn = Callable[[Params, Array, Array, IntArray], tuple[float, Params]]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON IDENTITY & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LESSON_ID: Final[str] = "tier_8_lesson_2a_grokking"

# Autograd, nn modules, and torch.optim.AdamW arrive with PyTorch.
POWER_UP_AVAILABLE: Final[bool] = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Format: lesson_id -> reason

PREREQUISITES: Final[Mapping[str, str]] = {
    "tier_8_lesson_2_tiny_gpt": "A full training loop with train/test evaluation",
    "tier_5_lesson_2a_superposition": "Hand-derived gradients checked by finite differences",
}

DEPENDENCIES: Final[Mapping[str, str]] = {
    "tier_2_lesson_4_cross_entropy": "Softmax cross-entropy and its gradient",
}

SUCCESSORS: Final[Mapping[str, str]] = {
    "tier_8_lesson_3_decoding": "Back to the main path: decoding the tiny GPT's outputs",
}

DEPENDENTS: Final[Mapping[str, str]] = {}

RELATED: Final[Mapping[str, str]] = {
    "tier_4_lesson_2a_momentum": "AdamW's first-moment average is momentum",
    "tier_5_lesson_2a_superposition": "Both read structure out of a small trained network",
}

POWER_UP_IN_LESSON: Final[str | None] = "tier_5_lesson_3_pytorch"
PAIN_FROM_LESSON: Final[str | None] = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE PAIN: An MLP on Modular Addition With Hand-Derived Gradients
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TARGET: Final[float] = 0.99


def make_dataset(
    p: int, frac: float, seed: int = 0
) -> tuple[IntArray, IntArray, IntArray, IntArray]:
    """
    Return every pair (a, b) with its label (a + b) mod p, and a train/test split of the rows.

    ``pairs`` has shape ``(p*p, 2)`` in row-major order, so row ``a*p + b``
    is the pair (a, b). A seeded permutation puts the first ``int(frac * p * p)``
    rows in ``train_idx`` and the rest in ``test_idx``.

    Examples
    --------
    >>> pairs, labels, train_idx, test_idx = make_dataset(5, 0.4)
    >>> pairs.shape, labels.shape
    ((25, 2), (25,))
    >>> len(train_idx), len(test_idx)
    (10, 15)
    >>> pairs[7].tolist(), int(labels[7])
    ([1, 2], 3)
    >>> pairs[24].tolist(), int(labels[24])
    ([4, 4], 3)

    The split covers every row exactly once:

    >>> sorted(train_idx.tolist() + test_idx.tolist()) == list(range(25))
    True
    """
    a, b = np.divmod(np.arange(p * p, dtype=np.int64), p)
    pairs = np.stack([a, b], axis=1)
    labels = (a + b) % p
    order = np.random.default_rng(seed).permutation(p * p).astype(np.int64)
    n_train = int(frac * p * p)
    return pairs, labels, order[:n_train], order[n_train:]


def one_hot(index: IntArray, size: int, dtype: Any = np.float32) -> Array:
    """
    Return the ``(len(index), size)`` matrix with a 1 at each row's index.

    ``one_hot(a, p) @ E`` equals ``E[a]``, and ``one_hot(a, p).T @ dX`` sums
    the rows of ``dX`` that belong to each token, which is the embedding
    gradient without ``np.add.at``.

    Examples
    --------
    >>> import numpy as np
    >>> one_hot(np.array([2, 0]), 3).tolist()
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    >>> E = np.arange(6.0).reshape(3, 2)
    >>> bool((one_hot(np.array([2, 0]), 3, np.float64) @ E == E[[2, 0]]).all())
    True
    """
    out = np.zeros((len(index), size), dtype=dtype)
    out[np.arange(len(index)), index] = 1
    return out


def init_params(p: int, d: int, h: int, seed: int = 0, dtype: Any = np.float32) -> Params:
    """
    Return the parameters ``E``, ``W1``, ``b1``, ``W2``, ``b2`` of the MLP.

    ``E`` is the ``(p, d)`` embedding shared by both inputs, ``W1`` is
    ``(2d, h)``, ``W2`` is ``(h, p)``, and the biases start at zero. ``E`` is
    N(0, 1), ``W1`` is N(0, 1)/√(2d), and ``W2`` is N(0, 1)/√h.

    Examples
    --------
    >>> params = init_params(7, 4, 5)
    >>> {k: v.shape for k, v in params.items()}
    {'E': (7, 4), 'W1': (8, 5), 'b1': (5,), 'W2': (5, 7), 'b2': (7,)}
    >>> params["E"].dtype, init_params(7, 4, 5, dtype=float)["E"].dtype
    (dtype('float32'), dtype('float64'))

    The same seed gives the same weights:

    >>> bool((init_params(7, 4, 5, seed=3)["W1"] == init_params(7, 4, 5, seed=3)["W1"]).all())
    True
    """
    rng = np.random.default_rng(seed)
    params = {
        "E": rng.normal(size=(p, d)),
        "W1": rng.normal(size=(2 * d, h)) / np.sqrt(2 * d),
        "b1": np.zeros(h),
        "W2": rng.normal(size=(h, p)) / np.sqrt(h),
        "b2": np.zeros(p),
    }
    return {k: v.astype(dtype) for k, v in params.items()}


def loss_and_grads(params: Params, OA: Array, OB: Array, y: IntArray) -> tuple[float, Params]:
    """
    Return the mean cross-entropy loss and its gradient for every parameter.

    ``OA`` and ``OB`` are the one-hot matrices of the two inputs, ``(N, p)``.
    The forward pass is X = [OA E, OB E], H = X W1 + b1, A = ReLU(H), and
    logits = A W2 + b2. Backward, with S the softmax of the logits::

        dL      = (S - onehot(y)) / N
        grad_W2 = Aᵀ dL                 grad_b2 = Σ dL
        dH      = (dL W2ᵀ) · (H > 0)
        grad_W1 = Xᵀ dH                 grad_b1 = Σ dH
        dX      = dH W1ᵀ
        grad_E  = OAᵀ dX[:, :d] + OBᵀ dX[:, d:]

    ``E`` feeds both input slots, so its gradient has two terms.

    Examples
    --------
    All-zero parameters give uniform probabilities, so the loss is ln 2 for
    p = 2. With both labels 0 and N = 2, ``grad_b2`` is the mean of
    (S - onehot) = (-1/2, 1/2), and every other gradient is zero because the
    hidden activations are zero:

    >>> import numpy as np
    >>> params = {k: np.zeros(s) for k, s in
    ...           {"E": (2, 1), "W1": (2, 1), "b1": (1,), "W2": (1, 2), "b2": (2,)}.items()}
    >>> OA, OB = one_hot(np.array([0, 1]), 2, np.float64), one_hot(np.array([1, 0]), 2, np.float64)
    >>> loss, grads = loss_and_grads(params, OA, OB, np.array([0, 0]))
    >>> round(loss, 6) == round(float(np.log(2)), 6)
    True
    >>> grads["b2"].tolist()
    [-0.5, 0.5]
    >>> max(float(np.abs(grads[k]).max()) for k in ("E", "W1", "b1", "W2"))
    0.0

    Near-zero weights give the loss of a uniform guess, ln p:

    >>> p = 13
    >>> params = {k: v * 0.01 for k, v in init_params(p, 8, 16).items()}
    >>> pairs, labels, _, _ = make_dataset(p, 0.5)
    >>> OA, OB = one_hot(pairs[:, 0], p), one_hot(pairs[:, 1], p)
    >>> loss, _ = loss_and_grads(params, OA, OB, labels)
    >>> abs(loss - float(np.log(p))) < 0.01
    True
    """
    E, W1, b1, W2, b2 = (params[k] for k in ("E", "W1", "b1", "W2", "b2"))
    n, d = len(y), E.shape[1]
    rows = np.arange(n)
    X = np.concatenate([OA @ E, OB @ E], axis=1)
    H = X @ W1 + b1
    A = np.maximum(H, 0)
    logits = A @ W2 + b2
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    total = exps.sum(axis=1, keepdims=True)
    loss = float(np.mean(np.log(total[:, 0]) - shifted[rows, y]))
    dL = exps / total
    dL[rows, y] -= 1
    dL /= n
    dH = (dL @ W2.T) * (H > 0)
    dX = dH @ W1.T
    grads: Params = {
        "E": OA.T @ dX[:, :d] + OB.T @ dX[:, d:],
        "W1": X.T @ dH,
        "b1": dH.sum(axis=0),
        "W2": A.T @ dL,
        "b2": dL.sum(axis=0),
    }
    return loss, grads


def accuracy(params: Params, OA: Array, OB: Array, y: IntArray) -> float:
    """
    Return the fraction of rows whose largest logit is the label.

    Examples
    --------
    With every weight zero except ``b2``, the logits are ``b2`` for every row,
    so the model always answers class 0 and is right on the rows labelled 0:

    >>> import numpy as np
    >>> params = {k: np.zeros(s) for k, s in
    ...           {"E": (2, 1), "W1": (2, 1), "b1": (1,), "W2": (1, 2), "b2": (2,)}.items()}
    >>> params["b2"] = np.array([1.0, 0.0])
    >>> a, b = np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])
    >>> OA, OB = one_hot(a, 2, np.float64), one_hot(b, 2, np.float64)
    >>> accuracy(params, OA, OB, np.array([0, 0, 1, 0]))
    0.75
    """
    E, W1, b1, W2, b2 = (params[k] for k in ("E", "W1", "b1", "W2", "b2"))
    H = np.concatenate([OA @ E, OB @ E], axis=1) @ W1 + b1
    logits = np.maximum(H, 0) @ W2 + b2
    return float(np.mean(np.argmax(logits, axis=1) == y))


def gradient_error(
    grad_fn: GradFn,
    params: Params,
    name: str,
    OA: Array,
    OB: Array,
    y: IntArray,
    eps: float = 1e-5,
) -> float:
    """
    Return the largest gap between ``grad_fn``'s gradient for one parameter and central differences.

    ``grad_fn`` has the signature of ``loss_and_grads``. Each entry of the
    chosen parameter is nudged by ±``eps`` and the loss from ``grad_fn`` is
    differenced. An exact gradient gives an error near ``eps``², a wrong one
    gives an error of the size of the missing term. Pass float64 parameters;
    float32 rounding swamps the difference.

    Examples
    --------
    The hand-derived gradient agrees with the numeric one for the shared
    embedding, the parameter with two terms:

    >>> import numpy as np
    >>> p = 5
    >>> params = init_params(p, 3, 7, dtype=np.float64)
    >>> pairs, labels, _, _ = make_dataset(p, 0.5)
    >>> OA, OB = one_hot(pairs[:, 0], p, np.float64), one_hot(pairs[:, 1], p, np.float64)
    >>> gradient_error(loss_and_grads, params, "E", OA, OB, labels) < 1e-6
    True

    The deliberate break: a gradient missing the b-slot term OBᵀ dX[:, d:] is far off.

    >>> def broken(params, OA, OB, y):
    ...     loss, grads = loss_and_grads(params, OA, OB, y)
    ...     E, W1, b1, W2, b2 = (params[k] for k in ("E", "W1", "b1", "W2", "b2"))
    ...     H = np.concatenate([OA @ E, OB @ E], axis=1) @ W1 + b1
    ...     S = np.exp(np.maximum(H, 0) @ W2 + b2)
    ...     S /= S.sum(axis=1, keepdims=True)
    ...     S[np.arange(len(y)), y] -= 1
    ...     dX = ((S / len(y)) @ W2.T * (H > 0)) @ W1.T
    ...     return loss, {**grads, "E": OA.T @ dX[:, :3]}
    >>> gradient_error(broken, params, "E", OA, OB, labels) > 1e-3
    True
    """
    _, analytic = grad_fn(params, OA, OB, y)
    numeric = np.zeros_like(params[name])
    for index in np.ndindex(*numeric.shape):
        up, down = params[name].copy(), params[name].copy()
        up[index] += eps
        down[index] -= eps
        loss_up = grad_fn({**params, name: up}, OA, OB, y)[0]
        loss_down = grad_fn({**params, name: down}, OA, OB, y)[0]
        numeric[index] = (loss_up - loss_down) / (2 * eps)
    return float(np.max(np.abs(analytic[name] - numeric)))


def adamw_step(
    params: Params,
    grads: Params,
    first: Params,
    second: Params,
    t: int,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-8,
    weight_decay: float = 1.0,
) -> None:
    """
    Update ``params`` and the moment estimates in place for step ``t`` (counting from 1).

    Adam's step is m̂ / (√v̂ + eps) with bias-corrected moments. Weight decay
    is decoupled: it shrinks each parameter directly instead of passing through
    the moments, so ``params -= lr * (adam_step + weight_decay * params)``.

    Examples
    --------
    With a zero gradient, only the decay acts: 1 - 0.1·1.0·1 = 0.9.

    >>> import numpy as np
    >>> params, grads = {"w": np.array([1.0])}, {"w": np.array([0.0])}
    >>> first, second = {"w": np.zeros(1)}, {"w": np.zeros(1)}
    >>> adamw_step(params, grads, first, second, 1, lr=0.1)
    >>> params["w"].round(6).tolist()
    [0.9]

    With gradient 1 and no decay, the bias-corrected first step is the full
    step ``lr``, whatever the gradient's scale:

    >>> params, grads = {"w": np.array([1.0])}, {"w": np.array([1.0])}
    >>> first, second = {"w": np.zeros(1)}, {"w": np.zeros(1)}
    >>> adamw_step(params, grads, first, second, 1, lr=0.1, weight_decay=0.0)
    >>> params["w"].round(6).tolist(), first["w"].round(6).tolist(), second["w"].round(6).tolist()
    ([0.9], [0.1], [0.02])
    """
    beta1, beta2 = betas
    for name, param in params.items():
        g, m, v = grads[name], first[name], second[name]
        m *= beta1
        m += (1 - beta1) * g
        v *= beta2
        v += (1 - beta2) * g * g
        step = (m / (1 - beta1**t)) / (np.sqrt(v / (1 - beta2**t)) + eps)
        param -= lr * (step + weight_decay * param)


@dataclass(slots=True)
class History:
    """
    Accuracy curves from one run and the first step each reached the target.

    Attributes
    ----------
    steps : list[int]
        Steps at which accuracy was measured.
    train_acc, test_acc : list[float]
        Accuracy on the training and test splits at each of those steps.
    train_step, test_step : int | None
        First measured step with accuracy at least ``TARGET``, or ``None``.
    """

    steps: list[int] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    test_acc: list[float] = field(default_factory=list)
    train_step: int | None = None
    test_step: int | None = None


def train(
    p: int = 59,
    frac: float = 0.5,
    d: int = 64,
    h: int = 256,
    seed: int = 0,
    max_steps: int = 20_000,
    check_every: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    on_check: Callable[[int, float, float], None] | None = None,
) -> History:
    """
    Train full-batch with AdamW until test accuracy reaches 99% or ``max_steps``.

    Accuracy is measured every ``check_every`` steps, and ``on_check(step,
    train_acc, test_acc)`` is called after each measurement. The seed fixes
    the split and the initial weights, so a run repeats exactly.

    Examples
    --------
    At p = 11 the 60 training pairs are memorized within a few hundred steps:

    >>> history = train(p=11, d=16, h=64, max_steps=300)
    >>> history.train_step is not None and history.train_step <= 300
    True
    >>> history.train_acc[-1]
    1.0
    >>> history.steps[:3]
    [100, 200, 300]
    """
    pairs, labels, train_idx, test_idx = make_dataset(p, frac, seed)
    params = init_params(p, d, h, seed)
    first = {k: np.zeros_like(v) for k, v in params.items()}
    second = {k: np.zeros_like(v) for k, v in params.items()}
    split = {}
    for name, idx in (("train", train_idx), ("test", test_idx)):
        split[name] = (one_hot(pairs[idx, 0], p), one_hot(pairs[idx, 1], p), labels[idx])
    history = History()
    for t in range(1, max_steps + 1):
        _, grads = loss_and_grads(params, *split["train"])
        adamw_step(params, grads, first, second, t, lr=lr, weight_decay=weight_decay)
        if t % check_every:
            continue
        train_acc = accuracy(params, *split["train"])
        test_acc = accuracy(params, *split["test"])
        history.steps.append(t)
        history.train_acc.append(train_acc)
        history.test_acc.append(test_acc)
        if history.train_step is None and train_acc >= TARGET:
            history.train_step = t
        if history.test_step is None and test_acc >= TARGET:
            history.test_step = t
        if on_check is not None:
            on_check(t, train_acc, test_acc)
        if history.test_step is not None:
            break
    return history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEEING IT: ASCII Line Chart (stdlib)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def line_chart(
    steps: Sequence[int],
    train_acc: Sequence[float],
    test_acc: Sequence[float],
    height: int = 11,
    width: int = 60,
) -> str:
    """
    Draw two accuracy curves in [0, 1] on a grid, ``o`` for train and ``x`` for test.

    Each of at most ``width`` columns samples one point; a cell where both
    curves land shows ``*``. Rows run from 1.0 at the top to 0.0 at the bottom.

    Examples
    --------
    >>> print(line_chart([0, 100, 200], [0.0, 0.5, 1.0], [0.0, 0.0, 1.0], height=3))
    1.0 |  *
    0.5 | o
    0.0 |*x
        +---
         step 0 to 200   o train  x test  * both
    """
    n = len(steps)
    cols = min(width, n)
    picks = [round(j * (n - 1) / max(cols - 1, 1)) for j in range(cols)]
    grid = [[" "] * cols for _ in range(height)]
    for marker, series in (("o", train_acc), ("x", test_acc)):
        for col, i in enumerate(picks):
            row = round((1 - series[i]) * (height - 1))
            grid[row][col] = marker if grid[row][col] == " " else "*"
    lines = []
    for r, cells in enumerate(grid):
        level = 1 - r / (height - 1)
        lines.append(f"{level:.1f} |{''.join(cells)}".rstrip())
    lines.append("    +" + "-" * cols)
    lines.append(f"     step {steps[0]} to {steps[-1]}   o train  x test  * both")
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
      LESSON INDEX: tier_8_lesson_2a_grokking
    ...
    Prerequisites (hard deps):
       → tier_8_lesson_2_tiny_gpt: A full training loop with train/test evaluation
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
    p: int = 59,
    frac: float = 0.5,
    d: int = 64,
    h: int = 256,
    max_steps: int = 20_000,
    report_every: int = 500,
) -> None:
    """
    Train on modular addition, print progress, the 99% steps, and both accuracy curves.

    Examples
    --------
    >>> demo(p=11, d=16, h=64, max_steps=300, report_every=100)  # doctest: +ELLIPSIS
    ═...
    step    train    test
    ...
    train accuracy first reached 99% at step ...
    ...
    """
    print("═" * 70)
    print(f"  {LESSON_ID.upper().replace('_', ' ')}")
    print("═" * 70)
    print()
    print(f"{'step':<8}{'train':<9}{'test'}")

    def report(step: int, train_acc: float, test_acc: float) -> None:
        if step % report_every == 0:
            print(f"{step:<8}{train_acc:<9.3f}{test_acc:.3f}", flush=True)

    history = train(p, frac, d, h, max_steps=max_steps, on_check=report)
    print()
    for label, step in (("train", history.train_step), ("test", history.test_step)):
        if step is None:
            print(f"{label} accuracy did not reach 99% within {max_steps} steps")
        else:
            print(f"{label} accuracy first reached 99% at step {step}")
    print()
    print(line_chart(history.steps, history.train_acc, history.test_acc))


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
