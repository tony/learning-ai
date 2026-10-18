<div align="center">

# Learning AI

Machine learning from the ground up, one runnable Python file per lesson:
derive each method by hand, check it against a measurement, then earn the
library that does it for you.

[![tests](https://github.com/tony/learning-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/tony/learning-ai/actions/workflows/tests.yml)

</div>

Each lesson is one script in [`src/`](src/) that declares its own dependencies
in a [PEP 723](https://peps.python.org/pep-0723/) header, so `uv` runs it with
nothing installed first:

```console
$ uv run --script src/042a_momentum.py
```

A lesson's doctests are its tests. Every function carries a worked example,
and each lesson includes a deliberate break that shows its checks can fail.
Many lessons rebuild an article from [Distill](https://distill.pub/) or a
landmark paper, scaled down to run on a laptop CPU. Lessons print to the
terminal and write no files.

**Contents** — [Lessons](#lessons) · [Quick Start](#quick-start) ·
[Requirements](#requirements) · [Create a Lesson](#create-a-lesson) ·
[Project Layout](#project-layout)

## Lessons

A lesson's number is its place in the [lesson plan](notes/lesson_plan.md):
`053` is Tier 5, Lesson 3. A letter marks a side quest that branches off that
lesson, so `052a` follows Lesson 5.2. Side quests are optional detours, most of
them built on a Distill article.

### Tier 3: vectors

#### [`034a_convolution.py`](src/034a_convolution.py): a sliding dot product

Slide a small kernel across an image and take a dot product at every
position. Hand-made line detectors at four orientations each light up for
lines at their own angle, the first thing vision networks learn. From
[An Overview of Early Vision in InceptionV1](https://distill.pub/2020/circuits/early-vision/).

### Tier 4: optimization

#### [`042a_momentum.py`](src/042a_momentum.py): why momentum works

Run gradient descent and heavy-ball momentum on the same badly conditioned
bowl and count their steps. Gradient descent pays for the condition number κ,
momentum only for √κ: at κ = 10,000, tens of thousands of steps against about
a thousand. From [Why Momentum Really Works](https://distill.pub/2017/momentum/).

### Tier 5: autodiff

#### [`052a_superposition.py`](src/052a_superposition.py): more features than dimensions

Squeeze n sparse features through fewer hidden dimensions, with every gradient
derived by hand. Dense inputs keep only as many features as there are
dimensions; sparse inputs keep them all, as overlapping directions. From
[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html).

#### [`052b_sparse_autoencoders.py`](src/052b_sparse_autoencoders.py): unpacking the features

Train a wider autoencoder on the hidden layer of 5.2a and ask it to explain
each activation as a sparse sum of directions. Its dictionary finds the
features 5.2a packed together, where no single hidden dimension means anything.
From [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html).

#### [`052c_tiny_cnn.py`](src/052c_tiny_cnn.py): a network that learns line detectors

Train a one-layer convolutional network to tell lines at four angles apart.
Nothing in the loss mentions edges, yet its kernels end up as line detectors,
and their tuning curves show it. From
[An Overview of Early Vision in InceptionV1](https://distill.pub/2020/circuits/early-vision/).

#### [`052d_reading_a_unit.py`](src/052d_reading_a_unit.py): three ways to read one channel

Read a channel of the 5.2c network through its top dataset examples, its
tuning curve over gratings, and a feature visualization found by gradient
ascent. Where the three agree, the reading is evidence. From
[Feature Visualization](https://distill.pub/2017/feature-visualization/) and
[Curve Detectors](https://distill.pub/2020/circuits/curve-detectors/).

#### [`052e_activation_atlas.py`](src/052e_activation_atlas.py): a map of activation space

Collect activation vectors from many images, lay them out in 2-D with a t-SNE
written by hand, average them over a grid, and turn each cell back into an
image. From [Activation Atlas](https://distill.pub/2019/activation-atlas/).

#### [`052f_contrastive_learning.py`](src/052f_contrastive_learning.py): CLIP's mechanism at toy scale

Train an "image" encoder and a "text" encoder so matching pairs score high
under InfoNCE, with both gradients derived by hand. Most units come to fire for
the same concept in both views. From
[Multimodal Neurons](https://distill.pub/2021/multimodal-neurons/) and
[CLIP](https://arxiv.org/abs/2103.00020).

#### [`052g_tsne_pitfalls.py`](src/052g_tsne_pitfalls.py): five ways t-SNE misleads

Run t-SNE on data whose true shape you know. Perplexity joins or splits the
same clusters, a cluster ten times more spread out comes out the same size,
uneven gaps come out even, noise comes out clumpy, and a run stopped early
leaves pinched shapes. Each pitfall is one measured number. From
[How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/).

#### [`053_earn_pytorch.py`](src/053_earn_pytorch.py): autograd against a hand-derived gradient

Derive the gradient of logistic regression by hand, let PyTorch's autograd
compute it from the forward pass, and check that the two agree to rounding
error. Then train the model once by hand and once with `torch.optim`, and
watch the loss curves coincide.

### Tier 6: tokenization

#### [`062_toy_bpe.py`](src/062_toy_bpe.py): GPT-2's tokenizer in the standard library

Start from the 256 byte values, merge the most frequent adjacent pair into a
new token, and repeat. Encode by replaying the merges, decode by expanding
tokens back to bytes, and measure bytes per token on held-out text. From
[Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909).

### Tier 7: transformers

#### [`073a_attention_scoring.py`](src/073a_attention_scoring.py): how attention scores keys

Measure why attention divides by √d_k: without it the softmax collapses onto
one key as d_k grows. Then train additive and dot-product scoring by hand, and
see why additive scoring with no bias cannot learn. From
[Attention Is All You Need](https://arxiv.org/abs/1706.03762) and
[Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473).

### Tier 8: a tiny GPT

#### [`082a_grokking.py`](src/082a_grokking.py): memorizing, then generalizing

Train a small network on addition modulo a prime from half of all pairs.
Training accuracy reaches 99% within a few hundred steps; test accuracy stays
near chance for thousands more, then climbs to 99%. From
[Power et al. (2022)](https://arxiv.org/abs/2201.02177).

## Quick Start

Run a lesson's doctests, then its demo:

```console
$ uv run --script src/052g_tsne_pitfalls.py
```

Run only the doctests:

```console
$ uv run --script src/052g_tsne_pitfalls.py --doctest-only
```

Show where a lesson sits in the curriculum, with its prerequisites and the
lessons it unlocks:

```console
$ uv run --script src/052g_tsne_pitfalls.py --index
```

Install the dev dependencies:

```console
$ uv sync --all-extras --dev
```

Run every lesson's doctests through pytest:

```console
$ uv run pytest
```

[`053_earn_pytorch.py`](src/053_earn_pytorch.py) installs torch from
PyTorch's CPU index through its own header, because PyPI has no torch wheel for
Python 3.15. `pytest` skips it when torch is not installed; run it with
`uv run --script`.

## Requirements

- Python 3.14 or newer. CI runs Python 3.15.
- [`uv`](https://docs.astral.sh/uv/). All tooling here runs through it, and it
  installs each lesson's dependencies from the lesson's header.

## Create a Lesson

Copy the template into `src/` and give it the number of its place in the plan:

```console
$ cp notes/lesson_template.py src/001_intro.py
```

Run it directly to see its demo output:

```console
$ uv run python src/001_intro.py
```

[notes/LESSON_TEMPLATE.md](notes/LESSON_TEMPLATE.md) lists what to edit, and
[.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) lists the gates a lesson
must pass.

## Project Layout

- `src/`: the lessons, one numbered file each.
- `notes/`: the lesson template, the [lesson plan](notes/lesson_plan.md), and
  reference notes. No lesson code lives here.
- `.github/`: CI, and the [writing](.github/WRITING.md) and
  [contributing](.github/CONTRIBUTING.md) rules.

See [AGENTS.md](AGENTS.md) for the full project map.
