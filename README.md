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

A lesson's number is its place in the
[lesson plan](notes/lesson_plan.md): `053` is Tier 5, Lesson 3. A letter
marks a side quest that branches off that lesson, so `052a` follows Lesson
5.2. Side quests are optional detours, most of them built on a Distill
article.

### Tier 3 — Vectors

| Lesson | What you build | Based on |
|--------|----------------|----------|
| [3.4a Convolution as a sliding dot product](src/034a_convolution.py) | Hand-made line detectors at four orientations, each lighting up for lines at its own angle | [Early Vision in InceptionV1](https://distill.pub/2020/circuits/early-vision/) |

### Tier 4 — Optimization

| Lesson | What you build | Based on |
|--------|----------------|----------|
| [4.2a Why momentum works](src/042a_momentum.py) | Gradient descent and heavy-ball momentum on a badly conditioned bowl; steps grow with κ for one and √κ for the other | [Why Momentum Really Works](https://distill.pub/2017/momentum/) |

### Tier 5 — Autodiff

| Lesson | What you build | Based on |
|--------|----------------|----------|
| [5.2a Toy models of superposition](src/052a_superposition.py) | An autoencoder that stores more sparse features than it has dimensions, as overlapping directions | [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) |
| [5.2b Sparse autoencoders](src/052b_sparse_autoencoders.py) | A dictionary whose directions line up with the features 5.2a packed together | [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) |
| [5.2c A tiny convolutional network](src/052c_tiny_cnn.py) | A one-layer CNN whose kernels become line detectors, though nothing in the loss mentions edges | [Early Vision in InceptionV1](https://distill.pub/2020/circuits/early-vision/) |
| [5.2d Reading a unit](src/052d_reading_a_unit.py) | One channel read three ways: top dataset examples, a tuning curve over gratings, and feature visualization | [Feature Visualization](https://distill.pub/2017/feature-visualization/), [Curve Detectors](https://distill.pub/2020/circuits/curve-detectors/) |
| [5.2e An activation atlas](src/052e_activation_atlas.py) | Activation vectors laid out by a hand-written t-SNE, averaged over a grid, and turned back into images | [Activation Atlas](https://distill.pub/2019/activation-atlas/) |
| [5.2f Contrastive learning](src/052f_contrastive_learning.py) | CLIP's mechanism at toy scale: an "image" encoder and a "text" encoder trained with InfoNCE | [Multimodal Neurons](https://distill.pub/2021/multimodal-neurons/), [CLIP](https://arxiv.org/abs/2103.00020) |
| [5.2g Reading t-SNE without being fooled](src/052g_tsne_pitfalls.py) | Five ways a t-SNE picture misleads, each measured on data whose true shape is known | [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) |
| [5.3 Earn PyTorch](src/053_earn_pytorch.py) | A hand-derived gradient checked against autograd, then one training run by hand and one with `torch.optim` | [PyTorch autograd](https://docs.pytorch.org/docs/stable/autograd.html) |

### Tier 6 — Tokenization

| Lesson | What you build | Based on |
|--------|----------------|----------|
| [6.2 Toy BPE](src/062_toy_bpe.py) | GPT-2's byte-level tokenizer in the standard library: train merges, encode, decode, and measure compression | [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909) |

### Tier 7 — Transformers

| Lesson | What you build | Based on |
|--------|----------------|----------|
| [7.3a How attention scores keys](src/073a_attention_scoring.py) | Why attention divides by √d_k, and additive scoring against dot-product scoring, trained by hand | [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473) |

### Tier 8 — A tiny GPT

| Lesson | What you build | Based on |
|--------|----------------|----------|
| [8.2a Grokking](src/082a_grokking.py) | A small MLP on modular addition that memorizes its training half, then generalizes thousands of steps later | [Power et al. (2022)](https://arxiv.org/abs/2201.02177) |

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

[Earn PyTorch](src/053_earn_pytorch.py) installs torch from PyTorch's CPU
index through its own header, because PyPI has no torch wheel for Python 3.15.
`pytest` skips it when torch is not installed; run it with `uv run --script`.

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
