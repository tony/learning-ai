# Learning AI

An educational, hands-on tutorial series for learning AI concepts in Python
through small, runnable lessons with doctests and type hints.

## Requirements

- Python 3.14+
- `uv` — required, not optional; all Python tooling below runs through it

## Quick Start

Install the dev dependencies:

```console
$ uv sync --all-extras --dev
```

Run the doctest suite:

```console
$ uv run pytest
```

## Create a Lesson

Copy the template into `src/` and give it the next number:

```console
$ cp notes/lesson_template.py src/001_intro.py
```

Run it directly to see its demo output:

```console
$ uv run python src/001_intro.py
```

Run its doctests through pytest:

```console
$ uv run pytest src/001_intro.py
```

## Project Layout

- `src/`: lesson modules (numbered files), the curriculum itself.
- `notes/`: the lesson template, the lesson plan, and reference notes — not
  lesson implementations.

See [AGENTS.md](AGENTS.md) for the full project map and
[.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for the gates a change is
held to.
