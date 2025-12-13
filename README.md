# Learning AI

An educational, hands-on tutorial series for learning AI concepts in Python through small, runnable
lessons with doctests and type hints.

## Requirements

- Python 3.14+
- `uv` (recommended)

## Quick Start

```bash
uv sync --all-extras --dev
uv run pytest
```

## Create a Lesson

```bash
cp notes/lesson_template.py src/001_intro.py
python src/001_intro.py
uv run pytest src/001_intro.py
```

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run mypy .
uv run pytest-watcher
```

## Project Layout

- `src/`: lesson modules (numbered files)
- `notes/`: templates and supporting material

