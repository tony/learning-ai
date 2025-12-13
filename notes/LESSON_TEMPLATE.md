# Lesson Template

Prefer starting from `notes/lesson_template.py`. It is runnable, has doctests, and includes a
small CLI for indexing/benchmarking/logging.

Guidelines:
- Name lessons with a numeric prefix (e.g., `src/001_intro.py`).
- We require **`uv`**. Prefer running lessons via `uv run python src/001_intro.py`.
- Include **doctests** (pytest runs them via `--doctest-modules`) and keep output stable.
- Keep type hints compatible with **`mypy --strict`**.
- Use **NumPy** docstring convention for function docstrings.
- Include clear **time/space complexity** notes where applicable.
- If output can vary (timing, concurrency), use doctest ellipses:
  `# doctest: +ELLIPSIS`
- Prefer the "stdlib pain → library power-up" structure for each concept.

## Start Here

```bash
cp notes/lesson_template.py src/001_intro.py
uv run python src/001_intro.py --index
uv run python src/001_intro.py
uv run pytest src/001_intro.py
```

## What To Edit (In Every New Lesson)

- Module docstring placeholders: lesson title, objectives, narrative, complexity.
- `LESSON_ID` (used for display and artifact paths).
- Lesson index mappings: `PREREQUISITES`, `DEPENDENCIES`, `SUCCESSORS`, `DEPENDENTS`, `RELATED`.
- Power-up wiring:
  - Set `POWER_UP_AVAILABLE = True` only when `solve_power_up()` is implemented here.
  - If power-up comes later: keep `POWER_UP_AVAILABLE = False` and set `POWER_UP_IN_LESSON`.
  - If this lesson is the power-up: set `PAIN_FROM_LESSON` to the earlier pain lesson.
- Implement `solve_pain()` in every lesson; implement `solve_power_up()` only when earned.

## CLI Options (Built In)

```bash
python lesson.py --bench              # Run benchmark
python lesson.py --bench-number 2000  # Benchmark iterations
python lesson.py --doctest-only       # Only run doctests
python lesson.py --no-doctest         # Skip doctests
python lesson.py --index              # Show lesson index
python lesson.py --artifact-dir path  # Where to log experiments
python lesson.py --no-log             # Skip writing artifacts
python lesson.py -v                   # Verbose doctest output
```

By default, experiments log to `artifacts/<LESSON_ID>/experiments.jsonl`.

## UV Script Style (Supported)

If a single lesson needs third-party dependencies but you don’t want to add them to the whole
project, use `uv`'s **inline script metadata** (already present in the template):

```python
# /// script
# requires-python = ">=3.14,<4.0"
# dependencies = [
#   # === EARNED LIBRARIES ONLY ===
#   # Example: "rich",
# ]
# ///
```

Run it with:

```bash
uv run --script src/010_example.py
uv lock --script src/010_example.py
```

Executable shebang style is also supported:

```python
#!/usr/bin/env -S uv run --script
```

## Quality Gates

```bash
uv run pytest --doctest-modules lesson.py -v   # Gate 1: Correctness
uv run ruff check lesson.py                    # Gate 2: Lint
uv run ruff format --check lesson.py           # Gate 2: Format
uv run mypy --strict lesson.py                 # Gate 3: Types
```
