# AGENTS.md

Personal, hands-on curriculum for learning AI concepts in Python: numbered
lesson modules with doctests, type hints, and a runnable CLI, built one
concept at a time from stdlib up to earned libraries.

Follow the conventions already in the tree, and keep a change scoped to what
was asked for.

## What is here

| Path                        | What it is                                         |
| ---------------------------- | --------------------------------------------------- |
| `src/`                       | Lesson modules (e.g. `src/001_intro.py`); the curriculum itself. Empty except `.gitkeep` until the first lesson is copied in. |
| `notes/lesson_template.py`   | Authoritative runnable template every lesson copies from. |
| `notes/LESSON_TEMPLATE.md`   | What to edit in a new lesson, and its quality gates. |
| `notes/lesson_plan.md`       | Curriculum order, prerequisites, and per-lesson scope. |
| `notes/libs/`                | Reference notes on library usage patterns.         |
| `.github/workflows/tests.yml`| CI: ruff check, ruff format --check, mypy, pytest. |
| `pyproject.toml`             | Project metadata; ruff, mypy, and pytest config.   |

## Which policy applies

- Documentation, user-facing text, commit messages, docstrings, and source
  comments: [.github/WRITING.md](.github/WRITING.md)
- Environment, the gates, tests, and pull requests:
  [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md)

Each of those is the single home for its subject. Where a rule seems to be
stated twice, the file listed above is the one that governs.

## Change discipline

- Make the smallest coherent change that solves the verified problem; keep
  unrelated cleanup out of it.
- Reuse an existing file, helper, API, or test before adding a new one.
- Add a file only for a durable boundary — a distinct responsibility,
  independent reuse, or splitting an oversized module — not for a single-use
  helper or a one-line re-export.
- Add a doctest for every function and method; add a working example before
  claiming a lesson is done.
- A passing gate is evidence only once it has been shown capable of failing.
  Pair a new test with a deliberate break that proves it bites.

New lessons start from `notes/lesson_template.py`; lesson implementations
live only in `src/`. `notes/` is reference material — templates and
planning — never lesson code. `mypy --strict` and `requires-python
>=3.14,<4.0` apply to everything under `src/`.

## References

- `notes/lesson_plan.md` — curriculum order and prerequisites.
- [NumPy docstring convention](https://numpydoc.readthedocs.io/en/latest/format.html) —
  the dialect `ruff`'s `pydocstyle` rule enforces here.
