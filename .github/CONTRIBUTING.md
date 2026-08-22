# Contributing

Thanks for looking. This is a personal, evolving curriculum, so the most
useful thing right now is a correction: a bug in `notes/lesson_template.py`,
a doctest that does not actually run, or a place the lesson plan misleads.
Open an issue with a reproduction.

How this project writes prose — README, commit messages, docstrings, and
source comments — is set out separately in [WRITING.md](WRITING.md). Read
that before changing any of it. The constraints every change is held to, and
the map of what is where, are in [AGENTS.md](../AGENTS.md).

## Getting set up

```console
$ uv sync --all-extras --dev
```

This installs the dev dependency group — pytest, ruff, mypy — and provisions
a `.venv`. `requires-python` is `>=3.14,<4.0`; `uv` provisions that
interpreter for you if it is not already installed.

## The gates

Format:

```console
$ uv run ruff format .
```

Lint:

```console
$ uv run ruff check . --fix --show-fixes
```

Type-check:

```console
$ uv run mypy .
```

The `.` argument matters: `[tool.mypy]` sets `files = ["src/"]`, and until a
lesson exists there, a bare `uv run mypy` fails with "There are no .py[i]
files in directory 'src'". Passing `.` explicitly, as CI does, checks
`notes/` too and succeeds regardless of what `src/` holds.

Test:

```console
$ uv run pytest
```

Documentation is a gate, not a courtesy. Doctests under `src/` and `notes/`
are executed by this same `pytest` invocation — `--doctest-modules` is
already in `addopts`, so there is no separate doctest step. Which files are
reachable, and the one mistake that silently deletes a test, are in
[WRITING.md](WRITING.md#documented-examples-that-run).

Before claiming a test or a gate works, show it failing. A gate that has
never been red is an assumption.

CI (`.github/workflows/tests.yml`) runs these same four commands — `ruff
check`, `ruff format --check`, `mypy .`, and `pytest` (invoked there as `uv
run py.test`) — on Python 3.14. It is the order of record.

## Tests

Doctests are the only test the suite currently runs. `notes/lesson_template.py`
also defines `test_solve_pain` and similar `pytest`-style functions as a
worked example of the pattern, but numbered lesson files (`001_intro.py`)
never match pytest's default `python_files` pattern (`test_*.py`,
`*_test.py`), so those functions are never collected — only the file's
doctests are. Do not rely on a `test_` function in a lesson file actually
running; if a lesson needs assertions beyond what a doctest can express, name
the file to match the pattern, or route the check through a doctest instead.

There is no `conftest.py` and no fixture wired into the doctest namespace —
see [WRITING.md](WRITING.md#documented-examples-that-run) for what that means
for a block you write.

Run a single lesson's tests during development:

```console
$ uv run pytest src/001_intro.py
```

`pytest-watcher` re-runs the suite on file changes:

```console
$ uv run pytest-watcher
```

## Pull requests

One subject per pull request. Unrelated cleanup found along the way belongs
in its own commit, and usually in its own pull request.

Discuss a substantial change via an issue before making it.

Commit format is in [WRITING.md](WRITING.md#commits).

## Decorum

- Participants will be tolerant of opposing views.
- Participants must ensure that their language and actions are free of
  personal attacks and disparaging personal remarks.
- When interpreting the words and actions of others, participants should
  always assume good intentions.
- Behavior which can be reasonably considered harassment will not be
  tolerated.

Based on [Ruby's Community Conduct Guideline](https://www.ruby-lang.org/en/conduct/).
