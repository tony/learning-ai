# Lesson Template

Prefer starting from `notes/lesson_template.py` (it is runnable and has doctests). Use this file
as a quick checklist/reference.

Guidelines:
- Name lessons with a numeric prefix (e.g., `src/001_intro.py`).
- Keep lessons **self-contained** and runnable: `python src/001_intro.py`
- Include **doctests** (pytest runs them via `--doctest-modules`).
- Keep type hints compatible with **`mypy --strict`**.
- Use **NumPy** docstring convention for function docstrings.
- Include clear **time/space complexity** notes where applicable.
- If output can vary (timing, concurrency), use doctest ellipses:
  `# doctest: +ELLIPSIS`

## Start Here

```bash
cp notes/lesson_template.py src/001_intro.py
python src/001_intro.py
uv run pytest src/001_intro.py
```

## Skeleton

```python
#!/usr/bin/env python
"""
[Lesson Number]. [Lesson Title].

Concepts:
- Core concept explanation
- Prerequisites

Narrative:
[Explain why this concept matters in a growing AI system]

Complexity:
- Best: O(?) when condition
- Average: O(?)
- Worst: O(?) when condition
- Space: O(?)

Doctests:
[Brief description of what the doctests demonstrate]
"""

import timeit
from typing import TypeVar

T = TypeVar("T")


def main_concept_function(value: T) -> T:
    """
    Purpose description.

    Complexity:
    - Best: O(?) when condition
    - Average: O(?)
    - Worst: O(?) when condition
    - Space: O(?)

    Examples
    --------
    >>> main_concept_function("example")
    'example'
    """
    return value


def main() -> None:
    """Demonstrate the concept and optionally time it."""
    example = "example"
    print(main_concept_function(example))

    timing = timeit.timeit(lambda: main_concept_function(example), number=10_000)
    print(f"Timing: {timing:.6f}s")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
```
