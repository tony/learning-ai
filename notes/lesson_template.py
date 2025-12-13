#!/usr/bin/env python
"""
[Lesson Number]. [Lesson Title].

Use this as the starting point for new lessons in `src/` (copy + edit).

Concepts:
- State the concept in 1–2 lines.
- List prerequisites (what earlier lessons or Python features are assumed).
- Keep the lesson focused: one concept per file.

Narrative:
- Explain why this concept matters in a growing AI system (models, data, eval, serving).

Complexity:
- Best case: O(?) with condition
- Average case: O(?)
- Worst case: O(?) with condition
- Space complexity: O(?)

Doctests:
- Include doctests demonstrating the core concept and edge cases.
- Keep output stable; if order/timing can vary, use ellipses:
  `# doctest: +ELLIPSIS`
- `uv run pytest` runs doctests automatically via `--doctest-modules`.

Type Hints & Mypy:
- Keep annotations compatible with `mypy --strict`.
- Prefer small, typed helpers over large untyped scripts.

Execution:
- Running `python path/to/lesson.py` should execute `main()` and print something useful.
"""

from __future__ import annotations

import timeit
from typing import TypeVar

T = TypeVar("T")


def main_concept_function(value: T) -> T:
    """
    Demonstrate the core concept of the lesson.

    Complexity:
    - Best: O(1)
    - Average: O(1)
    - Worst: O(1)
    - Space: O(1)

    Examples
    --------
    >>> main_concept_function(3)
    3
    >>> main_concept_function("ai")
    'ai'
    """
    return value


def main() -> None:
    """Run a small demonstration and optional timing."""
    example = "example"
    print(main_concept_function(example))

    timing = timeit.timeit(lambda: main_concept_function(example), number=10_000)
    print(f"Timing: {timing:.6f}s")  # Avoid timing output in doctests.


if __name__ == "__main__":
    import doctest

    # Run doctests when executed directly.
    doctest.testmod()
    main()
