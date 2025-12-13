#!/usr/bin/env python
"""
Lesson Template.

Use this as the starting point for new lessons in `src/`.

Concepts:
- Clearly state the main idea of the lesson
- List any prerequisites

Complexity:
- Best case: O(?) with condition
- Average case: O(?)
- Worst case: O(?) with condition
- Space complexity: O(?)

Narrative:
[Explain why this concept matters in a growing AI system]

Doctests:
[Briefly describe what the doctests demonstrate]
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
    print(f"Timing: {timing:.6f}s")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()

