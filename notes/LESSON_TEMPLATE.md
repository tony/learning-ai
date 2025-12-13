# Lesson Template

Use this template when creating a new lesson module in `src/`.

Guidelines:
- Keep lessons **self-contained** and runnable: `python path/to/lesson.py`
- Include **doctests** (run via `uv run pytest` with `--doctest-modules`)
- Keep type hints **mypy --strict** compatible
- Use **NumPy** docstring convention
- Include clear **time/space complexity** notes where applicable

If you want a runnable starting point, see `notes/lesson_template.py`.

## Skeleton

```python
#!/usr/bin/env python
"""
[Lesson Number]. [Lesson Title].

Concepts:
- Core concept explanation
- Best case: O(?) complexity with condition
- Average case: O(?) complexity
- Worst case: O(?) complexity with condition
- Space complexity: O(?)

Narrative:
[Explain why this concept matters in the growing system]

Doctests:
[Brief description of what the doctests demonstrate]
"""

import timeit
from typing import Any


def main_concept_function(value: Any) -> Any:
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
    timing = timeit.timeit(lambda: main_concept_function("example"), number=10_000)
    print(f"Timing: {timing:.6f}s")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
```

