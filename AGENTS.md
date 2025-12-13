# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Essential Commands
```bash
# Install dependencies
uv sync --all-extras --dev

# Run tests
uv run pytest

# Run tests with watch mode (auto-rerun on file changes)
uv run pytest-watcher

# Run a single lesson file
uv run pytest src/001_intro.py

# Lint and format code
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy .

# Run doctests for a specific file
python -m doctest -v src/001_intro.py
```

## Code Architecture

This is an educational project for learning AI concepts, organized as numbered lessons with a consistent structure.

### Project Structure
- `src/`: Lesson modules (e.g., `src/001_intro.py`)
- `notes/`: Templates and supporting material
- Each lesson is self-contained and runnable with `python path/to/lesson.py`

### Lesson Template Structure

All lessons start from `notes/lesson_template.py` (see also `notes/LESSON_TEMPLATE.md`).

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
from typing import Any

def main_concept_function(params: type) -> ReturnType:
    """
    Purpose description.
    
    Complexity:
    - Best: O(?) when condition
    - Average: O(?)
    - Worst: O(?) when condition
    - Space: O(?)
    
    Examples
    --------
    >>> main_concept_function(example_input)
    expected_output
    """
    # Implementation
    pass

def main() -> None:
    """
    Main demonstration with performance measurements.
    Shows practical examples and timing comparisons.
    """
    # Performance testing with timeit
    # Result interpretation with narrative context
    pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
```

### Narrative Guidance

Anchor lessons in practical AI work:
- Data collection and preprocessing
- Training and evaluation
- Serving/inference and monitoring
- Reliability, performance, and correctness

### Key Development Practices
- All code must pass `mypy --strict`
- Use numpy docstring convention
- Include comprehensive doctests in all implementations
- Each module should be runnable standalone with meaningful output
- Always include complexity analysis (time and space)
- Connect concepts to the narrative scenarios

### Testing Strategy
- Doctests are the primary testing method (automatically run by pytest)
- Tests should demonstrate usage and edge cases
- Performance timing in main() functions helps understand complexity
- Use minimal sleeps and ellipses for concurrency tests

## Git Commit Standards

### Commit Message Format
```
Component/File(commit-type[Subcomponent/method]): Concise description

why: Explanation of necessity or impact.
what:
- Specific technical changes made
- Focused on a single topic
```

### Common Commit Types
- **feat**: New features or enhancements
- **fix**: Bug fixes
- **refactor**: Code restructuring without functional change
- **docs**: Documentation updates
- **chore**: Maintenance (dependencies, tooling, config)
- **test**: Test-related updates
- **style**: Code style and formatting

### Dependencies Commit Format
- Python packages: `py(deps): Package update`
- Python dev packages: `py(deps[dev]): Dev package update`

### Examples

#### Feature Addition
```
core/schema(feat[Query]): Add fruit filtering by color

why: Users need to filter fruits by color in GraphQL queries
what:
- Add color filter parameter to fruits query
- Update resolver to handle color filtering
- Add tests for color filtering
```

#### Bug Fix
```
core/types(fix[FruitType]): Correct optional color relationship

why: Color field was incorrectly marked as required
what:
- Change color field to use Optional type
- Update tests to handle None values
```

#### Dependencies Update
```
py(deps[dev]): Add django-stubs for type checking

why: Improve type safety for Django models and ORM
what:
- Add django-stubs to dev dependencies
- Configure MyPy to use django-stubs plugin
```

### Guidelines
- Subject line: Maximum 50 characters
- Body lines: Maximum 72 characters
- Use imperative mood ("Add", "Fix", not "Added", "Fixed")
- One topic per commit
- Separate subject from body with blank line
- Mark breaking changes: `BREAKING:`
