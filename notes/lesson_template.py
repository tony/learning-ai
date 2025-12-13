#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<4.0"
# dependencies = [
#   # === EARNED LIBRARIES ONLY ===
#   # Add dependencies here AFTER completing the stdlib "pain" version.
#   # Example: "numpy>=1.26",  # Earned in Lesson 1.3
# ]
# ///
"""
Lesson [TIER].[NUMBER]: [TITLE].

Use this as the starting point for new lessons in `src/` (copy + edit).

This template standardizes:
- A "stdlib pain → library power-up" progression (with optional parity checks)
- A lightweight CLI for doctests, benchmarks, and a lesson index
- Minimal experiment logging (JSONL) to `artifacts/`

Prerequisites (must complete first)
----------------------------------
- Lesson X.Y: [Name] (for [concept])

Learning Objectives
-------------------
After completing this lesson, you will be able to:
1. [Objective 1]
2. [Objective 2]
3. [Objective 3]

The Big Idea
------------
[One sentence capturing the core insight.]

Why It Matters
--------------
[2-3 sentences connecting this concept to real AI systems: data, evaluation,
serving, agents, memory, monitoring.]

Stdlib Pain → Library Power-Up
------------------------------
Pain (this lesson):
- [What's slow, verbose, or error-prone]
- Lines of code: ~[N]
- Runtime: ~[X] ms for [N] elements
- Common bugs: [pitfalls]

Power-Up (Lesson X.Y — [library name]):
- [What the library provides]
- Lines of code: ~[M] (vs ~[N] manual)
- Runtime: ~[Y] ms for [N] elements
- Key APIs: [main functions]

Complexity Analysis
-------------------
Pain (manual):
- Time:  O([?]) best | O([?]) average | O([?]) worst
- Space: O([?])

Power-Up (when earned):
- Time:  O([?]) best | O([?]) average | O([?]) worst
- Space: O([?])

Execution
---------
Project-style:
    uv run python lesson.py

Script-style (self-contained deps):
    uv run --script lesson.py

Options:
    python lesson.py --bench
    python lesson.py --doctest-only
    python lesson.py --no-doctest
    python lesson.py --index
    python lesson.py --no-log
    python lesson.py -v

Quality Gates
-------------
    uv run pytest --doctest-modules lesson.py -v
    uv run ruff check lesson.py
    uv run ruff format --check lesson.py
    uv run mypy --strict lesson.py

Module Doctests
---------------
>>> solve_pain([1, 2, 3])
6
>>> solve_pain([])
0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import timeit
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON IDENTITY & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

LESSON_ID: Final[str] = "tier_0_lesson_0_template"
ARTIFACT_DIR: Final[Path] = REPO_ROOT / "artifacts" / LESSON_ID

# Set to True when solve_power_up() is implemented in THIS lesson.
# Set to False when the power-up comes in a FUTURE lesson.
POWER_UP_AVAILABLE: Final[bool] = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX (fill in as curriculum develops)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Format: lesson_id -> reason (keeps reasons queryable + concrete)

# Hard dependencies — must complete before this lesson
PREREQUISITES: Final[Mapping[str, str]] = {
    # "tier_0_lesson_1_bytes": "Understanding raw data representation",
}

# Soft dependencies — helpful background but not strictly required
DEPENDENCIES: Final[Mapping[str, str]] = {
    # "tier_0_lesson_2_big_o": "Complexity language for analysis",
}

# Lessons unlocked by completing this one (recommended next steps)
SUCCESSORS: Final[Mapping[str, str]] = {
    # "tier_1_lesson_3_numpy": "Enables vectorized operations",
}

# Later lessons that rely on this (even if not immediate successors)
DEPENDENTS: Final[Mapping[str, str]] = {
    # "tier_2_lesson_5_backprop": "Uses this pattern in gradient accumulation",
}

# Optional related lessons (tangents, alternatives, deeper dives)
RELATED: Final[Mapping[str, str]] = {
    # "tier_1_lesson_7_property_testing": "Stronger invariants than hand-picked cases",
}

# Bidirectional power-up linking:
# - If this is a PAIN lesson: set POWER_UP_IN_LESSON to the power-up lesson ID
# - If this is a POWER-UP lesson: set PAIN_FROM_LESSON to the pain lesson ID
POWER_UP_IN_LESSON: Final[str | None] = None  # e.g., "tier_1_lesson_3_numpy"
PAIN_FROM_LESSON: Final[str | None] = None  # e.g., "tier_0_lesson_2_manual_sum"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TYPE ALIASES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENTAL DISCIPLINE: REPRODUCIBILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Extend when you earn NumPy/PyTorch.

    Examples
    --------
    >>> set_seed(123)
    >>> random.random()  # doctest: +ELLIPSIS
    0.052...
    >>> set_seed(123)
    >>> random.random()  # doctest: +ELLIPSIS
    0.052...
    """
    random.seed(seed)
    # Extend when earned:
    # np.random.seed(seed)
    # torch.manual_seed(seed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENTAL DISCIPLINE: HASHING & LOGGING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def stable_hash(payload: JsonValue, length: int = 12) -> str:
    """
    Compute stable SHA-256 hash for a JSON-serializable payload.

    The JSON is canonicalized (sorted keys, no whitespace) so equivalent payloads
    hash identically regardless of insertion order.

    Examples
    --------
    >>> stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    True
    >>> len(stable_hash({"x": 1}))
    12
    """
    blob = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def log_experiment(
    config: JsonObject,
    results: JsonObject,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """
    Append an experiment record to a JSONL log with a config hash.

    Keep `config` and `results` JSON-serializable for reproducible hashing.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as d:
    ...     path = Path(d) / "experiments.jsonl"
    ...     entry = log_experiment({"seed": 42}, {"loss": 0.5}, path)
    ...     entry["config"]["seed"]
    42
    >>> "timestamp" in entry and "config_hash" in entry
    True
    """
    log_path = log_file or (ARTIFACT_DIR / "experiments.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "lesson_id": LESSON_ID,
        "config": config,
        "config_hash": stable_hash(config),
        "results": results,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return entry


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENTAL DISCIPLINE: BENCHMARKING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def benchmark[T](
    func: Callable[[], T],
    number: int = 1000,
    warmup: int = 10,
) -> dict[str, float]:
    """
    Benchmark `func` with warmup runs for more stable measurements.

    Examples
    --------
    >>> stats = benchmark(lambda: sum(range(100)), number=100, warmup=5)
    >>> stats["per_call_ms"] >= 0
    True
    >>> all(k in stats for k in ["total_s", "per_call_ms", "calls_per_s"])
    True
    """
    for _ in range(warmup):
        func()

    total = timeit.timeit(func, number=number)
    return {
        "total_s": total,
        "per_call_ms": (total / number) * 1000,
        "calls_per_s": number / total if total > 0 else float("inf"),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE PAIN: Manual/Stdlib Implementation (ALWAYS PRESENT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def solve_pain(data: Sequence[int]) -> int:
    """
    Solve using only stdlib — feel the pain, build intuition.

    This version is intentionally verbose/slow to motivate the power-up that
    will come in a later lesson (or below if `POWER_UP_AVAILABLE=True`).

    Complexity
    ----------
    Time
        O(n) — single pass.
    Space
        O(1) — constant.

    Examples
    --------
    >>> solve_pain([1, 2, 3])
    6
    >>> solve_pain([])
    0
    >>> solve_pain([10, -5, 5])
    10
    """
    result = 0
    for x in data:
        result += x
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE POWER-UP: Optimized/Library Implementation (CONDITIONAL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# If this lesson INTRODUCES the power-up:
# - set POWER_UP_AVAILABLE = True
# - implement the optimized version in solve_power_up()
#
# If the power-up comes in a FUTURE lesson:
# - keep POWER_UP_AVAILABLE = False
# - calling solve_power_up() will raise NotImplementedError


def solve_power_up(data: Sequence[int]) -> int:
    """
    Solve using the power-up — experience the relief.

    After earning [library] in Lesson X.Y, this replaces solve_pain.
    Edit the implementation below when `POWER_UP_AVAILABLE` is True.

    Complexity
    ----------
    Time
        O(n) — but faster constants (usually).
    Space
        O(1).

    Examples
    --------
    >>> # Only test when power-up is available in this lesson
    >>> solve_power_up([1, 2, 3]) if POWER_UP_AVAILABLE else 6
    6
    """
    if not POWER_UP_AVAILABLE:
        msg = f"Power-up not available in this lesson. See: {POWER_UP_IN_LESSON}"
        raise NotImplementedError(msg)

    # Replace with library call when earned, e.g.: return np.sum(data)
    return sum(data)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PARITY VERIFICATION (only when power-up is available)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def parity_cases() -> list[list[int]]:
    """
    Return canonical test cases for parity verification.

    Keep these small and deterministic. Add cases when you discover a bug.
    """
    return [[], [0], [1, 2, 3], [10, -10, 5], list(range(100))]


def check_parity(cases: Sequence[Sequence[int]]) -> None:
    """
    Assert pain and power-up versions produce identical results.

    If `POWER_UP_AVAILABLE` is False, this function is a no-op.

    Examples
    --------
    >>> check_parity([[1, 2, 3], []])  # No-op if power-up not available
    """
    if not POWER_UP_AVAILABLE:
        return

    for case in cases:
        pain_result = solve_pain(case)
        power_result = solve_power_up(case)
        if pain_result != power_result:
            msg = f"Parity failed: {pain_result} != {power_result} for {list(case)!r}"
            raise AssertionError(msg)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PYTEST TESTS (inline, discovered by pytest)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_solve_pain() -> None:
    """Test the pain implementation."""
    assert solve_pain([]) == 0
    assert solve_pain([42]) == 42
    assert solve_pain([1, 2, 3]) == 6
    assert solve_pain([-1, 1]) == 0


def test_parity_if_available() -> None:
    """Test parity only when power-up is in this lesson."""
    if POWER_UP_AVAILABLE:
        check_parity(parity_cases())


def test_stable_hash_order_independent() -> None:
    """Verify hashing is stable regardless of key order."""
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI ARGUMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True)
class Args:
    """Parsed command-line arguments."""

    seed: int
    doctest: bool
    doctest_only: bool
    verbose: bool
    bench: bool
    bench_number: int
    artifact_dir: Path
    show_index: bool
    no_log: bool


def parse_args(argv: list[str] | None = None) -> Args:
    """
    Parse CLI arguments for flexible execution.

    Examples
    --------
    >>> args = parse_args(["--doctest-only"])
    >>> args.doctest_only
    True
    >>> args = parse_args(["--bench", "--bench-number", "500"])
    >>> args.bench and args.bench_number == 500
    True
    """
    parser = argparse.ArgumentParser(
        description=f"Run {LESSON_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--doctest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run doctests before demo (default: True)",
    )
    parser.add_argument("--doctest-only", action="store_true", help="Run doctests and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose doctest output")
    parser.add_argument("--bench", action="store_true", help="Run micro-benchmark")
    parser.add_argument(
        "--bench-number",
        type=int,
        default=1000,
        help="Iterations for benchmark (default: 1000)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help=f"Where to write logs (default: {ARTIFACT_DIR})",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        dest="show_index",
        help="Show lesson index (dependencies, successors, related)",
    )
    parser.add_argument("--no-log", action="store_true", help="Skip writing artifacts/logs")
    ns = parser.parse_args(argv)
    return Args(
        seed=ns.seed,
        doctest=ns.doctest,
        doctest_only=ns.doctest_only,
        verbose=ns.verbose,
        bench=ns.bench,
        bench_number=ns.bench_number,
        artifact_dir=ns.artifact_dir,
        show_index=ns.show_index,
        no_log=ns.no_log,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LESSON INDEX DISPLAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _format_mapping(name: str, items: Mapping[str, str]) -> str:
    """Format a mapping of lesson_id -> reason for display."""
    if not items:
        return f"{name}: (none)"
    lines = "\n".join(f"   → {lesson_id}: {reason}" for lesson_id, reason in items.items())
    return f"{name}:\n{lines}"


def show_lesson_index() -> None:
    """Display the lesson's position in the curriculum."""
    print("═" * 70)
    print(f"  LESSON INDEX: {LESSON_ID}")
    print("═" * 70)
    print()

    print(_format_mapping("Prerequisites (hard deps)", PREREQUISITES))
    print()
    print(_format_mapping("Dependencies (soft deps)", DEPENDENCIES))
    print()
    print(_format_mapping("Successors (unlocks)", SUCCESSORS))
    print()
    print(_format_mapping("Dependents (rely on this)", DEPENDENTS))
    print()
    print(_format_mapping("Related (optional)", RELATED))
    print()

    print("Power-Up Timeline:")
    if PAIN_FROM_LESSON:
        print(f"   ← Pain from: {PAIN_FROM_LESSON}")
    if POWER_UP_AVAILABLE:
        print("   ✓ Power-up: Available in THIS lesson")
    elif POWER_UP_IN_LESSON:
        print(f"   → Power-up in: {POWER_UP_IN_LESSON}")
    else:
        print("   → Power-up in: (future lesson TBD)")
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEMONSTRATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def demo(args: Args) -> None:
    """Run a small, interactive demonstration of the lesson."""
    set_seed(args.seed)

    print("═" * 70)
    print(f"  {LESSON_ID.upper().replace('_', ' ')}")
    print("═" * 70)
    print()

    # 1. Pain implementation demo
    print("1. THE PAIN (Manual Implementation)")
    print("─" * 40)
    sample = list(range(10))
    result = solve_pain(sample)
    print(f"   solve_pain({sample}) = {result}")
    print()

    # 2. Parity check (only if power-up is available)
    if POWER_UP_AVAILABLE:
        print("2. PARITY CHECK: Pain == Power-Up")
        print("─" * 40)
        cases = parity_cases()
        check_parity(cases)
        pain_result = solve_pain(sample)
        power_result = solve_power_up(sample)
        print(f"   solve_pain({sample}) = {pain_result}")
        print(f"   solve_power_up({sample}) = {power_result}")
        print(f"   ✓ Verified on {len(cases)} test cases!")
        print()
    else:
        print("2. POWER-UP STATUS")
        print("─" * 40)
        if POWER_UP_IN_LESSON:
            print(f"   ⏳ Power-up earned in: {POWER_UP_IN_LESSON}")
        else:
            print("   ⏳ Power-up comes in a future lesson")
        print("   (Complete this lesson to unlock!)")
        print()

    # 3. Benchmark
    if args.bench:
        print("3. BENCHMARK")
        print("─" * 40)
        large = list(range(10_000))
        n = args.bench_number

        pain_stats = benchmark(lambda: solve_pain(large), number=n)
        print(f"   Data: {len(large):,} elements | Iterations: {n:,}")
        print(f"   Pain (manual): {pain_stats['per_call_ms']:.4f} ms/call")

        power_stats: dict[str, float] | None = None
        if POWER_UP_AVAILABLE:
            power_stats = benchmark(lambda: solve_power_up(large), number=n)
            print(f"   Power-Up:      {power_stats['per_call_ms']:.4f} ms/call")
            if power_stats["per_call_ms"] > 0:
                speedup = pain_stats["per_call_ms"] / power_stats["per_call_ms"]
                print(f"   Speedup: {speedup:.1f}x")
        print()

        # Log experiment (unless --no-log)
        if not args.no_log:
            results: JsonObject = {"pain_ms": pain_stats["per_call_ms"]}
            if POWER_UP_AVAILABLE and power_stats is not None:
                results["power_ms"] = power_stats["per_call_ms"]

            entry = log_experiment(
                config={"seed": args.seed, "data_size": len(large), "iterations": n},
                results=results,
                log_file=args.artifact_dir / "experiments.jsonl",
            )
            print(f"   Logged: {args.artifact_dir / 'experiments.jsonl'}")
            print(f"   Hash: {entry['config_hash']}")
            print()

    print("═" * 70)
    print("  LESSON COMPLETE")
    print("═" * 70)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main(argv: list[str] | None = None) -> int:
    """
    Parse args → (optional) doctests → demo.

    Returns
    -------
    int
        Exit code (0=success, 1=failure).
    """
    import doctest

    args = parse_args(argv)

    if args.show_index:
        show_lesson_index()
        return 0

    if args.doctest:
        results = doctest.testmod(verbose=args.verbose)
        if results.failed > 0:
            print(f"❌ DOCTESTS FAILED: {results.failed}/{results.attempted}")
            return 1
        print(f"✓ All {results.attempted} doctests passed\n")

    if args.doctest_only:
        return 0

    demo(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
