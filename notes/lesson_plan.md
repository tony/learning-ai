### LLM Mastery Lesson Plan: Gradual Journey from Basics to Advanced AI Systems

The plan assumes no math/AI knowledge, starts super simple (e.g., numbers as data with analogies), builds gradually with sub-lessons/side quests. Total time: 150-200 hours.

#### Tier 0: Toolchain Foundation
**Goal:** Every future lesson is executable, testable, linted, typed, and measurable.
**Total Time:** 12-20 hours.

- **Lesson 0.1: uv Scripts and Dependency Blocks**
  - **Objectives**: Run single .py with declared deps, no manual setup.
  - **Prerequisites**: None.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - uv basics: Faster pip.
    - Script deps: PEP 723 inline.
    - Shebang: `#!/usr/bin/env -S uv run`.
  - **The Big Idea**: Scripts run anywhere without manual installation.
  - **Analogy**: Self-unpacking suitcase – everything packed in file.
  - **stdlib Pain**: `import numpy` → ModuleNotFoundError → manual pip → pollution.
  - **Power-Up**: uv shebang auto-installs isolated deps on run.
  - **Resource**: [uv docs](https://github.com/astral-sh/uv).
  - **Demo Prompt**:
```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Lesson 0.1: uv Scripts
The Big Idea: Declare dependencies in the script itself.
Why It Matters: Anyone can run your code without setup.
Analogy: A self-unpacking suitcase.
>>> 1 + 1
2
"""
def demonstrate_import_pain() -> str:
    return """
    # THE PAIN (without uv):
    $ python script.py
    ModuleNotFoundError: No module named 'numpy'
    $ pip install numpy # Pollutes global environment
    $ python script.py # Finally works, but fragile
   
    # THE POWER-UP (with uv):
    $ uv run script.py # Just works, isolated, fast, reproducible
    """
def main() -> None:
    print("=== Lesson 0.1: The Pain of Manual Dependencies ===")
    print(demonstrate_import_pain())
   
    print("\n=== The Power-Up ===")
    print("Add this to any script:")
    print('#!/usr/bin/env -S uv run --script')
    print('# /// script')
    print('# dependencies = ["numpy"]')
    print('# ///')
    print("\nNow it runs anywhere with: uv run script.py")
if __name__ == "__main__":
    main()
```
  - **Practice Tips**: Create hello-world with numpy dep; run `uv run script.py` on fresh machine; compare startup vs pip.

- **Lesson 0.2: doctest — Tests as Documentation**
  - **Objectives**: Write tests in docstrings; combine examples with verification.
  - **Prerequisites**: 0.1.
  - **Estimated Time**: 1-2 hours.
  - **Key Subtopics**:
    - Doctests: `>>> function(arg)` with output.
    - Run: `python -m doctest script.py`.
    - Benefits: Tests ARE documentation.
  - **The Big Idea**: Tests live in docstrings as executable specs.
  - **Analogy**: Recipe with built-in taste tests – book checks itself.
  - **stdlib Pain**: Manual asserts scattered, no reports.
  - **Power-Up**: Doctests turn docstrings into verified specs.
  - **Resource**: Python docs on doctest.
  - **Demo Prompt**:
```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Lesson 0.2: doctest
The Big Idea: Tests in docstrings = verified documentation.
Analogy: A recipe with built-in taste tests.
>>> add(2, 3)
5
>>> add(-1, 1)
0
"""
def add(a: int, b: int) -> int:
    """Add two numbers.
   
    >>> add(10, 20)
    30
    >>> add(0, 0)
    0
    """
    return a + b
def main() -> None:
    import doctest
    results = doctest.testmod()
   
    print("=== Lesson 0.2: doctest Results ===")
    print(f"Tests run: {results.attempted}")
    print(f"Failures: {results.failed}")
   
    if results.failed == 0:
        print("\n✓ All doctests passed!")
        print("Run with: python -m doctest lesson.py -v")
if __name__ == "__main__":
    main()
```
  - **Practice Tips**: Add 5 doctests to function; break one, see error; run -v.

- **Lesson 0.3: pytest — Real Testing at Scale**
  - **Objectives**: Write test functions; combine pytest with doctests.
  - **Prerequisites**: 0.2.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Test functions: `def test_add(): assert add(2,3) == 5`.
    - Discovery: Finds test_*/ *_test.py.
    - Run both: `pytest --doctest-modules`.
  - **The Big Idea**: pytest makes test intent obvious/scales.
  - **Analogy**: Professional QC vs spot-checking.
  - **stdlib Pain**: Unittest verbose; parameterization painful.
  - **Power-Up**: Discovers tests, runs doctests, clear reports.
  - **Resource**: [pytest docs](https://docs.pytest.org).
  - **Demo Prompt**:
```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""
Lesson 0.3: pytest
>>> divide(10, 2)
5.0
"""
def divide(a: float, b: float) -> float:
    """Divide a by b.
   
    >>> divide(6, 3)
    2.0
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
# PYTEST TEST FUNCTIONS
def test_divide_positive():
    assert divide(10, 2) == 5.0
def test_divide_negative():
    assert divide(-10, 2) == -5.0
def test_divide_by_zero():
    import pytest
    with pytest.raises(ValueError):
        divide(10, 0)
def main() -> None:
    print("=== Lesson 0.3: pytest ===")
    print("Run tests with: uv run pytest lesson.py --doctest-modules -v")
    print("\nThis runs:")
    print(" 1. All test_* functions")
    print(" 2. All doctests in docstrings")
    print(" 3. Gives a clean report")
if __name__ == "__main__":
    main()
```
  - **Practice Tips**: Run `uv run pytest --doctest-modules -v`; add @pytest.mark.parametrize; seed fixture repeatable.

- **Lesson 0.4: ruff — Speed-of-Thought Linting**
  - **Objectives**: Catch bugs; enforce formatting.
  - **Prerequisites**: 0.1.
  - **Estimated Time**: 1-2 hours.
  - **Key Subtopics**:
    - ruff check: Lint/style.
    - ruff format: Auto-format.
    - pyproject.toml config.
  - **The Big Idea**: ruff catches/formats 100x faster.
  - **Analogy**: Spell-checker fixes grammar.
  - **stdlib Pain**: Manual policing, inconsistency.
  - **Power-Up**: Fixes style before bugs.
  - **Resource**: [ruff docs](https://github.com/charliermarsh/ruff).
  - **Demo Prompt**: (Greet/sum with hints; bad_function to fail ruff/mypy; main prints validation commands.)
  - **Practice Tips**: Introduce 5 lints; fix all; add "no print in code" rule.
- **Lesson 0.5: mypy — Structural Integrity**
  - **Objectives**: Catch type errors pre-runtime; typed APIs.
  - **Prerequisites**: 0.4.
  - **Estimated Time**: 1-2 hours.
  - **Key Subtopics**:
    - Hints: `def add(a: int, b: int) -> int`.
    - Run: `mypy lesson.py`.
    - Protocol for interfaces (e.g., Tokenizer).
  - **The Big Idea**: mypy turns runtime TypeErrors to compile-time.
  - **Analogy**: Engineer checking blueprints – pipes standardized threads.
  - **stdlib Pain**: Runtime surprises late.
  - **Power-Up**: Errors caught before run.
  - **Resource**: [mypy docs](https://github.com/python/mypy).
  - **Demo Prompt**: (Greet/sum with hints/Protocol; bad_function to fail mypy; main prints gates.)
  - **Practice Tips**: Type parsing function, catch mismatch; Protocol for Tokenizer (reuse later).
- **Lesson 0.6: Reproducibility + JSONL Logging**
  - **Objectives**: Seed randomness; log configs/results.
  - **Prerequisites**: 0.3.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Seeding: `random.seed()`, `np.random.seed()`.
    - JSONL structured logs.
    - Reconstruction from logs.
  - **The Big Idea**: Experiments need reproducibility/logging.
  - **Analogy**: Flight recorders for reconstruction.
  - **stdlib Pain**: Non-deterministic runs, lost results.
  - **Power-Up**: Seeded/logged experiments repeatable.
  - **Resource**: Python docs on random/json.
  - **Demo Prompt**: (Set_seed; log_experiment; main demos seeded random/log.)
  - **Practice Tips**: Run twice, identical logs; log parameter sweep.
- **Lesson 0.7: Benchmarking + Ablations**
  - **Objectives**: Measure performance; vary one variable.
  - **Prerequisites**: 0.6.
  - **Estimated Time**: 1-2 hours.
  - **Key Subtopics**:
    - `time.perf_counter()` timing.
    - Repeats stats.
    - Ablations: One param time.
  - **The Big Idea**: Benchmarking finds bottlenecks.
  - **Analogy**: Timing lap splits performance.
  - **stdlib Pain**: Manual timing unreliable.
  - **Power-Up**: Stable measurement patterns.
  - **Resource**: Python docs on time/numpy stats.
  - **Demo Prompt**: (Benchmark func with repeats/mean/std; main benchmarks slow_sum different n.)
  - **Practice Tips**: Benchmark loop vs vectorized increasing sizes; ablation hyperparam records results.
- **Lesson 0.8: Filesystem + CLI Basics**
  - **Objectives**: `pathlib` IO; `argparse` CLI.
  - **Prerequisites**: 0.1.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - `pathlib` ops.
    - `argparse` interfaces.
    - Folders (`artifacts/`).
  - **The Big Idea**: CLI tools repeatable vs manual.
  - **Analogy**: Power tools labeled buttons – structure prevents chaos.
  - **stdlib Pain**: Hardcoded paths, no args.
  - **Power-Up**: Flexible scripted runs.
  - **Resource**: Python docs on argparse/pathlib.
  - **Demo Prompt**: (CLI with --in/--out/--seed; save artifact JSON in main.)
  - **Practice Tips**: Build CLI --in/--out/--seed; every lesson saves artifact.

#### Tier 1 — Text as Data Without Neural Nets
**Goal:** Demystify next-token prediction using counting/sampling.
**Total Time:** 9-14 hours.

- **Lesson 1.1: Everything is Bytes and Unicode**
  - **Objectives**: Strings ↔ bytes ↔ ints; frequency tables.
  - **Prerequisites**: Tier 0.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - ASCII: Chars as ints.
    - Unicode normalization.
    - Frequency tables.
  - **The Big Idea**: Computers see numbers, not meaning.
  - **Analogy**: Mapping words to piano keys – meaning in patterns.
  - **stdlib Pain**: Unicode boundaries mishandled.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Python docs on strings/bytes.
  - **Demo Prompt**: (String_to_numbers, ascii_distance, char_frequency; main shows problem of meaning.)
  - **Practice Tips**: Compute frequency top 10; compare ASCII vs meaning distance, explain failure.
- **Lesson 1.2: Markov Chains (Character Next-Token)**
  - **Objectives**: Transition counts, sample next char, generate text.
  - **Prerequisites**: 1.1.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Transitions: Count follows what.
    - Sampling: Weighted random.
    - Generation: Chain predictions.
  - **The Big Idea**: Predict next counting previous.
  - **Analogy**: Predictive text phone – "I love" → "you".
  - **stdlib Pain**: Manual weighted sampling verbose.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Wikipedia Markov chains.
  - **Demo Prompt**: (MarkovChain class with fit/generate; main trains text, generates seeded.)
  - **Practice Tips**: Train two corpora, compare style; add seeding, doctest snapshot.
- **Lesson 1.3: N-grams (Word-Level Context) + Smoothing**
  - **Objectives**: Unigram/bigram/trigram with smoothing.
  - **Prerequisites**: 1.2.
  - **Estimated Time**: 3-5 hours.
  - **Key Subtopics**:
    - Unigrams/bigrams/trigrams.
    - Conditional P(next | context).
    - Smoothing unseen.
  - **The Big Idea**: More context = better predictions.
  - **Analogy**: Reading short memory buffer – more context better til overfits.
  - **stdlib Pain**: Unseen cause crashes.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: NLTK n-gram examples (reference only).
  - **Demo Prompt**: (NGramModel with fit/predict/probability; main compares n=2/3.)
  - **Practice Tips**: Compare "surprise" counting misses train/test; top-next-words tool.
- **Lesson 1.4: Sampling Policies (Greedy/Temperature/Top-k/Top-p)**
  - **Objectives**: Implement sampling discrete distributions.
  - **Prerequisites**: 1.3.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Greedy vs random.
    - Temperature scaling.
    - Truncation top-k/top-p.
  - **The Big Idea**: Policies control creativity/safety.
  - **Analogy**: Choosing strict rules vs improvisation – knobs control creativity.
  - **stdlib Pain**: Manual sorting/cumsum top-p verbose.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: HuggingFace generation code (reference).
  - **Demo Prompt**: (Sample_with_policy logits temperature/top_k/top_p; main compares outputs.)
  - **Practice Tips**: Generate 20 samples temperatures; log diversity (unique/repetition); ban list test.
- **Lesson 1.5: First Evaluation: "Surprise" by Counting Misses**
  - **Objectives**: Pre-cross-entropy: unseen brittleness.
  - **Prerequisites**: 1.4.
  - **Estimated Time**: 1-2 hours.
  - **Key Subtopics**:
    - Surprise as misses count.
    - Why smoothing needed.
  - **The Big Idea**: Measure model surprise – lower better.
  - **Analogy**: Yes/no questions guess next word.
  - **stdlib Pain**: Manual loop counting.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Perplexity intro articles.
  - **Demo Prompt**: (Surprise_score model/text; main compares train/test.)
  - **Practice Tips**: Compare surprise train vs test; add smoothing, measure improvement.

#### Tier 2 — Probability and "Surprise" (The Loss Appears)
**Goal**: Uncertainty computable; stable primitives.
**Total Time**: 8-12 hours.

- **Lesson 2.1: Randomness and Distributions via Simulation**
  - **Objectives**: Simulate processes, histograms; mean/variance.
  - **Prerequisites**: Tier 1.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Random processes/histograms dicts.
    - Mean/variance interpret.
    - Convergence more samples.
  - **The Big Idea**: Random follows patterns.
  - **Analogy**: Dice vs cards – different distributions.
  - **stdlib Pain**: Manual binning loops verbose.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Python random module docs.
  - **Demo Prompt**: (Histogram from random samples; sample_mean/variance; main prints uniform/normal stats.)
  - **Practice Tips**: Simulate dice sums expected shape; test mean converges.
- **Lesson 2.2: Log/Exp and Numeric Failure Modes**
  - **Objectives**: Underflow/overflow as bugs.
  - **Prerequisites**: 2.1.
  - **Estimated Time**: 1-2 hours.
  - **Key Subtopics**:
    - Log/exp basics.
    - Overflow/underflow examples.
  - **The Big Idea**: Numbers break silently math ops.
  - **Analogy**: Water pressure bursting pipes.
  - **stdlib Pain**: math.exp large overflows NaNs.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Floating point articles.
  - **Demo Prompt**: (Show math.exp overflow; simple NaN checks.)
  - **Practice Tips**: Trigger overflow loop; add guards.
- **Lesson 2.3: Softmax (Naive → Stable)**
  - **Objectives**: Naive/stable softmax; log-sum-exp.
  - **Prerequisites**: 2.2.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Naive softmax overflow.
    - Stable (subtract max).
    - Log-sum-exp.
  - **The Big Idea**: Turn numbers probabilities sum 1.
  - **Analogy**: Raw scores to pie chart slices sum 100%.
  - **stdlib Pain**: Naive exp overflows large, NaNs.
  - **Power-Up**: Stable primitives reusable.
  - **Resource**: Stable softmax explanations.
  - **Demo Prompt**: (Softmax_naive trigger overflow; softmax_stable works; main compares.)
  - **Practice Tips**: Case naive fails stable works; tests sum ~1.0.
- **Lesson 2.4: Cross-Entropy and Perplexity (Tiny Discrete Examples)**
  - **Objectives**: Compute entropy/cross-entropy/KL/perplexity.
  - **Prerequisites**: 2.3.
  - **Estimated Time**: 3-4 hours.
  - **Key Subtopics**:
    - Entropy uncertainty.
    - Cross-entropy grading confidence.
    - KL distribution distance.
    - Perplexity branching factor.
  - **The Big Idea**: Cross-entropy grades confidence – wrong confident hurts most.
  - **Analogy**: Entropy "unpredictability"; cross-entropy grading homework.
  - **stdlib Pain**: Manual sum/log loops verbose.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Shannon entropy papers.
  - **Demo Prompt**: (Cross_entropy/perplexity good/bad pred; main prints entropy certain/uncertain.)
  - **Practice Tips**: Perplexity uniform vs peaked; show KL asymmetry doctest.
- **Lesson 2.5: Earn NumPy – Batch Probability Ops + Benchmarks**
  - **Objectives**: Vectorize softmax/cross-entropy; benchmark vs lists.
  - **Prerequisites**: 2.4.
  - **Estimated Time**: 1-2 hours.
  - **Key Subtopics**:
    - Vectorize batches.
    - Parity tests lists.
    - Benchmarks.
  - **The Big Idea**: NumPy vectorizes massive speedups.
  - **Analogy**: Hand arithmetic to factory conveyor – work scale.
  - **stdlib Pain**: Slow loops shape bugs.
  - **Power-Up**: Fast readable batch math.
  - **Resource**: NumPy docs.
  - **Demo Prompt**: (List loops softmax/cross; NumPy vectorized, benchmark speedup.)
  - **Practice Tips**: Plot speedup vs batch size (artifacts/); regression test parity tolerance.
#### Tier 3 — Vectors: Meaning as Geometry
**Goal:** Similarity search/matrix ops as AI substrate.
**Total Time**: 8-12 hours.
- **Lesson 3.1: Dot Product and Cosine Similarity (Lists)**
  - **Objectives**: Implement dot/norm/cosine from scratch.
  - **Prerequisites**: Tier 2.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Dot alignment measure.
    - Norm length.
    - Cosine interpretation.
  - **The Big Idea**: Vector list properties.
  - **Analogy**: Cosine comparing compass directions – alignment > length.
  - **stdlib Pain**: Loops verbose/slow.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Linear algebra intros.
  - **Demo Prompt**: (Dot_product/norm/cosine_similarity with meaning vectors.)
  - **Practice Tips**: Toy meaning dataset query nearest; tests orthogonal/identical/opposite.
- **Lesson 3.2: Nearest Neighbors Brute Force (Feel O(N))**
  - **Objectives**: Top-k similarity search full scan; runtime growth.
  - **Prerequisites**: 3.1.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Full scan top-k.
    - Runtime N.
    - Early-exit heuristics.
  - **The Big Idea**: Similar comparing everything.
  - **Analogy**: Finding friend crowd looking every person.
  - **stdlib Pain**: O(N) slow thousands vectors.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: KNN intros.
  - **Demo Prompt**: (Topk_cosine with benchmark N=100/1000/10000.)
  - **Practice Tips**: Benchmark N=1k/10k/100k log timings; early-exit, when fails.
- **Lesson 3.3: Matrix Multiply with Nested Loops (Pain Tier)**
  - **Objectives**: Implement matmul shape checks; suffer scaling.
  - **Prerequisites**: 3.2.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Row-column multiplication.
    - Shape compatibility.
    - O(n³) scaling.
  - **The Big Idea**: Matrices transform vectors; multiply chains.
  - **Analogy**: Composing two machines – output one input next.
  - **stdlib Pain**: Triple loops slow, mis-index easy.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Matrix multiply intros.
  - **Demo Prompt**: (Matmul with benchmark size=50/100/150.)
  - **Practice Tips**: Tests catch shape mismatch/off-by-one; benchmark 50x50 vs 100x100.
- **Lesson 3.4: NumPy Upgrade – Vectorization + Matmul**
  - **Objectives**: Replace loop matmul with `@`; match results.
  - **Prerequisites**: 3.3.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Vectorization.
    - Broadcasting.
    - Parity tests.
  - **The Big Idea**: NumPy optimizes vector/matrix C.
  - **Analogy**: Hand-copying to printing press.
  - **stdlib Pain**: O(n³) unfeasible large n.
  - **Power-Up**: Same math 100x faster, 1 line vs 15.
  - **Resource**: NumPy docs.
  - **Demo Prompt**: (Matmul_manual vs A_np @ B_np, benchmark speedup.)
  - **Practice Tips**: Implement both, assert difference < tolerance; measure speedup JSONL.
- **Lesson 3.5: Mini-Capstone: Toy Vector Search Engine**
  - **Objectives**: Index 1k-100k vectors, query top-k, log latency.
  - **Prerequisites**: 3.4.
  - **Estimated Time**: 1-2 hours.
  - **Key Subtopics**:
    - Build index.
    - Query top-k.
    - Latency logging.
  - **The Big Idea**: Combine vectors/similarity for search.
  - **Analogy**: Library card catalog – find similar books fast.
  - **stdlib Pain**: Full scan slow large N.
  - **Power-Up**: NumPy speeds core ops.
  - **Resource**: Vector search intros.
  - **Demo Prompt**: (Toy_search_engine class with add/query; main benchmarks increasing N.)
  - **Practice Tips**: Random vectors different dims; log latency vs N plot (artifacts/).

#### Tier 4 — Learning: Optimization Before Neural Nets
**Goal**: "Training" as minimizing score pre-backprop.
**Total Time**: 10-14 hours.

- **Lesson 4.1: Finite-Difference Gradients (stdlib)**
  - **Objectives**: Numerical gradients (forward/central); epsilon tradeoffs.
  - **Prerequisites**: Tier 3.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Forward/central difference.
    - Epsilon (accuracy vs noise).
    - Verify simple functions.
  - **The Big Idea**: Gradient tells downhill direction.
  - **Analogy**: Two tiny steps hillside – height change tells downhill.
  - **stdlib Pain**: Manual function calls per param.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Numerical differentiation intros.
  - **Demo Prompt**: (Grad_fd with method=forward/central, epsilon sweep log error.)
  - **Practice Tips**: Gradient-check x^2/sin(x)/log(1+x^2); sweep epsilon, log curves.
- **Lesson 4.2: Gradient Descent + Learning Rate Failure Modes**
  - **Objectives**: Implement GD; observe divergence/oscillation/slow.
  - **Prerequisites**: 4.1.
  - **Estimated Time**: 3-4 hours.
  - **Key Subtopics**:
    - GD algorithm.
    - LR: High diverge, low slow.
    - Logging/stopping.
  - **The Big Idea**: Step opposite gradient til minimum.
  - **Analogy**: Walking downhill fog – step size decides convergence/chaos.
  - **stdlib Pain**: Manual loop management.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: GD intros.
  - **Demo Prompt**: (Gd_minimize with trajectory, LR comparison table.)
  - **Practice Tips**: LR sweep log final loss/steps; add momentum, compare; early stopping no improvement N steps.
- **Lesson 4.3: Linear Regression by Hand (MSE)**
  - **Objectives**: Train y = w*x + b GD; overfitting detection.
  - **Prerequisites**: 4.2.
  - **Estimated Time**: 3-4 hours.
  - **Key Subtopics**:
    - Linear formulation.
    - MSE loss.
    - Train/test split.
  - **The Big Idea**: Find line minimizing squared errors.
  - **Analogy**: Sliding/tilting ruler points minimize miss.
  - **stdlib Pain**: Manual param updates.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Linear regression intros.
  - **Demo Prompt**: (Fit_linear_regression with MSE, train/test losses.)
  - **Practice Tips**: Synthetic data recover w,b noise; add L2, observe shrinkage.
- **Lesson 4.4: Earn Matplotlib – Plot Loss Curves + Fitted Line**
  - **Objectives**: Visualize loss/predictions; save artifacts.
  - **Prerequisites**: 4.3.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Loss vs steps plots.
    - Predictions vs truth.
    - Deterministic saves.
  - **The Big Idea**: Visualization interpretable training.
  - **Analogy**: Loss curves EKG training – waveform tells stability/stuck/crash.
  - **stdlib Pain**: ASCII/CSV debug output pain.
  - **Power-Up**: matplotlib makes visualization trivial.
  - **Resource**: [matplotlib docs](https://matplotlib.org/stable/).
  - **Demo Prompt**: (Plot loss curves/fitted line; save artifacts/.)
  - **Practice Tips**: Plot multiple LR one chart; smoke test loss decreases fixed seed.
- **Side Quest 4.5: Classic ML Interlude (Earn scikit-learn)**
  - **Objectives**: Build k-NN classifier manual → scikit-learn pipeline ergonomics.
  - **Prerequisites**: 4.4 (optional).
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Feature engineering.
    - Cross-validation.
  - **The Big Idea**: Classic ML baselines for comparison.
  - **Analogy**: k-NN like finding similar friends for advice.
  - **stdlib Pain**: Manual distance/sorting verbose.
  - **Power-Up**: scikit-learn classifiers/regressors easy.
  - **Resource**: [scikit-learn docs](https://scikit-learn.org/stable/).
  - **Demo Prompt**: (Manual k-NN; then sklearn.KNeighborsClassifier.)
  - **Practice Tips**: Iris dataset classify; cross-val score.
#### Tier 5 — Autodiff: Stop Doing Backprop by Hand
**Goal**: Build backprop once, verify, upgrade PyTorch autograd.
**Total Time**: 12-18 hours.
- **Lesson 5.1: Manual Backprop on Tiny Computation Graph (stdlib)**
  - **Objectives**: Compute gradients hand small graph; verify FD.
  - **Prerequisites**: Tier 4.
  - **Estimated Time**: 3-4 hours.
  - **Key Subtopics**:
    - Chain rule code.
    - Local derivatives multiply paths.
    - Verification FD.
  - **The Big Idea**: Gradients flow backward.
  - **Analogy**: Tracing leak backward pipes – joint contributes spill.
  - **stdlib Pain**: Manual per graph.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Backprop intros.
  - **Demo Prompt**: (Manual_backprop (x + y) * y; FD verify.)
  - **Practice Tips**: 3 graphs (add/mul/tanh/ReLU), FD-check; gradcheck_scalar helper reuse.
- **Lesson 5.2: Build Scalar Autodiff Engine (micrograd-style)**
  - **Objectives**: `Value(data, grad)` op history/topo-backward.
  - **Prerequisites**: 5.1.
  - **Estimated Time**: 6-9 hours.
  - **Key Subtopics**:
    - Value class data/grad/op history.
    - Backward topo sorting.
    - Ops: add/mul/pow/tanh/relu.
  - **The Big Idea**: Autodiff records ops/computes gradients auto.
  - **Analogy**: Sensors pipe junctions – records changes propagate.
  - **stdlib Pain**: Manual grads every change.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: [micrograd](https://github.com/karpathy/micrograd) (~100 lines).
  - **Demo Prompt**: (Value with __add__/__mul__/tanh/backward; main computes df/da/db.)
  - **Practice Tips**: Implement add/mul/tanh, gradcheck FD; train tiny linear loss decreases; graph pretty printer debug topo.
- **Lesson 5.3: Earn PyTorch – Tensor Autograd + Training Loop Skeleton**
  - **Objectives**: Re-implement training PyTorch; reusable loop (seed/log/eval/checkpoint).
  - **Prerequisites**: 5.2.
  - **Estimated Time**: 3-5 hours.
  - **Key Subtopics**:
    - `torch.tensor` requires_grad=True.
    - .backward() .grad.
    - Training skeleton.
  - **The Big Idea**: PyTorch extends scalar autodiff tensors/GPUs.
  - **Analogy**: Upgrading hand tools machine shop – concepts scale big tensors auto gradients.
  - **stdlib Pain**: Hand-derived grads not scale toys.
  - **Power-Up**: Auto gradients tensors + optimizers.
  - **Resource**: [PyTorch docs](https://pytorch.org/docs/stable).
  - **Demo Prompt**: (Simple gradient; same example 5.2; linear regression loop skeleton with epochs/loss print.)
  - **Practice Tips**: Compare PyTorch gradients scalar engine matched case; implement logistic regression PyTorch learns synthetic; add device switch keep CPU default.
- **Lesson 5.4: Shape Discipline Micro-Lessons**
  - **Objectives**: Explicit invariants like (B,T,D); assertion helpers/typed shapes.
  - **Prerequisites**: 5.3.
  - **Estimated Time**: 1-2 hours.
  - **Key Subtopics**:
    - Shape invariants.
    - Assertion helpers.
    - Typed tensor shapes (best-effort).
  - **The Big Idea**: Shape discipline prevents bugs.
  - **Analogy**: Lego blocks – wrong shape won't fit.
  - **stdlib Pain**: Runtime shape mismatches late.
  - **Power-Up**: N/A (pure stdlib/PyTorch).
  - **Resource**: PyTorch tensor docs.
  - **Demo Prompt**: (Shape_check func; Protocol for tensor interfaces.)
  - **Practice Tips**: Add shape asserts toy model; test mismatch catches.
#### Tier 6 — Tokenization: Compression You Can See
**Goal**: Text → IDs matching real LLM pipelines.
**Total Time**: 9-14 hours.
- **Lesson 6.1: Naive Tokenization Breaks Reality**
  - **Objectives**: Regex tokenizer/detokenizer; failure modes.
  - **Prerequisites**: Tier 5.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Regex-based.
    - Failures: Contractions/punctuation/Unicode.
    - encode/decode interface.
  - **The Big Idea**: Simple tokenization fails real text.
  - **Analogy**: Chopping text scissors every space – mangles punctuation/Unicode/edges.
  - **stdlib Pain**: Regex fails contractions/punctuation/Unicode.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Tokenization problem articles.
  - **Demo Prompt**: (Regex tokenizer, torture corpus show failures.)
  - **Practice Tips**: Build torture-test corpus snapshot tokens; track round-trip accuracy.
- **Lesson 6.2: Implement Toy BPE (Byte-Pair Encoding)**
  - **Objectives**: Train merges, encode/decode.
  - **Prerequisites**: 6.1.
  - **Estimated Time**: 4-6 hours.
  - **Key Subtopics**:
    - BPE: Count pairs, merge frequent.
    - Encode/decode.
    - Compression measurement.
  - **The Big Idea**: BPE learns vocabulary merging frequent pairs.
  - **Analogy**: Inventing abbreviations replacing common letter pair – compression frequency.
  - **stdlib Pain**: Manual merge loops slow large corpora.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: BPE algorithm papers.
  - **Demo Prompt**: (BPETokenizer with train/encode; main trains text, prints compression.)
  - **Practice Tips**: Train two corpora, compare merges; add reserved <unk>/<pad>, test decode; benchmark encode speed merges increase, log JSONL.
- **Lesson 6.3: Earn tiktoken – Production Tokenization**
  - **Objectives**: Use production tokenizer; compare toy BPE.
  - **Prerequisites**: 6.2.
  - **Estimated Time**: 3-5 hours.
  - **Key Subtopics**:
    - tiktoken or HF tokenizers.
    - Speed/count comparison.
    - Token budget estimator.
  - **The Big Idea**: Production tokenizers handle edges.
  - **Analogy**: Standardized shipping containers – packed compatible units model expects.
  - **stdlib Pain**: Unicode corners/performance traps.
  - **Power-Up**: Battle-tested aligned real LLMs.
  - **Resource**: [tiktoken](https://github.com/openai/tiktoken).
  - **Demo Prompt**: (Wrap tiktoken Tokenizer protocol, compare speed/counts vs toy.)
  - **Practice Tips**: Compare token counts code/prose/messy web; tests encode/decode round-trips; CLI tokenize.py --count file.txt prints counts/top tokens.
- **Lesson 6.4: Data Batching Basics (No Model Yet)**
  - **Objectives**: Collation/padding/masks; deterministic shuffling.
  - **Prerequisites**: 6.3.
  - **Estimated Time**: 1-2 hours.
  - **Key Subtopics**:
    - Collation.
    - Padding/masks.
    - Shuffling seeded.
  - **The Big Idea**: Prepare data batches training.
  - **Analogy**: Packing boxes same size shipping.
  - **stdlib Pain**: Manual padding loops error-prone.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Data loader intros.
  - **Demo Prompt**: (Batch_collate func with padding/mask; main shuffles dataset seeded.)
  - **Practice Tips**: Test batch shapes consistent; verify shuffling deterministic seed.
#### Tier 7 — Transformers: Attention as Compute Pattern
**Goal**: Implement attention from scratch tiny sizes, scale correctly.
**Total Time**: 18-28 hours.
- **Lesson 7.1: Causal LM Objective (Shifted Targets)**
  - **Objectives**: Build input/target shifting; cross-entropy loss.
  - **Prerequisites**: Tier 6.
  - **Estimated Time**: 3-4 hours.
  - **Key Subtopics**:
    - Shifting: input[:-1] predicts target[1:].
    - Cross-entropy logits.
    - Padding mask.
  - **The Big Idea**: Predict next from previous.
  - **Analogy**: Writing future taped over – guess next visible.
  - **stdlib Pain**: Manual shifting/masking loops error-prone.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Transformer papers.
  - **Demo Prompt**: (Make_shifted_targets, compute cross-entropy tiny logits stable softmax.)
  - **Practice Tips**: Hand-compute loss 6-token toy match code; add padding confirm masked no affect loss.
- **Lesson 7.2: Single-Head Attention (Lists)**
  - **Objectives**: Scaled dot-product attention + causal mask.
  - **Prerequisites**: 7.1.
  - **Estimated Time**: 4-6 hours.
  - **Key Subtopics**:
    - softmax(QK^T / sqrt(d)) @ V.
    - Causal mask (no peeking).
    - Shape/mask correctness.
  - **The Big Idea**: Position looks previous gather info.
  - **Analogy**: Library search – query finds books (keys), retrieves contents (values).
  - **stdlib Pain**: List matmul/transpose verbose/slow.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Attention All You Need paper.
  - **Demo Prompt**: (Attention func Q/K/V causal=False/True; main prints weights.)
  - **Practice Tips**: Print weights T=4 verify causal zeros; toy obvious match attention concentrates; stability checks no NaNs large Q/K.
- **Lesson 7.3: Attention with NumPy (Same Math, Sane Code)**
  - **Objectives**: Vectorize attention matmul/broadcasting.
  - **Prerequisites**: 7.2.
  - **Estimated Time**: 3-4 hours.
  - **Key Subtopics**:
    - Matmul QK^T.
    - Broadcasting mask/scale.
    - Parity vs lists.
  - **The Big Idea**: NumPy makes attention clean/fast.
  - **Analogy**: Replacing hand arithmetic spreadsheet – logic updates tables once.
  - **stdlib Pain**: List ops slow shape bugs.
  - **Power-Up**: Matrix ops clean code.
  - **Resource**: NumPy broadcasting docs.
  - **Demo Prompt**: (Attention_np parity list version tolerance; main benchmarks.)
  - **Practice Tips**: Benchmark list vs NumPy increasing (T,d); add batch dimension test shapes.
- **Lesson 7.4: Multi-Head Attention (PyTorch)**
  - **Objectives**: Head split/merge shape discipline; gradients flow.
  - **Prerequisites**: 7.3.
  - **Estimated Time**: 4-6 hours.
  - **Key Subtopics**:
    - Reshape (B,T,d) → (B,H,T,d//H).
    - Mask across heads.
    - Gradient validation.
  - **The Big Idea**: Heads attend different aspects parallel.
  - **Analogy**: Several librarians search parallel – head specializes cues.
  - **stdlib Pain**: Head splitting lists bug farm.
  - **Power-Up**: PyTorch reshape/transpose + autograd.
  - **Resource**: Annotated Transformer repo.
  - **Demo Prompt**: (MultiHeadAttentionTorch class, dummy loss backward no errors.)
  - **Practice Tips**: Assert shapes every stage tests; 1-head match single-head; debug mode returns attention matrices.
- **Lesson 7.5: Transformer Block (Attn + FFN + Residual + LayerNorm)**
  - **Objectives**: Assemble stable block.
  - **Prerequisites**: 7.4.
  - **Estimated Time**: 4-8 hours.
  - **Key Subtopics**:
    - LayerNorm.
    - Residual connections.
    - FFN.
    - Stacking contracts.
  - **The Big Idea**: Block combines attention feed-forward.
  - **Analogy**: Two-stage editor – gather context (attention), rewrite locally (FFN), skip copies.
  - **stdlib Pain**: Manual norm/residual loops.
  - **Power-Up**: PyTorch nn.LayerNorm/residual easy.
  - **Resource**: minGPT (~300 lines).
  - **Demo Prompt**: (TransformerBlock class, forward/backward random tensors.)
  - **Practice Tips**: Stack 2 blocks overfit tiny dataset near-zero loss; parameter counting; pre-norm vs post-norm compare stability micro experiment.
#### Tier 8 — Build a Tiny GPT Locally
**Goal:** Train/sample real decoder-only transformer (small).
**Total Time**: 16-26 hours.
- **Lesson 8.1: Training Loop Plumbing (LM-Specific)**
  - **Objectives**: Dataloader/optimizer/clipping/checkpoints.
  - **Prerequisites**: 5.3,7.1.
  - **Estimated Time**: 3-5 hours.
  - **Key Subtopics**:
    - Dataloader sequences.
    - Optimizer steps.
    - Gradient clipping.
    - Checkpointing.
  - **The Big Idea**: Plumbing gearbox for engine.
  - **Analogy**: Building gearbox – slips no improvement.
  - **stdlib Pain**: Manual batching/logging verbose.
  - **Power-Up**: N/A (PyTorch).
  - **Resource**: nanoGPT training loop.
  - **Demo Prompt**: (Trainer class log_loss/checkpoint/resume.)
  - **Practice Tips**: Add resume, verify identical; overfit 1 batch sanity.
- **Lesson 8.2: Tiny GPT End-to-End (PyTorch)**
  - **Objectives**: Decoder-only embeddings/blocks/LM head.
  - **Prerequisites**: 7.5,8.1.
  - **Estimated Time**: 8-12 hours.
  - **Key Subtopics**:
    - Decoder-only architecture.
    - Embeddings + positional.
    - Generation.
  - **The Big Idea**: Tiny GPT same principles big one, fewer params.
  - **Analogy**: Miniature steam engine – principles fewer pistons.
  - **stdlib Pain**: Manual tensor handling.
  - **Power-Up**: PyTorch nn.Embedding/nn.Module.
  - **Resource**: [nanoGPT](https://github.com/karpathy/nanoGPT) (~1000 lines).
  - **Demo Prompt**: (GPTModel class, train small corpus, generate samples.)
  - **Practice Tips**: Train small dataset loss decreases; deterministic sampling tests seeded snapshot.
- **Lesson 8.3: Decoding Integration (Greedy/Temp/Top-k/Top-p)**
  - **Objectives**: Decoding logits strategies.
  - **Prerequisites**: 1.4,2.2,8.2.
  - **Estimated Time**: 3-5 hours.
  - **Key Subtopics**:
    - Greedy/random.
    - Temperature.
    - Top-k/top-p.
  - **The Big Idea**: Strategies control diversity/coherence.
  - **Analogy**: Steering boat – rudders trade safety exploration.
  - **stdlib Pain**: Manual logit manipulation.
  - **Power-Up**: PyTorch multinomial/topk.
  - **Resource**: Transformers generation code.
  - **Demo Prompt**: (Decode func policy param, compare outputs.)
  - **Practice Tips**: Log diversity unique/repetition; repetition penalty observe changes.
- **Lesson 8.4: Evaluation as Tests (Seeded + Smoke Checks)**
  - **Objectives**: Minimal eval suite deterministic checks.
  - **Prerequisites**: 8.2.
  - **Estimated Time**: 2-4 hours.
  - **Key Subtopics**:
    - Loss sanity.
    - Overfit checks.
    - Deterministic decode.
  - **The Big Idea**: Eval tests airbags for wrong.
  - **Analogy**: Airbags crashes.
  - **stdlib Pain**: Manual verification.
  - **Power-Up**: Pytest asserts metrics.
  - **Resource**: lm-evaluation-harness (reference).
  - **Demo Prompt**: (Eval_suite loss_decreases/overfit_test.)
  - **Practice Tips**: Golden run config assert bounds; small perplexity benchmark held-out.
#### Tier 9 — Inference Reality: KV Cache, Quantization, Serving
**Goal**: Why inference systems exist.
**Total Time**: 8-12 hours.
- **Lesson 9.1: KV Cache Simulation + Minimal Cache**
  - **Objectives**: Implement KV caching incremental decoding.
  - **Prerequisites**: 7.2,8.3.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Cache vs recompute.
    - Speedup measurement.
  - **The Big Idea**: Cache avoids recomputing attention previous.
  - **Analogy**: Keeping scratch-work – no re-derive algebra.
  - **stdlib Pain**: Full recompute O(n^2) generation.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: KV cache explanations.
  - **Demo Prompt**: (Kv_cache func, measure speedup vs recompute.)
  - **Practice Tips**: Benchmark length vs runtime with/without; tests cached vs uncached match.
- **Lesson 9.2: Quantization – int8 by Hand**
  - **Objectives**: Simple symmetric int8 quantization weights.
  - **Prerequisites**: 3.4.
  - **Estimated Time**: 3-4 hours.
  - **Key Subtopics**:
    - Quantization formula.
    - Error/speed tradeoffs.
  - **The Big Idea**: Trades precision speed/memory.
  - **Analogy**: Compressing audio – smaller files distortion acceptable.
  - **stdlib Pain**: Manual scale/round.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Quantization papers.
  - **Demo Prompt**: (Quantize_int8 weights, measure error.)
  - **Practice Tips**: Quantize matrix multiply compare max error; log accuracy/loss tiny model.
- **Lesson 9.3: Local Model Runners + HTTP Client (stdlib → httpx)**
  - **Objectives**: Wrap local runner interface; serve HTTP.
  - **Prerequisites**: 0.8,8.2.
  - **Estimated Time**: 3-5 hours.
  - **Key Subtopics**:
    - Subprocess local runner.
    - HTTP serving.
    - Streaming tokens.
  - **The Big Idea**: Serving stable interface inference.
  - **Analogy**: Ordering kitchen – stable interface vs pantry.
  - **stdlib Pain**: Manual HTTP plumbing brittle clients.
  - **Power-Up**: httpx robust request/response/streaming.
  - **Resource**: [httpx docs](https://www.python-httpx.org).
  - **Demo Prompt**: (LLMClient generate prompt stream=True; stdlib http.server baseline, httpx client upgrade.)
  - **Practice Tips**: Implement stdlib endpoint first, upgrade httpx timeouts/retries; test failure modes.
- **Lesson 9.4: Production-Adjacent Local Model Runner (Earned Integration)**
  - **Objectives**: Integrate Ollama/llama.cpp as engine.
  - **Prerequisites**: 9.3.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Treat as external engine.
    - LLMClient.generate interface.
  - **The Big Idea**: Run real local models production-like.
  - **Analogy**: Swapping toy engine real one.
  - **stdlib Pain**: Manual subprocess management.
  - **Power-Up**: Ollama/llama.cpp for fast inference.
  - **Resource**: [Ollama docs](https://ollama.ai), [llama.cpp](https://github.com/ggerganov/llama.cpp).
  - **Demo Prompt**: (Client wrapper for Ollama generate; compare toy GPT real model.)
  - **Practice Tips**: Test streaming/temperature on real model; benchmark vs tiny GPT.
#### Tier 10 — Local RAG from First Principles
**Goal**: Build local RAG pipeline citations/verifiers.
**Total Time**: 23-35 hours.
- **Lesson 10.1: Chunking + Hashing + Metadata**
  - **Objectives**: Chunkers overlap/stable IDs.
  - **Prerequisites**: 0.8,1.1.
  - **Estimated Time**: 3-4 hours.
  - **Key Subtopics**:
    - Overlap chunks.
    - Stable hashes IDs.
    - Metadata source/offsets.
  - **The Big Idea**: Chunking prepares documents retrieval.
  - **Analogy**: Cutting book index cards – big unsearchable, small loses meaning.
  - **stdlib Pain**: Manual splitting loses context.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: RAG chunking articles.
  - **Demo Prompt**: (Chunk func with overlap/hash/metadata.)
  - **Practice Tips**: Compare sizes retrieval quality small corpus; tests stable IDs reruns.
- **Lesson 10.2: TF-IDF Retrieval (stdlib Sparse Dicts)**
  - **Objectives**: TF/IDF/cosine sparse space.
  - **Prerequisites**: 2.3,3.1.
  - **Estimated Time**: 4-6 hours.
  - **Key Subtopics**:
    - TF/IDF calc.
    - Sparse vectors.
    - Retrieve top-k query.
  - **The Big Idea**: TF-IDF weights rare words more.
  - **Analogy**: Giving rare words louder voices – uncommon terms signal.
  - **stdlib Pain**: Sparse dict ops manual verbose.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: TF-IDF papers.
  - **Demo Prompt**: (Tfidf_retriever class with fit/retrieve.)
  - **Practice Tips**: Tiny corpus verify ordering; add stopword handling measure effect.
- **Lesson 10.3: Earn scikit-learn TF-IDF**
  - **Objectives**: Reproduce stdlib TF-IDF sklearn.
  - **Prerequisites**: 10.2.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - sklearn TfidfVectorizer.
    - Pipeline ergonomics.
    - Consistency guarantees.
  - **The Big Idea**: sklearn robust vectorization fast sparse ops.
  - **Analogy**: Swapping hand-built gears machined parts – mechanism fewer defects.
  - **stdlib Pain**: Boilerplate/sparse math pitfalls.
  - **Power-Up**: Robust pipelines.
  - **Resource**: [scikit-learn docs](https://scikit-learn.org/stable/).
  - **Demo Prompt**: (Sklearn_tfidf parity stdlib ranking small corpus.)
  - **Practice Tips**: Compare runtime/ranking parity same corpus; save fitted vectorizer reload deterministically.
- **Lesson 10.4: Semantic Embeddings (Baseline → sentence-transformers)**
  - **Objectives**: Baseline embedding (bag-of-words mean); upgrade pretrained.
  - **Prerequisites**: 3.1,10.2.
  - **Estimated Time**: 3-5 hours.
  - **Key Subtopics**:
    - Bag-of-words baseline.
    - Pretrained embeddings.
    - Improvement measurement.
  - **The Big Idea**: Embeddings encode meaning, not word overlap.
  - **Analogy**: Coordinates meaning map – nearby points similar stuff.
  - **stdlib Pain**: Lexical matching misses paraphrases.
  - **Power-Up**: Semantic similarity no training scratch.
  - **Resource**: [sentence-transformers](https://github.com/UKPLab/sentence-transformers).
  - **Demo Prompt**: (Embedding_retriever baseline vs pretrained; evaluate synonym queries.)
  - **Practice Tips**: Evaluate retrieval synonym-heavy (TF-IDF struggles); cache embeddings chunk hash no recompute.
- **Lesson 10.5: Vector Search (Brute Force → FAISS)**
  - **Objectives**: Brute-force as oracle; upgrade FAISS.
  - **Prerequisites**: 3.2,10.4.
  - **Estimated Time**: 4-6 hours.
  - **Key Subtopics**:
    - Brute-force correctness.
    - FAISS indexing.
    - Latency measurement.
  - **The Big Idea**: ANN indexes millisecond retrieval scale.
  - **Analogy**: City map shortcuts – no check every street.
  - **stdlib Pain**: O(N) scans collapse scale.
  - **Power-Up**: Millisecond ANN.
  - **Resource**: [FAISS](https://github.com/facebookresearch/faiss).
  - **Demo Prompt**: (Vector_search brute vs FAISS; benchmark latency.)
  - **Practice Tips**: Benchmark latency vs corpus size plot; recall tests ANN contains exact top-k most time.
- **Lesson 10.6: Persistence + Filters (SQLite)**
  - **Objectives**: Store chunks/metadata/embeddings SQLite.
  - **Prerequisites**: 10.1.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - SQLite schema.
    - Filters source/date/tags.
    - Migrations/versioning.
  - **The Big Idea**: Persistence enables find tomorrow without re-reading.
  - **Analogy**: Labeling shelving – find without re-reading.
  - **stdlib Pain**: Ad-hoc files no queries.
  - **Power-Up**: Queryable persistence audits.
  - **Resource**: Python sqlite3 docs.
  - **Demo Prompt**: (Persistence class store/retrieve/filter.)
  - **Practice Tips**: Rebuild index DB verify identical retrieval; add migrations schema changes.
- **Lesson 10.7: Full RAG Pipeline + Citation Enforcement**
  - **Objectives**: Retrieve/assemble/generate/attach citations offsets.
  - **Prerequisites**: 9.3,10.5.
  - **Estimated Time**: 5-8 hours.
  - **Key Subtopics**:
    - Retrieve → context → generate.
    - Grounded answers.
    - Reject ungrounded.
  - **The Big Idea**: RAG combines retrieval generation grounded answers.
  - **Analogy**: Citation enforcement showing work – answers without evidence no pass review.
  - **stdlib Pain**: Manual context assembly.
  - **Power-Up**: N/A (integrate previous).
  - **Resource**: RAG papers.
  - **Demo Prompt**: (Rag_pipeline query corpus citations.)
  - **Practice Tips**: "Must quote exact span" verifier claims; small eval set track groundedness metrics.
#### Tier 11 — Memory: Application Memory, Not Mysticism
**Goal**: Memory = state + retrieval + policy + provenance.
**Total Time**: 14-22 hours.
- **Lesson 11.1: Short-Term Memory (Window + Deque)**
  - **Objectives**: Rolling conversation window token budget.
  - **Prerequisites**: 6.3,9.3.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Deque buffer.
    - Truncation policies.
    - Preserve important parts.
  - **The Big Idea**: Short-term bounded needs cleanup.
  - **Analogy**: Desk surface – useful limited needs cleanup.
  - **stdlib Pain**: Manual truncation loses key info.
  - **Power-Up**: collections.deque efficient.
  - **Resource**: Deque docs.
  - **Demo Prompt**: (Short_memory class add_message/get_context under budget.)
  - **Practice Tips**: Simulate long chats prove budget never exceeds; rule keep system instruction/user goal.
- **Lesson 11.2: Summarizing Memory (Heuristic → LLM, Cached)**
  - **Objectives**: Heuristic/LLM summaries caching; provenance.
  - **Prerequisites**: 11.1.
  - **Estimated Time**: 3-5 hours.
  - **Key Subtopics**:
    - Heuristic summaries.
    - LLM summaries.
    - Track messages fed summary.
  - **The Big Idea**: Summarization compresses history.
  - **Analogy**: Compressing notes – keep conclusions drop chatter.
  - **stdlib Pain**: Manual summary drift.
  - **Power-Up**: LLM for quality summaries.
  - **Resource**: Summary techniques articles.
  - **Demo Prompt**: (Summary_memory heuristic then LLM cached.)
  - **Practice Tips**: Evaluate drift 50 turns; regression tests stability fixed transcript.
- **Lesson 11.3: Long-Term Memory Store (SQLite + Audit Trail)**
  - **Objectives**: Append-only event log provenance queries.
  - **Prerequisites**: 10.6.
  - **Estimated Time**: 3-5 hours.
  - **Key Subtopics**:
    - Append-only log.
    - "Why believe this?" queries.
    - Write policies.
  - **The Big Idea**: Long-term searchable auditable.
  - **Analogy**: Indexed diary – searchable timestamped hard rewrite.
  - **stdlib Pain**: Loose files no audit.
  - **Power-Up**: sqlite3 queryable persistence.
  - **Resource**: SQLite docs.
  - **Demo Prompt**: (Long_memory store/retrieve with audit trail.)
  - **Practice Tips**: Write policies store decisions not noise; compaction preserves auditability.
- **Lesson 11.4: Vector Memory (Reuse Embeddings + FAISS)**
  - **Objectives**: Store memories embeddings retrieve similarity.
  - **Prerequisites**: 10.4,10.5.
  - **Estimated Time**: 3-4 hours.
  - **Key Subtopics**:
    - Embeddings storage.
    - Similarity retrieval.
    - Deduplication threshold.
  - **The Big Idea**: Vector recalls similar past.
  - **Analogy**: Remembering resemblance – similar situations not exact strings.
  - **stdlib Pain**: No semantic search.
  - **Power-Up**: Reuse RAG embeddings/FAISS.
  - **Resource**: Vector memory articles.
  - **Demo Prompt**: (Vector_memory store/retrieve similar.)
  - **Practice Tips**: Compare lexical vs semantic recall paraphrased; add deduplication semantic threshold.
- **Lesson 11.5: Knowledge-Graph Memory (Adjacency → networkx)**
  - **Objectives**: Model entities/relations graph; query neighborhoods.
  - **Prerequisites**: 11.3.
  - **Estimated Time**: 3-5 hours.
  - **Key Subtopics**:
    - Entities/relations.
    - Query neighborhoods.
    - Export subgraphs.
  - **The Big Idea**: Graphs capture relationships facts.
  - **Analogy**: Relationship map – facts meaning from connections.
  - **stdlib Pain**: Adjacency dicts limited ops.
  - **Power-Up**: networkx graph ops/algorithms.
  - **Resource**: [networkx docs](https://networkx.org).
  - **Demo Prompt**: (Knowledge_graph add_node/edge/query; main adds supports/contradicts, consistency check.)
  - **Practice Tips**: Add supports/contradicts/depends-on edges run consistency; export subgraph explaining rationale.
- **Lesson 11.6: Memory Policy (Store/Retrieve/Forget)**
  - **Objectives**: Define what becomes memory; retrieval gating; tombstones (soft delete).
  - **Prerequisites**: 11.1-11.5.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Confidence thresholds.
    - Citation requirements recalled facts.
  - **The Big Idea**: Memory without policy landfill opinions.
  - **Analogy**: Filing cabinet needs labels.
  - **stdlib Pain**: Unfiltered memory clutter.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Memory policy articles.
  - **Demo Prompt**: (Memory_policy enforce store/retrieve with thresholds.)
  - **Practice Tips**: Enforce policy tests (disallowed types never stored); simulate forget old/irrelevant.
#### Tier 12 — System Messages: Control Surfaces and Defenses
**Goal**: Predictable behavior/injection resistance/fail-closed outputs.
**Total Time**: 8-11 hours.
- **Lesson 12.1: Prompt Assembly as Typed Data**
  - **Objectives**: Represent prompts dataclasses; deterministic rendering/hashing caching.
  - **Prerequisites**: 0.5,9.3.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Dataclasses messages (system/user/tool).
    - Rendering hashing.
  - **The Big Idea**: Prompts structured data not ad-hoc strings.
  - **Analogy**: Compiling program – structure prevents nonsense.
  - **stdlib Pain**: String concat errors.
  - **Power-Up**: dataclasses typed assembly.
  - **Resource**: Python dataclasses docs.
  - **Demo Prompt**: (Prompt_assembly class with add_message/render/hash.)
  - **Practice Tips**: Snapshot rendering regression tests; prompt diff tool debugging changes.
- **Lesson 12.2: Injection Defense Baseline (Quarantine + Heuristics)**
  - **Objectives**: Separate instructions untrusted text; policies quoting/delimiters/refusal embedded instructions.
  - **Prerequisites**: 10.7,12.1.
  - **Estimated Time**: 3-4 hours.
  - **Key Subtopics**:
    - Quarantine retrieved text.
    - Delimiters/refusal policies.
  - **The Big Idea**: Untrusted input isolated sanitized.
  - **Analogy**: Handling suspicious mail – isolate decision-maker.
  - **stdlib Pain**: Mixed input injection risks.
  - **Power-Up**: Heuristics detect/refuse malicious.
  - **Resource**: Prompt injection articles.
  - **Demo Prompt**: (Injection_defense func quarantine/quote retrieved text.)
  - **Practice Tips**: Red-team corpus malicious snippets test defenses; safe context renderer always quotes retrieval.
- **Lesson 12.3: Structured Outputs + Validation (stdlib → pydantic)**
  - **Objectives**: Validate JSON stdlib; upgrade pydantic models strict parsing/errors.
  - **Prerequisites**: 12.1.
  - **Estimated Time**: 3-4 hours.
  - **Key Subtopics**:
    - Stdlib schema checks.
    - Pydantic models.
    - Retry/repair logic.
  - **The Big Idea**: Outputs typed validated.
  - **Analogy**: Customs forms – missing fields stop package.
  - **stdlib Pain**: Hand-rolled validation brittle.
  - **Power-Up**: pydantic strict schemas nice errors.
  - **Resource**: [pydantic docs](https://pydantic.dev).
  - **Demo Prompt**: (Validate_json stdlib then pydantic model; main forces malformed, shows retry.)
  - **Practice Tips**: Force malformed JSON verify retry/repair; test suite valid/invalid samples.
#### Tier 13 — Thinking: Workflow Control as Software
**Goal**: "Reasoning" pipelines/traces/verifiers/search.
**Total Time**: 15-24 hours.
- **Lesson 13.1: Plan → Execute → Verify Loop + Traces**
  - **Objectives**: Loop explicit plan steps/verification; produce replayable traces.
  - **Prerequisites**: 12.3.
  - **Estimated Time**: 4-6 hours.
  - **Key Subtopics**:
    - Plan/execute/verify stages.
    - JSONL traces.
    - Fail-closed verification fails.
  - **The Big Idea**: Reasoning structured loops not prayer.
  - **Analogy**: Pilot checklist – plan/act/confirm/repeat.
  - **stdlib Pain**: Ad-hoc steps no traceability.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Reasoning chain articles.
  - **Demo Prompt**: (Reasoning_loop func with plan/execute/verify; main emits JSONL trace.)
  - **Practice Tips**: Tracer emits JSONL each step; fail-closed policy verification fails.
- **Lesson 13.2: Tree/Graph Search for Solutions (stdlib → networkx)**
  - **Objectives**: BFS/DFS/scored best-first search; represent states/transitions.
  - **Prerequisites**: 3.2,13.1.
  - **Estimated Time**: 4-6 hours.
  - **Key Subtopics**:
    - Search algorithms.
    - State modeling scoring.
    - Pruning.
  - **The Big Idea**: Search explores solution spaces systematically.
  - **Analogy**: Exploring maze – systematic beats intuition stakes high.
  - **stdlib Pain**: Dict-based graphs limited.
  - **Power-Up**: networkx queries/algorithms.
  - **Resource**: [networkx docs](https://networkx.org).
  - **Demo Prompt**: (Search func BFS/DFS/best-first; main solves small puzzle.)
  - **Practice Tips**: Solve planning puzzle (grid/scheduling) search; pruning rules measure gains.
- **Lesson 13.3: Tool-Augmented Reasoning Runtime (Registry + Dispatch)**
  - **Objectives**: Typed tool registry/dispatch; input/output validation/timeouts.
  - **Prerequisites**: 12.3.
  - **Estimated Time**: 3-5 hours.
  - **Key Subtopics**:
    - Tool registry.
    - Dispatch validation.
    - Sandbox rules.
  - **The Big Idea**: Tools extend LLM controlled interfaces.
  - **Analogy**: Prosthetic limbs – powerful safe strict controls.
  - **stdlib Pain**: Manual tool calls error-prone.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Tool calling articles.
  - **Demo Prompt**: (Tool_runtime register/call with validation.)
  - **Practice Tips**: Implement tools calculator/file/retriever/code stub; tests invalid calls rejected.
- **Lesson 13.4: Workflow DAG Executor (Optional)**
  - **Objectives**: Execute multi-step workflows dependency graphs; cache outputs skip recompute.
  - **Prerequisites**: 13.2,13.3.
  - **Estimated Time**: 4-7 hours.
  - **Key Subtopics**:
    - DAG execution.
    - Caching nodes.
    - Cancellation/partial failures.
  - **The Big Idea**: Workflows DAGs cached outputs.
  - **Analogy**: Build system – tasks run dependencies ready.
  - **stdlib Pain**: Manual sequencing no caching.
  - **Power-Up**: N/A (pure stdlib/networkx optional).
  - **Resource**: DAG executor examples.
  - **Demo Prompt**: (Dag_executor with dependencies/cache.)
  - **Practice Tips**: Workflow ingest→index→retrieve→answer→verify; cancellation partial failure handling.
#### Tier 14 — Analysis: Text, PDFs, OCR, Images, Datasets
**Goal**: Ingest real-world mess structured artifacts.
**Total Time**: 21-32 hours.
- **Lesson 14.1: Text Extraction → Structured JSON**
  - **Objectives**: Clean/normalize text structured records; preserve provenance.
  - **Prerequisites**: 10.1,12.3.
  - **Estimated Time**: 3-4 hours.
  - **Key Subtopics**:
    - Normalization.
    - Entity/claim records.
    - Provenance source/offsets/hashes.
  - **The Big Idea**: Extraction distills signal noise.
  - **Analogy**: Distilling ore – separate signal slag.
  - **stdlib Pain**: Manual parsing loses provenance.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Text extraction articles.
  - **Demo Prompt**: (Extract_text to JSON with provenance.)
  - **Practice Tips**: Extract entities/claims typed records validators; tests stable hashing/deterministic output.
- **Lesson 14.2: PDFs Without OCR (stdlib Pain → PyMuPDF/pdfplumber)**
  - **Objectives**: Prove stdlib can't parse PDFs; extract text/layout real library.
  - **Prerequisites**: 0.8.
  - **Estimated Time**: 4-6 hours.
  - **Key Subtopics**:
    - Stdlib crude hacks fail.
    - Layout-aware extraction.
    - Page metadata.
  - **The Big Idea**: PDFs require specialized tools extract content.
  - **Analogy**: Sealed shipping containers – right tools unload.
  - **stdlib Pain**: PDFs not plain text; parsing trap.
  - **Power-Up**: Stable extraction metadata/page structure.
  - **Resource**: [PyMuPDF docs](https://pymupdf.readthedocs.io).
  - **Demo Prompt**: (Stdlib attempt fail; PyMuPDF extract text/layout.)
  - **Practice Tips**: Extract 3 PDFs compare formatting loss; "PDF → chunks → RAG" ingestion step.
- **Lesson 14.3: PDFs with OCR (subprocess Pain → pytesseract + Pillow)**
  - **Objectives**: Render pages images, OCR, clean artifacts; track confidence.
  - **Prerequisites**: 14.2.
  - **Estimated Time**: 5-8 hours.
  - **Key Subtopics**:
    - Render to images.
    - OCR run.
    - Clean common artifacts.
  - **The Big Idea**: OCR recovers text images inherent noise.
  - **Analogy**: Reading frosted glass – recover text errors guaranteed.
  - **stdlib Pain**: Scanned docs images; no text parse.
  - **Power-Up**: Image→text recovery index/cite.
  - **Resource**: [pytesseract docs](https://github.com/madmaze/pytesseract).
  - **Demo Prompt**: (Subprocess render/OCR pain; pytesseract + Pillow pipeline.)
  - **Practice Tips**: Compare accuracy different DPI; verifier flags low-confidence pages.
- **Lesson 14.4: Images (Pillow → optional OpenCV)**
  - **Objectives**: Load/transform images compute features (histograms/edges); standard pipeline.
  - **Prerequisites**: 0.8.
  - **Estimated Time**: 4-6 hours.
  - **Key Subtopics**:
    - Load/transform.
    - Simple features.
    - Pipeline inputs/outputs.
  - **The Big Idea**: Processing reveals structure visual data.
  - **Analogy**: Adjusting microscope – changes reveal/hide structure.
  - **stdlib Pain**: No image handling.
  - **Power-Up**: Pillow transforms/crops/preprocessing.
  - **Resource**: [Pillow docs](https://pillow.readthedocs.io).
  - **Demo Prompt**: (Image_preprocess with transforms; main creates thumbnail + metadata JSON.)
  - **Practice Tips**: Pipeline thumbnails + metadata JSON; tests deterministic transforms fixed image.
- **Lesson 14.5: Datasets (csv/statistics Pain → pandas/duckdb)**
  - **Objectives**: EDA stdlib csv first (pain), then pandas/duckdb.
  - **Prerequisites**: 0.8,3.4.
  - **Estimated Time**: 5-8 hours.
  - **Key Subtopics**:
    - Manual parsing/aggregation.
    - Groupby/joins hand.
    - Queries large files.
  - **The Big Idea**: Dataframes make analysis operations not loops.
  - **Analogy**: Well-labeled spreadsheets program – analysis operations.
  - **stdlib Pain**: Manual aggregation unmaintainable.
  - **Power-Up**: Fast expressive analysis/querying.
  - **Resource**: [pandas docs](https://pandas.pydata.org), [duckdb docs](https://duckdb.org).
  - **Demo Prompt**: (Csv module manual stats; pandas groupby; duckdb SQL on file.)
  - **Practice Tips**: Compute groupby stats hand then pandas replicate; duckdb query CSV compare speed.
- **Lesson 14.6: SQL-on-Files (duckdb)**
  - **Objectives**: Query big-ish data without loading all; export results citable evidence.
  - **Prerequisites**: 14.5.
  - **Estimated Time**: 2-4 hours.
  - **Key Subtopics**:
    - SQL views.
    - Parquet.
    - Reproducible queries.
  - **The Big Idea**: SQL reasoning tool data.
  - **Analogy**: Microscope tables.
  - **stdlib Pain**: Full load memory intensive.
  - **Power-Up**: duckdb SQL on files fast.
  - **Resource**: DuckDB docs.
  - **Demo Prompt**: (Duckdb_query on CSV/Parquet; main exports results JSON with provenance.)
  - **Practice Tips**: Create query scripts; store outputs with provenance; benchmark vs pandas large files.
- **Lesson 14.7: Index Ingestion Outputs into RAG (Citations Across Modalities)**
  - **Objectives**: Convert records chunks; retrieval returns page/spans/source metadata.
  - **Prerequisites**: 10.1-10.6,14.2-14.6.
  - **Estimated Time**: 2-4 hours.
  - **Key Subtopics**:
    - Citation format.
    - Cross-modal IDs.
  - **The Big Idea**: Multi-source RAG needs unified citation standard.
  - **Analogy**: Bibliography across books/papers/datasets.
  - **stdlib Pain**: No cross-modality handling.
  - **Power-Up**: N/A (integrate previous).
  - **Resource**: Multi-modal RAG articles.
  - **Demo Prompt**: (Index_ingestion for PDFs/images/CSVs into RAG; query with citations pointing page/spans.)
  - **Practice Tips**: Answer queries citations pointing page numbers/spans; test cross-modal (e.g., cite image description + table stat).
#### Tier 15 — Specialized Agents: Multi-Component Systems
**Goal**: Agents loops + tools + memory + policies + logs.
**Total Time**: 15-24 hours.
- **Lesson 15.1: Single-Agent Loop (No Frameworks)**
  - **Objectives**: Minimal agent state machine tool calls; log actions enforce boundaries.
  - **Prerequisites**: 13.3.
  - **Estimated Time**: 4-6 hours.
  - **Key Subtopics**:
    - Observe/decide/act/update.
    - Log actions.
    - Tool boundaries.
  - **The Big Idea**: Agents loops: observe/decide/act/report/repeat.
  - **Analogy**: Intern checklist – observe/decide/act/report/repeat.
  - **stdlib Pain**: Ad-hoc state management.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Agent loop articles.
  - **Demo Prompt**: (Single_agent class with loop/tool_call.)
  - **Practice Tips**: Add tools retriever/calculator/file; policy every answer cite retrieval or "no evidence".
- **Lesson 15.2: Specialized Agents (Research/Code/Data/Doc)**
  - **Objectives**: 4 role agents distinct tool access/output schemas; route tasks rules/classifier.
  - **Prerequisites**: 15.1,12.3.
  - **Estimated Time**: 6-10 hours.
  - **Key Subtopics**:
    - Research/code/data/doc agents.
    - Distinct tools/schemas.
    - Routing.
  - **The Big Idea**: Specialists optimized narrow tasks.
  - **Analogy**: Different lab instruments – optimized narrow measurement.
  - **stdlib Pain**: Generic agents no specialization.
  - **Power-Up**: N/A (build on single-agent).
  - **Resource**: Specialized agent examples.
  - **Demo Prompt**: (Specialized_agents classes with tools/schemas; router func.)
  - **Practice Tips**: "Code agent" outputs patches/runs tests; "data agent" generates pandas/duckdb queries.
- **Lesson 15.3: Multi-Agent Orchestration (Supervisor + Workers)**
  - **Objectives**: Supervisor decomposes tasks workers; aggregate results verification/conflict detection.
  - **Prerequisites**: 15.2,16.2 (parallel best after judges).
  - **Estimated Time**: 5-8 hours.
  - **Key Subtopics**:
    - Decompose subtasks.
    - Aggregate verification.
    - Conflict detection.
  - **The Big Idea**: Orchestration maintains coherence specialists.
  - **Analogy**: Film director – specialists act, supervisor coherence.
  - **stdlib Pain**: Manual coordination.
  - **Power-Up**: N/A (build on specialized).
  - **Resource**: Multi-agent orchestration articles.
  - **Demo Prompt**: (Orchestrator class decompose/assign/aggregate.)
  - **Practice Tips**: Run 3 workers parallel merge outputs citations; "disagreement detector" flags contradictions.
#### Tier 16 — Expert Judging and Continuous Synthesis
**Goal**: Quality from evaluation loops, not hope.
**Total Time**: 13-20 hours.
- **Lesson 16.1: Generate Multiple Candidates (Seeded Diversity)**
  - **Objectives**: Generate N candidates controlled diversity knobs; track seeds/configs repeatability.
  - **Prerequisites**: 8.3,0.6.
  - **Estimated Time**: 2-3 hours.
  - **Key Subtopics**:
    - Controlled diversity.
    - Seeded generation.
    - Cluster similarity.
  - **The Big Idea**: Candidate generation variety without chaos.
  - **Analogy**: Brainstorming constraints – variety without chaos.
  - **stdlib Pain**: Uncontrolled randomness.
  - **Power-Up**: N/A (use sampling).
  - **Resource**: Diversity generation articles.
  - **Demo Prompt**: (Generate_candidates with diversity knobs; main clusters by similarity.)
  - **Practice Tips**: Generate 10 candidates cluster similarity; save with metadata/hashes.
- **Lesson 16.2: Judges and Scorecards (Rubric Objects)**
  - **Objectives**: Define scorecards typed objects (accuracy/grounding/completeness/clarity); machine-checkable evaluation outputs.
  - **Prerequisites**: 12.3.
  - **Estimated Time**: 3-5 hours.
  - **Key Subtopics**:
    - Rubric objects.
    - Automatic checks (e.g., citations).
    - Store evaluations SQLite trends.
  - **The Big Idea**: Rubrics make evaluation comparable not vibes.
  - **Analogy**: Grading key – answers comparable instead vibes.
  - **stdlib Pain**: Subjective evaluation inconsistent.
  - **Power-Up**: N/A (dataclasses/SQLite).
  - **Resource**: Rubric design articles.
  - **Demo Prompt**: (Scorecard class evaluate; judge rejects missing citations.)
  - **Practice Tips**: Build judge rejects missing citations auto; store evaluations SQLite trend tracking.
- **Lesson 16.3: Pros/Cons + Synthesis with Rationale**
  - **Objectives**: Extract pros/cons per candidate; synthesize final traceability (which contributed what).
  - **Prerequisites**: 16.1,16.2.
  - **Estimated Time**: 4-6 hours.
  - **Key Subtopics**:
    - Pros/cons extraction.
    - Synthesis merge.
    - Traceability citations.
  - **The Big Idea**: Synthesis combines strengths reduces brittleness.
  - **Analogy**: Alloying metals – combine strengths reduce brittleness.
  - **stdlib Pain**: Manual merge loses traceability.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Synthesis techniques articles.
  - **Demo Prompt**: (Synthesis func pros/cons/merge with rationale.)
  - **Practice Tips**: Force two disagree require reconciliation; constraint synthesis cite sources/spans.
- **Lesson 16.4: Regression Suites + Metrics Tracking (SQLite)**
  - **Objectives**: Fixed eval set run every change; track metrics time detect regressions.
  - **Prerequisites**: 16.2.
  - **Estimated Time**: 4-6 hours.
  - **Key Subtopics**:
    - Fixed eval set.
    - Metrics tracking.
    - Regression detection.
  - **The Big Idea**: Regression tests prevent re-learning old mistakes.
  - **Analogy**: Immune memory – stops relearning old mistakes.
  - **stdlib Pain**: No automated tracking.
  - **Power-Up**: SQLite for trend storage.
  - **Resource**: Regression testing articles.
  - **Demo Prompt**: (Regression_suite run_eval/track_metrics; fail if regress threshold.)
  - **Practice Tips**: Add golden answers deterministic tasks; fail CI metrics regress beyond threshold.
#### Tier 17 — Credit-Sparing Sub-Tasks, Parallelism, QA Chains
**Goal**: Model calls only irreducible; verify cheaply first.
**Total Time**: 10-15 hours.
- **Lesson 17.1: Deterministic Sub-Tasks + Caching (SQLite)**
  - **Objectives**: Hash inputs cache outputs; separate deterministic from LLM steps.
  - **Prerequisites**: 10.6,13.4.
  - **Estimated Time**: 3-4 hours.
  - **Key Subtopics**:
    - Input hashing.
    - Cache outputs.
    - Hit-rate logging.
  - **The Big Idea**: 80% work no need LLM call.
  - **Analogy**: Remembering solved exercises – no redo every time.
  - **stdlib Pain**: Redundant computation.
  - **Power-Up**: sqlite3 caching.
  - **Resource**: Caching patterns articles.
  - **Demo Prompt**: (Cache_layer get/set by hash; main caches embeddings/chunking.)
  - **Practice Tips**: Cache embeddings/chunking by hash; add hit-rate logging JSONL.
- **Lesson 17.2: Parallel Sub-Tasks (asyncio + futures)**
  - **Objectives**: Run independent subtasks concurrent timeouts; aggregate deterministically.
  - **Prerequisites**: 17.1.
  - **Estimated Time**: 3-5 hours.
  - **Key Subtopics**:
    - Concurrency.
    - Timeouts.
    - Aggregation.
  - **The Big Idea**: Parallelism speeds independent work.
  - **Analogy**: Hiring more hands – faster coordination tight.
  - **stdlib Pain**: Sequential slow.
  - **Power-Up**: asyncio concurrent futures parallel.
  - **Resource**: Python asyncio docs.
  - **Demo Prompt**: (Parallel_executor run_tasks aggregate.)
  - **Practice Tips**: Parallelize retrieval sources merge top-k; add cancellation runaway tasks.
- **Lesson 17.3: QA Sub-Task Chains for Certainty**
  - **Objectives**: Layered verifiers schema/citation/consistency; produce pass/fail + reasons.
  - **Prerequisites**: 12.3,16.2.
  - **Estimated Time**: 4-6 hours.
  - **Key Subtopics**:
    - Layered verifiers.
    - Repair loops.
    - Final artifact.
  - **The Big Idea**: QA chains catch defects shipping.
  - **Analogy**: Multi-stage factory inspection – caught before shipping.
  - **stdlib Pain**: Manual verification inconsistent.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: QA chain articles.
  - **Demo Prompt**: (Qa_chain verify layered; main adversarial examples fail stages.)
  - **Practice Tips**: Adversarial examples fail each stage; add repair loop attempts fixes re-verifies.
#### Tier 18 — Organization Finale: Cognitive Operating System
**Goal**: Persistent context + graph organization + safe exploration + coherent synthesis.
**Total Time**: 28-45 hours.
- **Lesson 18.1: Project-Wide Context Store (Append-Only Event Log)**
  - **Objectives**: Append-only log decisions/assumptions/evidence/outcomes; queries.
  - **Prerequisites**: 10.6,16.4.
  - **Estimated Time**: 4-6 hours.
  - **Key Subtopics**:
    - Append-only events.
    - "Explain why" queries trace supporting.
    - Integrate agents/chats.
  - **The Big Idea**: Shared state persists sessions.
  - **Analogy**: Accounting ledger – no edit history, append truth/corrections.
  - **stdlib Pain**: Loose state lost restarts.
  - **Power-Up**: sqlite3 queryable log.
  - **Resource**: Event sourcing articles.
  - **Demo Prompt**: (Context_store append_event/query_trace.)
  - **Practice Tips**: Store run config/metrics events; "explain why" queries trace supporting events.
- **Lesson 18.2: Mind-Map Analytic Engine (Graph + SQLite Persistence)**
  - **Objectives**: Typed nodes/edges (goal/decision/task/evidence/output); persist graph; structural analyses (reachability/contradictions).
  - **Prerequisites**: 11.5,18.1.
  - **Estimated Time**: 6-10 hours.
  - **Key Subtopics**:
    - Node/edge types.
    - Persistence queries.
    - Analyses.
  - **The Big Idea**: Mind maps organize knowledge searchable structure.
  - **Analogy**: Filesystem ideas – nodes files, edges links, queries search.
  - **stdlib Pain**: Dict graphs no persistence/queries.
  - **Power-Up**: networkx ops (after adjacency pain).
  - **Resource**: [networkx docs](https://networkx.org).
  - **Demo Prompt**: (MindMap add_node/edge/persist/query; main adds supports/contradicts, consistency check.)
  - **Practice Tips**: Add edge types supports/contradicts/depends-on/derived-from run consistency; export context bundle node (ancestors + evidence).
- **Lesson 18.3: Visual Export (DOT → graphviz)**
  - **Objectives**: Export graph DOT deterministically; generate visual snapshots per fork/run.
  - **Prerequisites**: 18.2.
  - **Estimated Time**: 3-5 hours.
  - **Key Subtopics**:
    - DOT export.
    - Visual snapshots.
    - Stable ordering diffs.
  - **The Big Idea**: Visualization reveals structure can't hold memory.
  - **Analogy**: Turning stack trace diagram – see structure.
  - **stdlib Pain**: Manual DOT string building tedious.
  - **Power-Up**: graphviz rendering (after DOT pain).
  - **Resource**: [graphviz docs](https://graphviz.org).
  - **Demo Prompt**: (To_dot func; main exports subgraphs.)
  - **Practice Tips**: Export subgraph single decision evidence; stable node ordering meaningful diffs.
- **Lesson 18.4: Non-Destructive Forking (Copy-on-Write Overlays, Diff/Merge)**
  - **Objectives**: Fork mindmap without copying everything; diff/merge conflicts detection; retain provenance.
  - **Prerequisites**: 18.2.
  - **Estimated Time**: 6-10 hours.
  - **Key Subtopics**:
    - Copy-on-write overlays.
    - Diff computation.
    - Merge conflict detection.
  - **The Big Idea**: Forks explore without collapsing original.
  - **Analogy**: Parallel universes – explore without collapsing timeline.
  - **stdlib Pain**: Full copies wasteful.
  - **Power-Up**: N/A (pure stdlib).
  - **Resource**: Git-like forking articles.
  - **Demo Prompt**: (Fork_mindmap, diff/merge with conflict check.)
  - **Practice Tips**: Run two competing forks merge winner; track provenance merged nodes/edges.
- **Lesson 18.5: Prefabricated Nodes/Decision Trees from One Prompt**
  - **Objectives**: Generate placeholder nodes (tasks/questions) goal spec; link dependency tree; validate completeness.
  - **Prerequisites**: 13.4,18.2.
  - **Estimated Time**: 4-6 hours.
  - **Key Subtopics**:
    - Placeholder generation single prompt.
    - Dependency linking.
    - Completeness validation.
  - **The Big Idea**: Prefabs set structure fast fill details safely.
  - **Analogy**: Scaffolding building site – set structure fast fill details safely.
  - **stdlib Pain**: Manual node creation tedious.
  - **Power-Up**: scikit-learn trees (after manual if-trees pain).
  - **Resource**: [scikit-learn decision trees](https://scikit-learn.org/stable/modules/tree.html).
  - **Demo Prompt**: (Prefab_nodes from prompt generate placeholders/link; main tracks completion.)
  - **Practice Tips**: Prefab "build local RAG" track completion per node; validation all placeholders resolved before "ship".
- **Lesson 18.6: Expert Characters + Alignment/Coherence Gates**
  - **Objectives**: Expert profiles stable evaluation styles/constraints; enforce coherence linked nodes/decisions; alignment strategic goals.
  - **Prerequisites**: 16.2,18.4.
  - **Estimated Time**: 5-8 hours.
  - **Key Subtopics**:
    - Expert profiles consistent stances.
    - Coherence checks linkages.
    - Alignment gates pass before merge.
  - **The Big Idea**: Experts review board enforce coherence time/forks.
  - **Analogy**: Review board – consistent stances enforce coherence.
  - **stdlib Pain**: Subjective evaluation inconsistent.
  - **Power-Up**: N/A (dataclasses for profiles).
  - **Resource**: LLM judge articles.
  - **Demo Prompt**: (Expert_character class get_system_prompt/evaluate; panel run on forks store judgments nodes.)
  - **Practice Tips**: Run experts competing forks store judgments graph nodes; "strategic goal alignment" gate pass before merge.
#### The Cognitive Operating System Capstone
**Goal**: Build CLI tool where vague goal (e.g., "Plan marketing campaign") decomposes into mind map, agents fill nodes parallel, experts judge/synthesize, fork alternatives, merge best – persistent state across sessions.
**Metaphor Table**:
| Component | Role | Analogy |
|-----------|------|---------|
| LLM | CPU | Raw intelligence |
| Short-term Memory | RAM | Context management |
| Long-term Memory | Disk | Persistent storage |
| Mind Map | Filesystem | Knowledge organization |
| Orchestrator | Process Manager | Sub-task coordination |
| Experts | Judges | Quality control |
| DAG Executor | Scheduler | Workflow execution |
**Total Time**: ~200-320 hours (20-30 weeks at 10 hrs/week).
**Milestone Map**:
| End of Tier | Capability |
|-------------|------------|
| 3 | Similarity/matrix intuition (build retrieval baselines) |
| 5 | Autodiff/PyTorch (train real models) |
| 7 | Attention/transformer blocks (understand core machine) |
| 8 | **Tiny local GPT trained/sampled** |
| 10 | **Local RAG citations/verifiers** |
| 11 | Memory systems (short/long/vector/graph) |
| 12 | System messages/injection defense |
| 13 | Thinking pipelines/tool-augmented reasoning |
| 14 | Real-world ingestion (PDF/OCR/images/data) |
| 16 | Evaluation/synthesis/regression discipline |
| 17 | Credit-sparing subtasks/parallelism/QA chains |
| 18 | **Cognitive OS mind-map/forks/expert governance** |
#### Appendix A: Open Source Spine
Read code, not blogs. Primary references:
| Project | Purpose | Lines | Link |
|---------|---------|-------|------|
| micrograd | Autodiff | ~100 | github.com/karpathy/micrograd |
| minGPT | Clean GPT | ~300 | github.com/karpathy/minGPT |
| nanoGPT | Training | ~1000 | github.com/karpathy/nanoGPT |
| tiktoken | Tokenization | - | github.com/openai/tiktoken |
| sentence-transformers | Embeddings | - | github.com/UKPLab/sentence-transformers |
| FAISS | Vector search | - | github.com/facebookresearch/faiss |
| ChromaDB | Vector DB | - | github.com/chroma-core/chroma |
| LangChain | RAG/Agents | - | github.com/langchain-ai/langchain |
| CrewAI | Multi-agent | - | github.com/joaomdmoura/crewAI |
#### Appendix B: "Aha!" Library Summary
| Library | Pain | Relief |
|---------|------|--------|
| NumPy | Triple loops matrices | `@` C, 100x faster |
| PyTorch | Hand-derive gradients | `loss.backward()` |
| tiktoken | BPE edges/Unicode bugs | Battle-tested GPT-compatible |
| sentence-transformers | Train embeddings scratch | Pre-trained semantic similarity |
| FAISS | O(N) brute search | Millisecond ANN |
| ChromaDB | Manual embedding/metadata | Simple API |
| PyMuPDF | Can't read PDF binary | 3-line extract |
| pytesseract | Can't read scanned images | Image → text |
| Pandas | csv manual stats | One-line EDA |
| LangChain | 100+ lines agent loop | 5-line agents |
#### Appendix C: Experimental Discipline
Every experiment:
```python
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def log_experiment(config: Dict[str, Any], results: Dict[str, Any],
                   log_file: str = "experiments.jsonl") -> None:
    log = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results
    }
    Path(log_file).open("a").write(json.dumps(log) + "\n")
def run_ablation(base_config: Dict[str, Any], key: str,
                 values: List[Any], run_fn) -> List[Dict]:
    results = []
    for value in values:
        config = {**base_config, key: value}
        set_seed(config.get("seed", 42))
        result = run_fn(config)
        log_experiment(config, result)
        results.append({"config": config, "result": result})
    return results
```
#### Appendix D: Side Quests (Optional)
Not critical path, provide context:
- **SQ-A: Classic ML Interlude** (after Tier 4): scikit-learn classifiers/regressors, feature engineering, cross-validation.
- **SQ-B: RNN/LSTM History Detour** (after Tier 7): Why attention replaced recurrence, vanishing gradient, bidirectional RNNs.
- **SQ-C: Earn Agent Framework** (after Tier 15): Map primitives to LangChain/LlamaIndex/CrewAI, understand abstractions.
- **SQ-D: Deployment & Serving** (after Tier 9): Docker containers, load balancing, monitoring.
- **SQ-E: Fine-Tuning Techniques** (after Tier 8): LoRA/PEFT, RLHF basics, evaluation-driven iteration.
#### Final Tips
- **Projects**: Hands-on Python scripts building systems (e.g., capstone CLI for goal planning with mind maps/agents).
- **Resources**: uv/pytest/ruff/mypy docs, "Hands-On ML" book, Andrew Ng courses, Karpathy videos, papers (e.g., Attention All You Need).
- **Assessment**: Run pytest/ruff/mypy per lesson.
- **Next**: Scale to distributed systems/web apps.
