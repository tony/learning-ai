# Writing

How this project writes prose, for humans and agents alike. It governs
`README.md`, commit messages, docstrings, and source comments — every surface
a reader reaches in this repository.

For environment setup, the gates, and pull request workflow, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Voice

Three surfaces, one voice. A docstring says what a caller may rely on; a
commit message says why a change was made; prose says what happens. All three
are present tense, lead with the thing being described, and stop. Why a lesson
was structured a certain way belongs in the commit message, which is
timestamped and attached to the diff.

The most useful editing operation is deleting the introductory sentence.

Lead with verbs and name concrete things. Put identifiers in backticks. Prefer
short declarative sentences, one operational fact each. Do not explain Python
to Python developers; do explain what a lesson demonstrates and why.

Type annotations describe shape. Documentation describes meaning. A sentence
that restates a signature has said nothing.

Use MUST, SHOULD, and MAY only where the normative sense is meant. Say what
actually happens rather than that something is "supported".

| Instead of                       | Prefer                             |
| --------------------------------- | ---------------------------------- |
| "We added…"                       | "`benchmark()` now accepts…"       |
| "New and improved"                | "`log_experiment` now…"            |
| "powerful", "seamless"            | state the capability               |
| "easily", "simply", "just"        | omit                               |
| "simple", "obvious", "intuitive"  | omit                                |
| "robust"                          | name the failure that is handled   |
| "comprehensive"                   | name what is covered               |
| "production-ready"                | state the guarantee                |
| "optimized", "blazingly fast"     | give the magnitude                 |
| "various fixes"                   | name the components                |
| "under the hood"                  | omit unless observable             |
| "please note that", "note that"   | state the fact                     |
| "leverage", "utilize"             | "use"                              |
| "delve into"                      | "read", or omit                    |
| "best practices"                  | name the practice                  |
| "in order to"                     | "to"                                |

## Who you are writing for

The default reader is fluent in Python and new to this lesson — often that
reader is you, months later. They can read a signature; they cannot recall why
a lesson made the choice it did. Serve them first.

- **Second person, present tense, active.** "You seed the generator", not "The
  generator is seeded".
- **Concept before code.** Open a lesson's module docstring by saying what the
  lesson demonstrates and why it matters, before the signature or the
  complexity notes. The template's own shape — Big Idea, then Why It Matters,
  then the pain/power-up progression — is this rule applied to a lesson.
- **Say when they can stop.** Lead with the default path and let a skimmer
  leave after one paragraph.
- **Name the trade-off.** If a power-up costs something a stdlib version does
  not — a dependency, a loss of transparency — say so, and say what it buys.

## README

A README is the shortest path from "what is this?" to running the first
lesson, not the project's autobiography.

The first sentence is a contract: it says what this repository is, concretely
enough that a reader knows whether to keep reading.

Get to a runnable command before anything the reader can skip.

State the minimum Python version in prose, not only as a requirement in
`pyproject.toml` — `requires-python` is the authority; the README must agree
with it.

Shell examples are executable, not illustrative fiction. Never
`your-command <some-options>`. See
[Documented examples that run](#documented-examples-that-run) for which blocks
are collected as tests and which are prose that merely looks like a command.

State defaults explicitly. State negative guarantees where they exist — what a
lesson does not do, what a script does not write outside its own artifact
directory.

Headings stay conventional and stable, because people deep-link them.

## Documented examples that run

Examples in this repository are tests, but only some of them, and knowing
which ones is the whole point of this section.

**A fence tag is cosmetic. Only a `>>> ` prompt executes, and only inside a
`.py` file under `testpaths`.** `[tool.pytest]` in `pyproject.toml` sets
`addopts = [..., "--doctest-modules"]`, which makes pytest import every Python
module under `testpaths = ["src", "notes"]` and run its docstring doctests.
There is no markdown doctest collector configured — no `--doctest-glob`, no
docutils or sybil plugin. A `>>> ` block inside a `.md` file (for example in
`notes/lesson_plan.md` or `notes/libs/`) is not executed by anything; it reads
as a test but is not one. Do not mistake a prompted block in a markdown file
for a doctest here — that mistake is invisible because nothing fails.

`README.md` is not in `testpaths`. Do not add `>>> ` prompts to the README —
a prompt implies the block is verified, and here it would not be. Illustrative
shell or Python snippets in the README use a plain fenced block instead.

**Removing prompts from a `.py` docstring silently deletes a test.** A block
written as

    ```python
    solve_pain([1, 2, 3])
    ```

is prose that looks like a test. The same block with prompts is a test:

    ```python
    >>> solve_pain([1, 2, 3])
    6
    ```

When editing a file under `src/` or `notes/` that contains examples, count the
prompts before and after.

**The fence tag is `python`**, not `pycon`, not bare, even though some
existing reference notes under `notes/libs/` use `pycon` — those files are
markdown and inert either way; match `python` in anything meant to run.

**No `doctest_namespace` fixture exists.** There is no `conftest.py` in this
repository, so a doctest starts with an empty namespace: it cannot use
`tmp_path` or any other pytest fixture. Import everything a block needs inside
the block itself.

**`# doctest: +SKIP` is not permitted.** It tests nothing. If an example
cannot pass, fix the example or fix the code — never downgrade it to a
`.. code-block::` or an unprompted fence to make it pass.

**Every function and method has a working doctest.** If you cannot write one
that actually exercises the code, stop and ask rather than committing a
placeholder.

**Option flags.** `ELLIPSIS` and `NORMALIZE_WHITESPACE` are enabled globally
via `doctest_optionflags`, so `...` elides variable output and whitespace
differences do not fail a comparison:

    Examples
    --------
    >>> import time
    >>> time.time()  # doctest: +ELLIPSIS
    1...

**Async doctests** wrap the coroutine in `asyncio.run` so the doctest stays
synchronous:

    >>> import asyncio
    >>> async def example():
    ...     return "result"
    >>> asyncio.run(example())
    'result'

`notes/libs/cpython-asyncio-doctest.md` is a deeper reference for async
doctest patterns — read it before inventing one, but remember its examples do
not execute (see above); treat them as a checked-by-eye pattern library, not a
test suite.

**Docstring examples** use the NumPy `Examples` section:

    Examples
    --------
    >>> stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    True

## Docstrings

The prime directive: never restate the type. The annotation is the source of
truth; the docstring carries what the annotation cannot.

Document the dimensions the type system cannot encode: what a call mutates,
what it owns, what order results come back in, what exceptions it raises and
when, whether calling it twice does anything the second time, what a boundary
value (zero, empty, the maximum) does.

**Classes with fields** — dataclasses, `NamedTuple` — document every field in
an `Attributes` section:

```python
@dataclass(frozen=True, slots=True)
class Args:
    """Parsed command-line arguments.

    Attributes
    ----------
    seed : int
        Random seed applied before the demo runs. Default 42.
    artifact_dir : Path
        Directory artifacts and logs are written to.
    """
```

A type says how a field is shaped, not what it holds. Describing each one
keeps that meaning next to the code, and anything that renders the class —
autodoc, a REPL, an editor tooltip — has a description to show instead of a
bare name.

**Every function that does real work states its complexity** in a
`Complexity` section, using Big-O for both dimensions:

```python
def solve_pain(data: Sequence[int]) -> int:
    """Solve using only stdlib.

    Complexity
    ----------
    Time
        O(n) — single pass.
    Space
        O(1) — constant.
    """
```

The first sentence stands alone; tooling truncates there. PEP 257 applies:
triple double quotes, an imperative one-line summary ending in a period, a
blank line before any extended description.

One docstring dialect per repository (NumPy), enforced by ruff's `pydocstyle`
rule rather than relitigated in review.

## Source comments

A comment ships only if it passes all three gates. Fail any: delete or
rewrite. Borderline: delete — borderline means the information is
reconstructible, which is what makes deletion cheap.

**Loss.** Three years from now, would losing this cost a future reader real
time rediscovering intent, an invariant, or a failure mode the code and tests
do not already make obvious?

**Elite.** Would SQLite, Redis, the Go standard library, or CPython write this
comment, at this length? Those projects state the constraint and stop. They do
not argue with an imagined objector.

**Upkeep.** Will it stay true without maintenance? A comment that hand-syncs a
value the code owns — a count, an offset, a duplicated constant — is false the
first time that value moves.

### Ceiling

One or two lines. A comment reaching four is either carrying several facts, in
which case split it, or arguing, in which case cut it to the fact.

Rationale, alternatives weighed, and the story of how the code got here belong
in the commit message: timestamped, attached to the exact diff, and free to
maintain.

### Keep

- Why over how: stdlib quirks, platform constraints, tradeoffs still part of
  the contract.
- Invariants, preconditions, ordering, and lifetime requirements that types
  and tests cannot express.
- Code that looks wrong but is not, so a later cleanup does not reintroduce
  the bug.
- A high-level sketch of an algorithm whose local operations do not reveal
  the whole.

### Delete

- Narration of the next lines; code translated into English.
- Restated names, types, defaults, or control flow.
- Values duplicated from the code and hand-synced.
- Justification, hedging, or apology for a choice.
- History version control already holds, including commented-out code.
- Ticket and issue numbers. They say nothing to a reader without tracker
  access, and they rot when the tracker moves. Unfinished work goes in the
  tracker, not the source.
- Transient observations — "currently", "for now" — that go stale with no
  nearby edit.

### The upkeep gate in practice

It reaches values that track our own code. It does not reach frozen external
facts.

Bad (Delete):

```python
# There are 321 tests to complete for servers.
```

Good (Keep):

```python
# CPython < 3.11 has no ExceptionGroup, so this branch stays.
```

### Documentation exception

Doctests, minimal usage examples, and NumPy-style `Parameters`, `Returns`,
`Attributes`, and `Complexity` sections on public functions are exempt from
the loss gate — they serve the reader, not the maintainer. They are exempt
from nothing else. Ceiling: a good man page entry.

## Terminology and capitalization

Pick the domain noun and keep it. This curriculum's own vocabulary is
`lesson`, `pain` (the stdlib version), and `power-up` (the library version) —
use those words consistently rather than "exercise", "baseline", or
"optimized version". If the code calls a function `solve_pain`, call it "the
pain implementation" in prose, not "the naive version" in one place and "the
slow path" in another.

Stable vocabulary is what makes search and an agent's retrieval work at all.

Python and PyPI keep their own capitalization. Do not write counts into prose
— how many lessons exist, how many doctests there are. They go stale silently
and no reader needs them.

## Markdown

Prose wraps at 80 columns. Table rows, badge lines, and long links are exempt,
because breaking them harms rendering.

GitHub alert blocks — `> [!NOTE]`, `> [!WARNING]` — render as literal text
outside GitHub, so reserve them for at most one load-bearing warning per
document.

Do not use a local absolute path or an email address in anything published.

## Code blocks

Code blocks are paste-and-run units: pasting one block runs exactly one
intended action. Doctests and other executed examples are exempt — the test
suite runs them, nobody pastes them.

- **One command per block.** Multiple steps may share a block only when
  explicitly chained with `&&`, `;`, or `\` continuations — the chain is then
  one logical command.
- **Explanations go in prose above the block**, never as `#` comments inside
  it.
- **Command menus are per-command blocks with prose lead-ins**, not tables.
- **Shell commands use the `console` tag with a `$ ` prefix.** This separates
  interactive commands from scripts and enables prompt-aware copy.
- **Split long commands with `\`** — one flag or flag+value pair per indented
  continuation line, positional arguments last.

Good — run a single lesson's doctests verbosely:

```console
$ uv run pytest \
    -v \
    src/001_intro.py
```

Bad:

```console
# Run a single lesson's doctests verbosely
$ uv run pytest src/001_intro.py -v
```

## Commits

```
Scope(type[detail]): concise description

why: Explanation of necessity or impact.

what:
- Specific technical changes made
- Focused on a single topic
```

Keep the subject to 50 characters or fewer, imperative mood ("Add", "Fix" —
not "Added", "Fixed"), and wrap body lines at 72. Separate the `why:` and
`what:` blocks with a blank line. Mark a breaking change — a change to a
lesson's public function signature or output — with `BREAKING:` in the body.

Subjects are plain English. Never put curriculum codes ("0.3", "Tier 2") or
other repo-internal shorthand in the subject line — a reader of
`git log --oneline` should understand every title cold.

Routine maintenance commits drop the colon and take a capitalized
description, which is what distinguishes them at a glance:

```
py(deps[dev]) Bump dev packages
ai(rules[AGENTS]) Judge comments by three gates
```

Everything that changes behavior keeps the colon.

Common types:

- **feat**: New features or enhancements
- **fix**: Bug fixes
- **refactor**: Code restructuring without functional change
- **docs**: Documentation updates
- **chore**: Maintenance (dependencies, tooling, config)
- **test**: Test-related updates
- **style**: Code style and formatting
- **py(deps)**, **py(deps[dev])**: Dependencies, dev dependencies
- **ai(rules[AGENTS])**: AI rule updates
- **ai(claude[rules])**: Claude Code rules (`CLAUDE.md`)

Example:

```
docs(README[setup]): Document the uv-based install

why: New clones failed on the old pip instructions.

what:
- Replace pip commands with uv equivalents
- Note the supported Python floor
```

For a multi-line message, use a heredoc so the formatting survives:

```console
$ git commit -m "$(cat <<'EOF'
Scope(feat[detail]): Concise description

why: Explanation of the change.

what:
- First change
- Second change
EOF
)"
```

## Slop prevention

Treat AI slop as review-hostile noise, not as proof that text or code is
wrong. The goal is to maximize information density.

- **AI signatures.** No "Generated by", no conversational filler, no
  unexplained emoji, no tool metadata.
- **Brittle references.** No hard-coded line numbers, fragile file/lesson
  counts, dated "as of" claims, bare SHAs, or local absolute paths — unless
  they are strict evidentiary artifacts, such as an `experiments.jsonl` log
  entry.
- **Diff narration.** Do not restate what moved, was renamed, or was removed
  in anything the reader holds alongside the diff: code, docstrings, README,
  or a pull request description. The diff and commit message already carry
  it.
- **Branch-internal narrative.** Do not mention intermediate states,
  abandoned approaches, or "no longer" behavior in a docstring, README, or
  comment — that belongs in the commit message, not the artifact.
- **Low-value scaffolding.** No ownerless TODOs, unused future-proofing,
  debug artifacts, or defensive wrappers around failure modes nothing can
  reach.
- **Prose inflation.** The diction table under [Voice](#voice) governs;
  replace an inflated word with a concrete description of behavior,
  constraints, or trade-offs.
- **Coded labels.** Write rules and findings as plain imperatives. No `[R1]`,
  `Option B`, or any index a reader has to decode.

**Durable source links.** When a lesson or note links to an upstream
implementation — `micrograd`, `nanoGPT`, CPython's own source — link to a
pinned tag or a 7-character commit SHA reachable from that project's trunk,
not to its `main` branch and never to a pull-request head. A `blob/main/…`
link rots silently: the file moves, the line shifts, and the anchor lands on
unrelated code while still resolving. Line anchors (`#L120-L145`) are only
safe on a pinned ref.

Preserve the "why". Never delete a comment documenting an invariant, a
platform quirk, or an upstream workaround — those are the facts
[Source comments](#source-comments) keeps, and every other comment is judged
by it.
