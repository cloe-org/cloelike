# AGENTS.md

Instructions for AI coding agents working in `cloelike`. Humans: see [CONTRIBUTING.md](CONTRIBUTING.md) and the [PR template](.github/pull_request_template.md).

## Project

`cloelike` is the likelihood module of the CLOE cosmological analysis for Euclid: it combines theory predictions from `cloelib` (and `cosmolib`) with data vectors, covariances and mixing matrices to compute the log-likelihood of the Euclid primary probes (photometric 3x2pt, COSEBIs, spectroscopic galaxy clustering and BAO). It is used by `cloerun` and by the `playground` notebooks, so public class names, settings keys and parameter names are relied on downstream.

Where to look first (do not guess, read the code and docs):

- `cloelike/`: one module per likelihood. `EuclidLikelihood_photo_base.py` holds the shared photometric logic (intrinsic-alignment and galaxy-bias models, nuisance keys); read it before editing `EuclidLikelihood_photo_Cls.py`, `EuclidLikelihood_photo_COSEBIs.py` or `EuclidLikelihood_photo_2pcf.py`. `EuclidLikelihood_GCspectro_*.py` are the spectroscopic likelihoods; `bias_relations.py` the bias relations. Public classes are listed in `cloelike/__init__.py`.
- Theory predictions live in `cloelib`: if a fix belongs there, say so instead of working around it in `cloelike`.
- `tests/`: pytest suite. `docs/` + `mkdocs.yml`: documentation, API pages in `docs/api/`.
- Installation and dependencies: [README.md](README.md), [docs/getting_started/installation.md](docs/getting_started/installation.md) and `pyproject.toml`.

## Commands

Run from the repository root, inside the `cloe-org-env` environment (see the installation page).

```bash
pre-commit run --all-files --show-diff-on-failure   # ruff, prettier, file hygiene
python -m pytest tests/test_<file>.py               # targeted tests
python -m pytest tests/test_<file>.py::test_<name>  # single test
```

CI runs style, then tests, then build, on Python 3.12 with the backends listed in `.github/workflows/ci.yml`. GitHub CI is the final authority; local runs are supporting evidence.

## Tests and optional backends

- Run the tests of the files you changed, plus their direct dependents. Do not run the full `python -m pytest` suite by default.
- **Ask the human before running tests that need a heavy backend or download data.** Most test files use CAMB, HMcode2020Emu, COMET or `euclidlib`, and download data vectors and covariances from Zenodo; they are slow and need network access. `tests/test_bias_relations.py` needs neither.
- If a backend is not installed or a test was not run, say so explicitly in your final report. Never present unrun checks as passing.
- Add or update tests for every behaviour change; for bug fixes add a regression test. Use explicit tolerances (`rtol`, `atol`).
- Never commit downloaded data files to the repository.

## Code conventions

Formatting and imports are enforced by `ruff` and `prettier`; do not restate or fight them. What the linters do not catch:

- **Settings and parameters**: never rename or silently change settings keys, nuisance-parameter names, units or defaults; `cloerun` configurations and `playground` notebooks depend on them.
- **Consistency across likelihoods**: shared behaviour belongs in the base module, not copied into each likelihood class. If you change one likelihood, check whether its siblings need the same change.
- **Readable over clever**: prefer a few extra clear lines to compact or niche constructs. Maintainers must be able to follow every line.
- **Minimal diffs**: change only what the task requires. Do not rewrite, rename, reformat or reorganise code you were not asked to touch, and do not remove support for optional models or backends unless asked.
- **Docstrings**: NumPy style, short, describing parameters, returns and units. Do not write long docstrings that hide the useful information.
- **Public API**: if you change it, update the docstrings, `cloelike/__init__.py` and the pages under `docs/api/`.
- Do not add stray files (`NOTES.md`, `PLAN.md`, `SUMMARY.md`, scratch scripts), data files, binaries or commented-out code to the repository.

## Git and pull-request workflow

- One branch per issue, branched from `main`. The PR title must be `gh-<issue number>: <description>` (CI check `PR Title`).
- **Never `git push`, and never commit or stage unless the human explicitly asks.** When asked, stage named files only and show what will be committed first.
- Follow the [PR template](.github/pull_request_template.md). **Never tick the PR-template checkboxes on the human's behalf** (in particular "I have reviewed and understand ..."), and do not write the AI-assistance answer as if no AI was used. If asked to draft a PR description, leave those for the human and say that they must be completed. If asked to draft the AI Assistance section, state your harness and exact model name/version.
- Remind the human that they are responsible for this PR: they must review every changed line, understand it, and explain in their own words in the PR summary what it does and why. The `PR Description` CI check reports which required items are missing.
- Design questions or API changes that go beyond the issue: stop and ask the human to discuss with @cloe-org/cloe-maintainers first, as described in [CONTRIBUTING.md](CONTRIBUTING.md).
- **Do not edit this file** (`AGENTS.md`), unless the human tells you they are a cloelike maintainer and asks for it. Only @cloe-org/cloe-maintainers change it. If it looks outdated or wrong for the task at hand, tell the human so they can report it to the maintainers instead of changing it yourself.

## Limits of this file

These instructions are advisory. Rules that can be enforced mechanically belong in tooling (pre-commit, CI), not here. Keep this file short and only add what changes agent behaviour.
