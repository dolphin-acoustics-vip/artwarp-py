# Contributing to artwarp-py

Thanks for helping improve ARTwarp-py. This page covers how we branch, open PRs, and write commits.

## Branch model

```
feature/*  ──PR──►  dev  ──semester release──►  main  ──►  PyPI + Zenodo
```

| Branch | Role |
|--------|------|
| **`main`** | Stable releases only. What we publish to PyPI and Zenodo. |
| **`dev`** | Integration branch. Feature work lands here via PRs. |
| **`feature/*`** | Active work (e.g. `feature/gui`, `feature/ocean-integration`). |

**Flow**

1. Branch from **`dev`** (or from `main` only if `dev` has not diverged yet).
2. Open a **PR into `dev`**. Review and merge there.
3. About once a semester, open a **PR from `dev` → `main`** for a stable release.
4. After `main` is updated: tag the release, publish to **PyPI**, and let **Zenodo** mint/update the DOI from the GitHub Release.

Do **not** open feature PRs directly into `main` unless it is an urgent hotfix agreed with maintainers.

## Current feature branches

- `feature/ocean-integration` — OCEANS / contour-pipeline integration
- `feature/gui` — GUI work

## Commit message format

```
BRANCH [CATEGORY] Short imperative description
 - TODO: follow-up task          # omit if none
 - NOTE: important detail        # omit if none
```

- **BRANCH** — uppercase form of the branch you are on  
  Examples: `MAIN`, `DEV`, `FEATURE-GUI`, `OCEAN-INTEGRATION`
- **CATEGORY** — one of:  
  `INIT` · `FEATURE` · `FIX` · `DOCS` · `REFACTOR` · `CHORE` · `TEST` · `STYLE` · `WIKI`
- **Description** — short, imperative, no trailing period  
  Good: `Add deterministic sample order`  
  Bad: `Added stuff` / `fix bugs.`

### Examples

```text
DEV [FIX] Correct JWT expiry handling in middleware
 - TODO: add integration test for token refresh

FEATURE-GUI [DOCS] Document the new plot panel shortcuts

OCEAN-INTEGRATION [FEATURE] Draft OCEANS wav-to-contour pipeline
 - NOTE: auth against production endpoints still unverified

MAIN [CHORE] Bump release metadata to v1.0.2 for Zenodo and PyPI
```

## Pull requests

- Target **`dev`** for feature work.
- Keep PRs focused; one concern per PR when practical.
- CI should be green (tests, coverage gate, lint) before merge.
- Update `CHANGELOG.md` when the change is user-visible.
- Bump version / PyPI / Zenodo **only** on the semester `dev` → `main` release (maintainers).

## Local setup (short)

```bash
git clone https://github.com/dolphin-acoustics-vip/artwarp-py.git
cd artwarp-py
git checkout dev
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

More detail: [docs/dev/ENVIRONMENT_SETUP.md](docs/dev/ENVIRONMENT_SETUP.md) and [docs/user/INSTALLATION.md](docs/user/INSTALLATION.md).

## Code style

- Format with **Black** and **isort**; lint with **Flake8**; type-check with **MyPy** (see CI).
- Prefer clear names and small functions over clever one-liners.
- Match the tone of nearby code (including existing `@author` / comment style).

## License

By contributing, you agree that your contributions are licensed under the same terms as the project (**LGPL-3.0-or-later**). See [LICENSE](LICENSE).
