# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python package for privacy-preserving VLM embedding perturbation. Core mechanisms live in `privacy/` (`vmf.py`, `gaussian.py`, `laplace.py`, `norm_preserving.py`). Model integration code lives in `models/`, with `qwenvl_wrapper.py` as the main wrapper. Use `experiments/` for comparison scripts, `examples/` for runnable demos, `scripts/` for plotting and validation utilities, and `asset/` for generated figures. Keep new modules focused and place them next to related mechanisms or experiments.

## Build, Test, and Development Commands
Use `uv` for environment and dependency management.

- `uv sync`: install and lock project dependencies from `pyproject.toml` and `uv.lock`.
- `uv run experiments/compare_mechanisms.py`: run algorithm-level comparisons across privacy mechanisms.
- `uv run experiments/qwenvl_comparison.py --epsilon 0.5`: run the QwenVL comparison with an explicit privacy budget.
- `uv run examples/qwenvl_demo.py`: execute the end-to-end usage example.
- `uv run scripts/validate_vmf_approximation.py`: validate the vMF approximation numerically.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, type hints where they improve clarity, and short docstrings for public classes and methods. Use `snake_case` for functions, variables, and module names; use `PascalCase` for classes such as `QwenVLPrivacyWrapper`. Keep mechanism names aligned with current string identifiers: `'vmf'`, `'gaussian'`, `'laplace'`, and `'norm_preserving'`. Prefer small, explicit NumPy/Torch steps over dense one-liners.

## Testing Guidelines
There is no dedicated `tests/` directory yet, so treat runnable validation scripts as the current test surface. Before opening a PR, run the relevant `uv run` command for the module you changed and confirm outputs are sensible. For new tests, prefer `tests/test_<module>.py` and cover numerical edge cases such as zero vectors, shape reshaping, and dtype/device preservation.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects, often with Conventional Commit prefixes such as `fix(privacy): ...`. Keep commits scoped to one logical change. PRs should include a concise summary, affected modules, commands run for verification, and screenshots only when plots or assets changed. Link the related issue or experiment note when applicable.

## Configuration Notes
Target Python `>=3.12` as declared in `pyproject.toml`. Large model downloads are external to this repo, so avoid committing cached weights, generated checkpoints, or local notebooks outputs.
