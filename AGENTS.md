# Repository Guidelines

## Project Structure & Module Organization
- `main.py` contains the current entry point and example runtime logic.
- `pyproject.toml` defines project metadata and Python version requirements.
- `README.md` is present but currently empty; update it when adding user-facing docs.
- No separate `src/`, `tests/`, or asset directories exist yet.

## Build, Test, and Development Commands
- `python main.py` runs the default entry point.
- `python -m pip install -e .` installs the project in editable mode (useful once dependencies are added).
- There are no build or test scripts configured yet; add them in `pyproject.toml` if needed.

## Coding Style & Naming Conventions
- Follow standard Python style (PEP 8) with 4-space indentation.
- Use `snake_case` for functions and variables, `PascalCase` for classes.
- Prefer explicit, descriptive names (e.g., `sort_inputs`, `transformer_block`).
- No formatter or linter is configured; if one is introduced, document the command here.

## Testing Guidelines
- No test framework or tests are present.
- If tests are added, use a `tests/` directory and name files `test_*.py` (e.g., `tests/test_main.py`).
- Test runner: `python -m unittest`.

## Commit & Pull Request Guidelines
- There is no Git commit history yet, so no established commit message convention.
- Until a convention is adopted, keep commit messages short and imperative (e.g., "Add data loader").
- PRs should include a clear description of changes and any relevant usage notes.

## Security & Configuration Tips
- This repository has no secrets or config files; do not add credentials directly to source.
- If configuration is introduced, prefer environment variables and document defaults.
