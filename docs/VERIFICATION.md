# Verification — 21 November 2026

Python 3.12 on Linux; editable package installed with `[test,cq]` extras after `git submodule update --init --recursive`.

- `python -m pytest tests -q`: **147 passed**, one host pytest/AnyIO deprecation warning.
- `python validation/verify_meanline.py`: **41/41 checks passed**. These independently implement thermodynamics, velocity-triangle and continuity equations; they are not experimental validation.

Added `tests/__init__.py` to prevent an installed top-level `tests` package from shadowing local fixtures. Before the fix, six test modules failed collection with `ModuleNotFoundError: tests.designs`. No mean-line equations were changed.

`requirements.txt` delegates runtime requirements to pyproject.toml. Optional CAD dependencies remain `.[cq]`; tests require `.[test]`. Existing upstream/submodule attribution remains intact.

The 90–150 N class is a target application, not an engine thrust prediction. No 3D RANS, full-engine cycle coupling, experimental validation or Windows/8 GB performance benchmark was executed.
