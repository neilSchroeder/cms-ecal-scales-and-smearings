# AGENTS.md

Guidance for AI coding agents working in the **cms-ecal-scales-and-smearings** repo
(derivation + validation of residual energy scales and smearings for the CMS ECAL).

See [README.md](README.md) for the full physics workflow and
[README_pyval.md](README_pyval.md) for the validation tool.

## Entry points

| Script | Purpose |
| --- | --- |
| [pymin.py](pymin.py) | Derive scales/smearings: prune → run-divide → time-stability → minimize. CLI args documented in `main()` docstring. |
| [pyval.py](pyval.py) | Produce data/MC validation plots after applying scales/smearings. |
| [run_tests.py](run_tests.py) | Ad-hoc test runner (mostly commented out). **Prefer pytest** — see Testing. |

Console scripts are also registered in [pyproject.toml](pyproject.toml): `pymin`, `pyval`.

## Environment & commands

- Python **3.10–3.12** (see [pyproject.toml](pyproject.toml)). A `.venv/` and `uv.lock` are present — use `uv` or activate `.venv` directly. The legacy `env.yml` conda env (`scales-env`) still works.
- Install (dev): `uv sync` (or `pip install -e . && pip install pytest pytest-cov`).
- Run tests: `pytest` (configured `testpaths = ["python/tests"]`).
- Run with coverage: `pytest --cov=python`.
- Single test: `pytest python/tests/test_zcat.py -k name`.

## Architecture (read these READMEs, don't duplicate them)

The `python/` package is the entire codebase; entry-point scripts at repo root import from it.

| Area | Purpose | Reference |
| --- | --- | --- |
| [python/utilities/](python/utilities/) | Pipeline stages: pruner, divide_by_run, time_stability, minimizer, scale_data, smear_mc, reweight_pt_y, write_files, condor_handler, data_loader, numba_hist, adamw_minimizer. | [python/utilities/README.md](python/utilities/README.md) |
| [python/classes/](python/classes/) | `SSConfig` (singleton, paths), `MinimizationConfig` (typed config dataclass), `ZCat` (di-electron category), `PyValConstants`/`DataConstants`/`CategoryConstants`/`PlottingConstants`, Breit-Wigner & Crystal Ball PDFs. | [python/classes/README.md](python/classes/README.md) |
| [python/helpers/](python/helpers/) | Functions abstracted out of `minimizer.py`, `pymin.py`, `pyval.py`, `plots.py`. | [python/helpers/README.md](python/helpers/README.md) |
| [python/plotters/](python/plotters/) | Plot styles registered via `PlottingConstants`. `plot_cats.py` is deprecated. | [python/plotters/README.md](python/plotters/README.md) |
| [python/tools/](python/tools/) | Standalone post-processing scripts (add uncertainties, combine steps, validate coverage). | [python/tools/README.md](python/tools/README.md) |
| [python/tests/](python/tests/) | Pytest suite. |
| [config/](config/) | Category `.tsv` files (`cats_step{2..5}*.tsv`) and systematics. Format documented in [README.md](README.md). |
| [scripts/](scripts/) | Reference shell invocations of `pymin.py` per era/step — good source for realistic CLI flag combinations. |
| [condor/](condor/) | Historical condor submission outputs (gitignored content). |

## Data + I/O conventions

- Inputs: ROOT files with a tree (default `selected`) and specific branches (`runNumber`, `R9Ele[3]`, `etaEle[3]`, `energy_ECAL_ele[3]`, `invMass_ECAL_ele`, `gainSeedSC[3]`, `eleID[3]`). Pruned to `.tsv`/`.csv` via `pruner.py`.
- Input file lists are TSVs with header `type \t treeName \t filePath` (`type` ∈ {`data`, `sim`}).
- Outputs land in `datFiles/` (scales/smearings/weights `.dat`/`.tsv`) and `workspace/pymin/{data,plots,condor}/` locally, or `/eos/home-<u>/<user>/pymin/...` when on lxplus. `SSConfig.is_on_eos` switches automatically.
- Scales/smearings file naming: `..._scales.dat`, `..._smearings.dat`, `..._onlystep_scales.dat`. The minimization step (step2/3/4/5) is inferred from the categories file name by `helper_pymin.get_step`.

## Project-specific conventions

- `SSConfig` is a **singleton** — never instantiate paths manually; call `SSConfig()` and use its `DEFAULT_*_PATH` attributes. Always call `ss_config.ensure_directories()` at entry points (not at import time).
- Argparse flags use `_kFoo` (camelCase with underscore prefix) for boolean workflow switches (`_kClosure`, `_kPrune`, `_kPlot`, `_kTestMethodAccuracy`, `_kScanNLL`, `_kFixScales`, `_kDebug`). Preserve this style when adding flags.
- Minimization options are bundled into the `MinimizationConfig` dataclass and threaded through `minimizer → helper_minimizer → data_loader → zcat`. Don't add new pipeline kwargs ad-hoc — extend `MinimizationConfig`.
- Hot paths (histogramming, NLL, smearing) are `@numba.njit`-decorated in [python/utilities/numba_hist.py](python/utilities/numba_hist.py) and [python/classes/zcat_class.py](python/classes/zcat_class.py). New code on these paths must be numba-friendly (no Python objects, typed arrays, no pandas inside jitted functions). When in doubt, keep the heavy loop in a separate `@njit` helper.
- Category TSVs are tab-separated with the columns shown in [README.md](README.md); rows are `scale` or `smear`. Coverage in eta/R9/Et must be complete — use `python/tools/scales_validator.py`.
- Hard-coded physics constants live in [python/classes/constant_classes.py](python/classes/constant_classes.py) — change there first.
- Run plots/styles: register new plot styles in `PlottingConstants` rather than branching in `plots.py`.

## Pitfalls

- Don't run code at import time that creates directories or reads config — `SSConfig.ensure_directories()` is intentionally explicit.
- `run_tests.py` is **not** the test entry point; most calls are commented out. Use `pytest`.
- The `python/` package is exposed as a top-level package (`packages = ["python"]` in `pyproject.toml`); imports look like `from python.utilities import ...` — keep them absolute, not relative.
- Some files are intentionally excluded from coverage (see `[tool.coverage.run]` `omit` list); these are largely plotting/condor/tools scripts. Don't expect them to be exercised by tests.
- `--condor` submission requires lxplus + a valid `--queue`; locally, drop the flag.
- numba's first call JIT-compiles — initial test runs are slow; this is expected.

## When making changes

- Touch the minimum needed. Prefer editing existing files over adding new ones.
- Match existing style (snake_case functions, docstrings with `Args/Returns` dashed blocks like in `pymin.py:main`).
- If you add a CLI flag, also wire it through `helper_pymin.get_options()` → `MinimizationConfig`.
- Add or extend a pytest in `python/tests/` for new behavior; mirror existing naming (`test_<module>.py`).
