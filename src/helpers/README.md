# Helpers

This directory contains helper files for the framework.

## `helper_minimzer.py`

Contains many of the functions that were valuable to abstract away from `python/utilities/minimizer.py` for clarity.

| Function | Description |
| --- | --- |
| `add_transverse_energy` | Adds the transverse energy to the data. |
| `get_smearing_index` | Gets the smearing index (from the config/cats file) for a given event. |
| `clean_up` | Cleans up the dataframes, calls `add_transverse_energy`, if necessary and drops the columns that are not needed for the minimization. |
| `extract_cats` | Extracts the zcat categories defined the config/cats file. |
| `set_bounds` | Sets the bounds for the minimization. |
| `deactivate_cats` | Deactivates the categories that are not needed for the minimization or that meet deactivation criteria. |
| `target_function` | The target function for the minimization. |
| `scan_nll` | Scans the NLL to determine the starting values for the scales and smearings. |
| --- | --- |

## `helper_plots.py`

Contains many of the functions that were valuable to abstract away from `python/plotters/plots.py` and `python/plotters/make_plots.py` for clarity.

## `helper_pymin.py`

Contains many of the functions that were valuable to abstract away from `pymin.py` for clarity.

## `helper_pyval.py`

Contains many of the functions that were valuable to abstract away from `pyval.py` for clarity.