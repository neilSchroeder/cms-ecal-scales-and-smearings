# Utilities

These are utilities that work together to make the framework run. They are heavily supported by the functions in `python/helpers/`.

| Utility | Description |
| === | === |
| `condor_handler.py` | Functions used when running `pymin.py` with the `--condor` option. Writes condor files and submits a job to condor. |
| --- | --- |
| `data_loader.py` | Functions for loading root files into dataframes and manipulating dataframes. |
| --- | --- |
| `divide_by_run.py` | Divides the data into run bins of at least 10,000 events. |
| --- | --- |
| `minimizer.py` | The main utility of the framework. This is the utility that is used to run the minimization process. |
| --- | --- |
| `numba_hist.py` | A histogram class that uses numba to speed up the filling process. |
| --- | --- |
| `pruner.py` | Prunes the data to only include the variables and entries that are needed for the minimization. |
| --- | --- |
| `reweight_pt_y.py` | Reweights the Pt(Z) and Y(Z) distributions in MC to match the data. |
| --- | --- |
| `scale_data.py` | Applies the scales to data. |
| --- | --- |
| `smear_mc.py` | Applies the smearings to MC. |
| --- | --- |
| `time_stability.py` | Calculates the time stability corrections for data. |
| --- | --- |
| `write_files.py` | Handles all the writing of files for scales and smearings. |
| --- | --- |

