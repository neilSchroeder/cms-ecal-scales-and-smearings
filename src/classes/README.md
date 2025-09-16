# Classes

## `config_class.py`

This is the class that is used to store the configuration of the framework. It's most important task is to load the proper 
folders to write outputs to. It is a singleton class, meaning that there can only be one instance of it. This is to ensure that multiple instances of folders aren't created.

## `constant_classes.py`

This file does a lot of the heavy lifting in terms of hard coded stuff. If you need to change something in the code, this is mostly likely one of the first places you'll have to look. It contains the following classes:

| Class | Description |
| --- | --- |
| `PyValConstants` | Contains the constants that are used in the validation process. |
| `DataConstants` | Contains the constants that are used in the data processing processes. |
| `CategoryConstants` | Contains the constants that are used in the category files. |
| `PlottingConstants` | Contains the constants that are used in the plotting processes. |

## `zcat_class.py`

Arguably the most important of the classes here. This class is used to store the dielectron category (zcat) information. This class is manipulated mostly by `python/utilities/minimizer.py` and (by extension) `python/helpers/helper_minimizer.py` 