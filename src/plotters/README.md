# Directory: cms-ecal-scales-and-smearings/python/plotters

## File Descriptions

### [make_plots.py](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/plotters/make_plots.py?ref_type=heads)
Description: This file contains the code used to call the plots from a script. For example, if you wanted to plot some data you would 
make the following calls:

```
from python.plots.make_plots import plot

def plot_some_data(data, mc, categories, **options):
    """
    Plot some data using the plot function from python.plots.make_plots.

    Args:
        data (pd.DataFrame): the data
        mc (pd.DataFrame): the MC
        categories (str): path to the tsv file that defines the plotting categories
        **options (dict): additional options you'd like to pass.
    Returns:
        None
    """

    return plot(data, mc, categories, **options)

plot_some_data(data, mc, categories, **options)
```

### [plots.py](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/plotters/plots.py?ref_type=heads)
Description: This file contains the code for plotting different types of plots including "validation" style and "paper" style plots.
If you want to add new plot styles, you'll put the python code for them here. Be sure to add your new plot style to the [PlottingConstants](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/classes/constant_classes.py?ref_type=heads) class in `python.classes.constant_classes`.

### [plot_run_stability.py](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/plotters/plot_run_stability.py?ref_type=heads)
Description: This file contains the code to plot the run_stability plots.
TODO: put this code into [plots.py](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/plotters/plots.py?ref_type=heads) to condense

### [fit_bw_cb.py](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/plotters/fit_bw_cb.py?ref_type=heads)
Description: This file contains code to fit and plot a breit wigner convoluted crystal ball distribution to data.
TODO: put this code into [plots.py](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/plotters/plots.py?ref_type=heads) to condense

### [plot_cats.py](/https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/plotters/plot_cats.py?ref_type=heads)
Description: This file contains some old code for plotting invariant mass distributions in Z Categories. It is deprecated. Use with caution.
