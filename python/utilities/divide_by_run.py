import numpy as np
import pandas as pd

"""
Author:
    Neil Schroeder, schr1077@umn.edu, neil.raymond.schroeder@cern.ch
"""

def divide(data, min_num_events):
    """
    Divide the data by run number using the min_num_events to find the edges
    of the bins. The bins will not overlap.
    ----------
    Args:
        data: pandas dataframe, must contain a column named "runNumber"
        min_num_events: Integer which sets the criteria for the minimum
                        number of events per run bin. The default value
                        for this is 10,000.
    ----------
    Returns:
        bins: list of tuples, each tuple contains the lower and upper edges
                of the run bins.
    ----------
    """
    INFO = "[INFO][python/divide_by_run][divide]"
    print(f"{INFO} Dividing the data by run with minimum event requirement set to {min_num_events}")
    runs = data.loc[:,'runNumber'].unique()
    runs.sort()
    print(f"{INFO} There are {len(runs)} runs ranging from {runs[0]} to {runs[-1]}")
    run_counts = [np.sum(np.array(data['runNumber'].between(i, i, inclusive='both').values)) for i in runs]
    bins = []
    i = int(0)
    while i < len(runs):
        if run_counts[i] > min_num_events:
            bins.append( (runs[i], runs[i]) )
            i+=1
        else:
            count = run_counts[i]
            high_edge = i
            while count < min_num_events:
                if high_edge < len(runs)-1:
                    high_edge += 1
                    count += run_counts[high_edge]
                else:
                    high_edge = len(runs)-1
                    break
            bins.append( (runs[i], runs[high_edge]) )
            i = high_edge + 1

    return bins
