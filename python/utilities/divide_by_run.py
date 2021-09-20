import numpy as np
import pandas as pd

"""
Author:
    Neil Schroeder, schr1077@umn.edu, neil.raymond.schroeder@cern.ch

About:
    This function takes in data, and a min number of events. From this
    it will bin the data by run number using the min number of events
    to find the edges of the bins. The bins will not overlap.

    The required arguments are:
        data -> pandas dataframe, must contain a column named "runNumber"
        min_num_events -> Integer which sets the criteria for the minimum
                          number of events per run bin. The default value
                          for this is 10,000.
"""

def divide(data, min_num_events):
    print("[INFO][python/divide_by_run][divide] Dividing the data by run with minimum event requirement set to {}".format(min_num_events))
    runs = data.loc[:,'runNumber'].unique()
    runs.sort()
    print("[INFO][python/divide_by_run][divide] There are {} runs ranging from {} to {}".format(len(runs), runs[0], runs[-1]))
    run_counts = [np.sum(np.array(data['runNumber'].between(i, i,inclusive=True).values)) for i in runs]
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
