import numpy as np
import pandas as pd

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

    print(bins)
    return bins
