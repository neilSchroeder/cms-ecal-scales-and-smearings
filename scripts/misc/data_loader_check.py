from python.tools.data_loader import custom_cuts
from python.tools.data_loader import get_dataframe
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def main():

    check_eta_cuts = [
        ((0, 1), (0, 1.4442)),
        ((-1, 1), (-1, 1.4442)),
        ((0, 1), (-1, 1.4442)),
        ((-1, 1), (0, 1.4442)),
    ]
    check_r9_cuts = [
        ((0.96, 10.0), (0, 10)),
        ((0.96, -1), (0, 10)),
        ((0.96, 10), (-1, 10)),
        ((0.96, 10), (0, -1)),
        ((0.96, -1), (-1, 10)),
        ((0.96, -1), (0, -1)),
        ((0.96, 10), (-1, -1)),
        ((0.96, -1), (-1, -1)),
    ]
    check_et_cuts = [
        ((32, 14000), (20, 14000)),
        ((32, -1), (20, 14000)),
        ((32, 14000), (20, -1)),
        ((32, -1), (20, -1)),
    ]

    data_path = "examples/data/pruned_ul18_data.csv"
    df = get_dataframe([data_path])
    num_events = -1
    for eta_cut in check_eta_cuts:
        for r9_cut in check_r9_cuts:
            for et_cut in check_et_cuts:
                x = custom_cuts(
                    df,
                    inv_mass_cuts=(80, 100),
                    eta_cuts=eta_cut,
                    r9_cuts=r9_cut,
                    et_cuts=et_cut,
                )
                print(f"eta_cut: {eta_cut}")
                print(f"r9_cut: {r9_cut}")
                print(f"et_cut: {et_cut}")
                print(len(x))
                if num_events == -1:
                    num_events = len(x)
                else:
                    assert num_events == len(x)
                    num_events = len(x)
                print("=====================================")


if __name__ == "__main__":
    main()
