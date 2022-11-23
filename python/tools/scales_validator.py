import pandas as pd
import argparse as ap

"""
takes in a scales file and checks that all eta, et, r9 are covered
"""

def main():

    parser = ap.ArgumentParser(description="validate scales file")

    parser.add_argument("-s", "--scalesFile", required=True,
                        help="Scales file to validate")

    args = parser.parse_args()

    scales = pd.read_csv(args.scalesFile, sep='\t', header=None)

    if not validate_scales(scales):
        print("[failed] consult earlier error and fix scales file")
    else:
        print("[success] scales file contains full coverage")


def validate_et(et_min, et_max):
    # validates that all Et is covered
    pass

def validate_r9(r9_min, r9_max):
    # validates that all r9 is covered (0 - 10)
    if len(r9_min) != len(r9_max):
        print("#"*40)
        print("[error] r9 lists not equal in length")
        print(r9_min)
        print(r9_max)
        return False

    r9_coverage = 10.

    coverage = sum([abs(round(r9_max[i],4) - round(r9_min[i],4)) for i in range(len(r9_max))])
    if coverage != r9_coverage:
        print("#"*40)
        print("[error] r9 coverage not complete")
        for i in range(len(r9_min)):
            print(r9_min[i], r9_max[i])
        return False

    return True


def validate_eta(eta_min, eta_max):
    # valides that all eta is covered (0 - 2.5)
    if len(eta_min) != len(eta_max):
        print("#"*40)
        print("[error] eta lists not equal in length") 
        return False

    eb_coverage = 1.4442
    ee_coverage = 2.5 - 1.566
    ee_ext_coverage = 2.5
    _kExtendedEE = False
    for i in range(len(eta_min)):
        if eta_min[i] < 1.4442:
            eb_coverage -= abs(eta_max[i] - eta_min[i])
        elif eta_min[i] >= 1.4442 and eta_min[i] < 2.5:
            ee_coverage -= abs(eta_max[i] - eta_min[i])
        else:
            _kExtendedEE = True
            ee_ext_coverage -= abs(eta_max[i] - eta_min[i])


    if eb_coverage > 0:
        print("#"*40)
        print("[error] coverage in EB not complete")
        for i in range(len(eta_min)):
            print(eta_min[i], eta_max[i])
        return False
    
    if ee_coverage > 0:
        print("#"*40)
        print("[error] coverage in EE not complete")
        for i in range(len(eta_min)):
            print(eta_min[i], eta_max[i])
        return False

    if _kExtendedEE and ee_ext_coverage > 0:
        print("[warning] coverage past 2.5 not complete")
        for i in range(len(eta_min)):
            print(eta_min[i], eta_max[i])

    return True


def validate_scales(df):
    #validates the coverage in each run category

    run_min_unique = df.loc[:,0].unique()

    for run in run_min_unique:
        print("[info] checking run bin starting with " + str(run))
        run_df = df[df.loc[:,0].values == run]
        eta_min_unique = run_df.loc[:,2].unique()
        eta_max_unique = run_df.loc[:,3].unique()

        if not validate_eta(eta_min_unique, eta_max_unique):
            print("validate eta failed")
            return False
        
        for eta_index, eta in enumerate(eta_min_unique):
            eta_run_df = run_df[run_df.loc[:,2] == eta]
            r9_min_unique = eta_run_df.loc[:,4].unique()
            r9_max_unique = eta_run_df.loc[:,5].unique()

            if not validate_r9(r9_min_unique, r9_max_unique):
                print("validate r9 failed in eta bin:")
                print(eta, eta_max_unique[eta_index])
                return False
    
            for r9_index, r9 in enumerate(r9_min_unique):
                r9_eta_run_df = eta_run_df[eta_run_df.loc[:,4] == r9]
                et_min_unique = r9_eta_run_df.loc[:,6].unique()
                et_max_unique = r9_eta_run_df.loc[:,7].unique()
                if len(et_min_unique) > 1:
                    if not validate_et(et_min_unique, et_max_unique):
                        print("validate et failed in eta, R9 bin:")
                        print(eta, eta_max_unique[eta_index])
                        print(r9, r9_max_unique[r9_index])
                        return False

    return True


if __name__ == "__main__":
    main()
