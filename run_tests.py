#!/usr/local/bin/python3
from python.tests.test_write_smearings import test_wf_write_smearings
from python.tests.test_ss_config import test_ss_config
from python.tests.test_plot_run_stability import test_plot_run_stability
from python.tests.test_helper_pymin_functions import (
    test_write_results,
    test_get_step,
)


def main():

    # test_wf_write_smearings()
    # test_ss_config()
    # test_plot_run_stability()
    test_write_results()
    # test_get_step()


if __name__ == '__main__':
    main()