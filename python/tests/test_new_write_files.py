import sys

import numpy as np
import pandas as pd

# set path two directories up
sys.path[0] = "/".join(sys.path[0].split("/")[:-2])

from python.utilities import write_files as wf

# Test the function
wf.combine_scale_steps(
    "datFiles/test_write_coarse_scales.dat",
    "datFiles/test_write_fine_scales.dat",
    "datFiles/output_test_write_coarse_fine_scales.dat",
)

wf.combine_scale_steps(
    "datFiles/test_write_fine_scales.dat",
    "datFiles/test_write_coarse_scales.dat",
    "datFiles/output_test_write_fine_coarse_scales.dat",
)

wf.combine_scale_steps(
    "~/test_step5.dat",
    "~/test_step6.dat",
    "~/test_step5_6.dat",
)
