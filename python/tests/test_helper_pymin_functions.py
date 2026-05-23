from collections import namedtuple
import sys
from unittest.mock import patch
from python.helpers.helper_pymin import (
    write_results,
    get_step,
    get_cmd,
    get_options,
)


def test_write_results():
    """Test the write_results function from helper_pymin.py"""

    class MyArgs:
        def __init__(
            self, scales_file, cats_file, time_stability, from_condor, closure, output
        ):
            self.scales = scales_file
            self.catsFile = cats_file
            self._kTimeStability = time_stability
            self._kFromCondor = from_condor
            self._kClosure = closure
            self.output = output

    write_results(
        MyArgs(
            "datFiles/step1_ul18_test_scales.dat",
            "config/cats_step2.tsv",
            False,
            False,
            False,
            "test",
        ),
        [1] * 8 + [0.02] * 8,
    )


def test_get_step():
    """Test the get_step function from helper_pymin.py"""

    class MyArgs:
        def __init__(self, dat_file, time_stability):
            self.catsFile = dat_file
            self._kTimeStability = time_stability

    args = MyArgs("datFiles/cats_step3.dat", False)
    assert get_step(args) == "3"

    args = MyArgs("datFiles/cats_step2.dat", False)
    assert get_step(args) == "2"

    args = MyArgs("datFiles/cats_step3_timeStability.dat", True)
    assert get_step(args) == -1

    args = MyArgs(None, False)
    assert get_step(args) == -1

    args = MyArgs(None, True)
    assert get_step(args) == -1


def test_get_cmd():
    """Test get_cmd returns a space-separated command string."""
    original_argv = sys.argv
    try:
        sys.argv = ["pymin.py", "--scales", "file.dat"]
        cmd = get_cmd()
        assert "pymin.py" in cmd
        assert "--scales" in cmd
        assert "file.dat" in cmd
    finally:
        sys.argv = original_argv


def test_get_cmd_with_spaces():
    """Test get_cmd quotes arguments containing spaces."""
    original_argv = sys.argv
    try:
        sys.argv = ["pymin.py", "file with spaces.dat"]
        cmd = get_cmd()
        assert '"file with spaces.dat"' in cmd
    finally:
        sys.argv = original_argv
