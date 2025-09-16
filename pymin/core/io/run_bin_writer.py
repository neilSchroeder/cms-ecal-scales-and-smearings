"""Run bin writer service for CMS ECAL calibration pipeline."""

import logging
from pathlib import Path
from typing import List, Tuple


class RunBinWriter:
    """Service for writing run bins in legacy format for pipeline compatibility."""

    def __init__(self, output_dir: Path, logger: logging.Logger = None):
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_run_bins(self, run_bins: List[Tuple[int, int]], output_tag: str) -> Path:
        """Write run bins to .dat file in legacy format.

        Parameters
        ----------
        run_bins : List[Tuple[int, int]]
            List of (run_low, run_high) tuples
        output_tag : str
            Output tag for file naming

        Returns
        -------
        Path
            Path to written file
        """
        # Generate output filename (legacy format)
        output_file = self.output_dir / f"run_divide_{output_tag}.dat"

        self.logger.info("Writing %d run bins to: %s", len(run_bins), output_file)

        with open(output_file, "w") as f:
            # Write header (legacy format for time stability step)
            f.write("runNumber\trunNumber\n")

            # Write run bins
            for run_low, run_high in run_bins:
                f.write(f"{run_low}\t{run_high}\n")

        self.logger.info("Successfully wrote run bins file")
        return output_file

    def write_summary_stats(
        self, run_bins: List[Tuple[int, int]], output_tag: str, total_events: int
    ) -> Path:
        """Write summary statistics for validation.

        Parameters
        ----------
        run_bins : List[Tuple[int, int]]
            Run bins created
        output_tag : str
            Output tag for naming
        total_events : int
            Total events processed

        Returns
        -------
        Path
            Path to summary file
        """
        summary_file = self.output_dir / f"run_divide_summary_{output_tag}.txt"

        with open(summary_file, "w") as f:
            f.write(f"Run Division Summary - {output_tag}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total run bins created: {len(run_bins)}\n")
            f.write(f"Total events processed: {total_events}\n")
            f.write(f"Average events per bin: {total_events / len(run_bins):.1f}\n")
            f.write("\nRun bin details:\n")

            for i, (run_low, run_high) in enumerate(run_bins, 1):
                if run_low == run_high:
                    f.write(f"Bin {i:3d}: Run {run_low}\n")
                else:
                    f.write(f"Bin {i:3d}: Runs {run_low} - {run_high}\n")

        return summary_file
