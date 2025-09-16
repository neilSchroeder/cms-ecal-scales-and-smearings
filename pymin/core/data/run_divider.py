"""Run division service using efficient Polars operations."""

import logging
from typing import List, Tuple
import polars as pl


class RunDivider:
    """Service for dividing CMS runs into bins with sufficient statistics."""

    def __init__(self, min_events: int = 10000, logger: logging.Logger = None):
        self.min_events = min_events
        self.logger = logger or logging.getLogger(__name__)

    def divide_runs(self, data: pl.DataFrame) -> List[Tuple[int, int]]:
        """Divide runs into bins with minimum event requirement.

        This preserves the exact numerical algorithm from the legacy
        divide_by_run.divide() function while using modern Polars operations.

        Parameters
        ----------
        data : pl.DataFrame
            Input data containing 'runNumber' column

        Returns
        -------
        List[Tuple[int, int]]
            List of (run_low, run_high) tuples defining run bins
        """
        self.logger.info(
            "Dividing data by run with minimum event requirement: %d", self.min_events
        )

        # Validate required column exists
        if "runNumber" not in data.columns:
            raise ValueError("Data must contain 'runNumber' column")

        # Get unique runs and sort them (preserving legacy behavior)
        runs = (
            data.select("runNumber").unique().sort("runNumber")["runNumber"].to_list()
        )

        self.logger.info("Found %d unique runs", len(runs))

        # Count events per run using efficient Polars groupby
        run_counts = (
            data.group_by("runNumber")
            .agg(pl.count("runNumber").alias("count"))
            .sort("runNumber")
        )

        # Convert to dictionary for fast lookup
        count_dict = dict(
            zip(run_counts["runNumber"].to_list(), run_counts["count"].to_list())
        )

        # Group runs to meet minimum event requirement (legacy algorithm)
        bins = []
        i = 0

        while i < len(runs):
            current_run = runs[i]
            current_count = count_dict[current_run]

            if current_count >= self.min_events:
                # Single run has enough events
                bins.append((current_run, current_run))
                self.logger.debug(
                    "Single run bin: %d (%d events)", current_run, current_count
                )
                i += 1
            else:
                # Combine multiple runs until threshold met
                combined_count = current_count
                high_edge = i

                while combined_count < self.min_events and high_edge < len(runs) - 1:
                    high_edge += 1
                    combined_count += count_dict[runs[high_edge]]

                bins.append((runs[i], runs[high_edge]))
                self.logger.debug(
                    "Multi-run bin: %d-%d (%d events)",
                    runs[i],
                    runs[high_edge],
                    combined_count,
                )

                i = high_edge + 1

        self.logger.info("Created %d run bins", len(bins))

        # Log statistics for validation
        total_events = sum(count_dict.values())
        self.logger.info("Total events: %d", total_events)

        return bins

    def validate_bins(self, bins: List[Tuple[int, int]], data: pl.DataFrame) -> bool:
        """Validate that run bins meet minimum event requirements.

        Parameters
        ----------
        bins : List[Tuple[int, int]]
            Run bins to validate
        data : pl.DataFrame
            Original data for validation

        Returns
        -------
        bool
            True if all bins meet requirements
        """
        for run_low, run_high in bins:
            bin_events = len(
                data.filter(
                    (pl.col("runNumber") >= run_low) & (pl.col("runNumber") <= run_high)
                )
            )

            if bin_events < self.min_events:
                self.logger.warning(
                    "Bin %d-%d has only %d events (< %d required)",
                    run_low,
                    run_high,
                    bin_events,
                    self.min_events,
                )
                return False

        return True
