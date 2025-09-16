"""Polars-based data reader for efficient CMS ECAL data processing."""

import logging
from pathlib import Path
from typing import List, Union

import polars as pl

from pymin.config.defaults import UNSET


class PolarsReader:
    """Service for reading CMS data files using efficient Polars operations."""

    def __init__(self, file_format: str = "tsv", logger: logging.Logger = UNSET):
        self.file_format = file_format
        self.logger = logger or logging.getLogger(__name__)

    def read_files(self, file_paths: List[str]) -> pl.DataFrame:
        """Read multiple files and concatenate efficiently.

        Parameters
        ----------
        file_paths : List[str]
            List of file paths to read (supports .gz compression)

        Returns
        -------
        pl.DataFrame
            Concatenated data from all files
        """
        dataframes = []

        for file_path in file_paths:
            self.logger.debug("Reading file: %s", file_path)

            try:
                # Detect if file is gzip compressed
                is_compressed = str(file_path).endswith(".gz")

                if self.file_format == "tsv":
                    df = pl.read_csv(
                        file_path,
                        separator="\t",
                        try_parse_dates=False,
                        infer_schema_length=1000,
                        # Polars automatically detects .gz files
                        encoding="utf8",
                    )
                elif self.file_format == "csv":
                    df = pl.read_csv(file_path, encoding="utf8")
                else:
                    raise ValueError(f"Unsupported file format: {self.file_format}")

                dataframes.append(df)

                # Log compression info for debugging
                compression_info = " (gzip compressed)" if is_compressed else ""
                self.logger.debug(
                    "Loaded %d rows from %s%s", len(df), file_path, compression_info
                )

            except Exception as e:
                self.logger.error("Failed to read %s: %s", file_path, e)
                raise

        if not dataframes:
            raise ValueError("No data loaded from input files")

        # Efficient concatenation with Polars
        combined_df = pl.concat(dataframes, how="vertical")
        self.logger.info(
            "Combined %d files into %d total rows", len(dataframes), len(combined_df)
        )

        return combined_df

    def read_single_file(self, file_path: Union[str, Path]) -> pl.DataFrame:
        """Read a single file efficiently.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to file (supports .gz compression)

        Returns
        -------
        pl.DataFrame
            Data from file
        """
        return self.read_files([str(file_path)])

    def scan_lazy(self, file_paths: List[str]) -> pl.LazyFrame:
        """Create lazy scan for memory-efficient processing of large datasets.

        This is particularly useful for CMS datasets that exceed memory limits.

        Parameters
        ----------
        file_paths : List[str]
            List of file paths to scan

        Returns
        -------
        pl.LazyFrame
            Lazy frame for memory-efficient operations
        """
        lazy_frames = []

        for file_path in file_paths:
            if self.file_format == "tsv":
                lazy_df = pl.scan_csv(
                    file_path,
                    separator="\t",
                    try_parse_dates=False,
                    infer_schema_length=1000,
                )
            else:
                lazy_df = pl.scan_csv(file_path)

            lazy_frames.append(lazy_df)

        # Concatenate lazy frames
        if len(lazy_frames) == 1:
            return lazy_frames[0]
        else:
            return pl.concat(lazy_frames, how="vertical")
