"""Modern TSV writing service with compression and merging."""

import gzip
import logging
from pathlib import Path
from typing import Generator
import pandas as pd

from pymin.config.defaults import UNSET


class TsvWriter:
    """Service for writing TSV files with modern features."""

    def __init__(
        self, output_dir: Path, compression: bool = True, logger: logging.Logger = UNSET
    ):
        self.output_dir = output_dir
        self.compression = compression
        self.logger = logger or logging.getLogger(__name__)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_merged_output(
        self,
        data_stream: Generator[pd.DataFrame, None, None],
        output_prefix: str,
        file_type: str,
    ) -> Path:
        """Write data stream to merged TSV file.

        Parameters
        ----------
        data_stream : Generator[pd.DataFrame, None, None]
            Stream of data chunks
        output_prefix : str
            Prefix for output file
        file_type : str
            File type identifier

        Returns
        -------
        Path
            Path to created output file
        """
        # Determine output file path
        extension = ".tsv.gz" if self.compression else ".tsv"
        output_file = self.output_dir / f"{output_prefix}_{file_type}{extension}"

        self.logger.info(f"Writing merged output to {output_file}")

        first_chunk = True
        total_events = 0

        # Open file with appropriate mode (compressed or not)
        file_opener = gzip.open if self.compression else open
        mode = "wt" if self.compression else "w"

        with file_opener(output_file, mode) as f:
            for chunk in data_stream:
                # Write header only for first chunk
                write_header = first_chunk

                # Write chunk to file
                chunk.to_csv(
                    f, sep="\t", index=False, header=write_header, lineterminator="\n"
                )

                total_events += len(chunk)
                first_chunk = False

                self.logger.debug(f"Wrote chunk with {len(chunk)} events")

        self.logger.info(f"Completed writing {total_events} events to {output_file}")
        return output_file

    def write_dataframe(
        self, data: pd.DataFrame, output_path: Path, append: bool = False
    ) -> Path:
        """Write single DataFrame to TSV file.

        Parameters
        ----------
        data : pd.DataFrame
            Data to write
        output_path : Path
            Output file path
        append : bool
            Whether to append to existing file

        Returns
        -------
        Path
            Path to written file
        """
        mode = "a" if append else "w"
        write_header = not append

        if self.compression and not output_path.suffix.endswith(".gz"):
            output_path = output_path.with_suffix(output_path.suffix + ".gz")
            file_opener = gzip.open
            mode = mode + "t"
        else:
            file_opener = open

        with file_opener(output_path, mode) as f:
            data.to_csv(
                f, sep="\t", index=False, header=write_header, lineterminator="\n"
            )

        self.logger.info(f"Written {len(data)} events to {output_path}")
        return output_path
