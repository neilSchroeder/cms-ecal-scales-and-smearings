"""Modern ROOT file reading service with streaming capabilities."""

import logging
from pathlib import Path
from typing import Generator, List
import pandas as pd

from pymin.config.defaults import UNSET

try:
    import uproot
except ImportError:
    uproot = None


class RootFileReader:
    """Service for reading ROOT files with modern streaming patterns."""

    def __init__(
        self,
        tree_name: str,
        branches: List[str],
        chunk_size: int = 100000,
        logger: logging.Logger = UNSET,
    ):
        self.tree_name = tree_name
        self.branches = branches
        self.chunk_size = chunk_size
        self.logger = logger or logging.getLogger(__name__)

        if uproot is None:
            raise ImportError("uproot required for ROOT file operations")

    def read_files_chunked(
        self, file_paths: List[Path]
    ) -> Generator[pd.DataFrame, None, None]:
        """Read multiple ROOT files as chunked data stream.

        Parameters
        ----------
        file_paths : List[Path]
            List of ROOT file paths

        Yields
        ------
        pd.DataFrame
            Chunks of data from ROOT files
        """
        for file_path in file_paths:
            self.logger.debug(f"Reading {file_path}")

            try:
                with uproot.open(file_path) as root_file:
                    tree = root_file[self.tree_name]

                    # Use uproot's built-in chunking for memory efficiency
                    yield from tree.iterate(
                        expressions=self.branches,
                        step_size=self.chunk_size,
                        library="pd",
                    )

            except Exception as e:
                self.logger.error(f"Failed to read {file_path}: {e}")
                raise

    def read_single_file(self, file_path: Path) -> pd.DataFrame:
        """Read entire ROOT file into DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to ROOT file

        Returns
        -------
        pd.DataFrame
            Complete data from ROOT file
        """
        with uproot.open(file_path) as root_file:
            tree = root_file[self.tree_name]
            return tree.arrays(self.branches, library="pd")
