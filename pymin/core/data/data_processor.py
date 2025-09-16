"""Modern data processing service for CMS ECAL calibration."""

import logging
from typing import Generator, Optional
import numpy as np
import pandas as pd

from pymin.config.prune_config import SelectionConfig


class DataProcessor:
    """Service for applying physics cuts and data transformations."""

    def __init__(
        self,
        selection_config: Optional[SelectionConfig] = None,
        logger: logging.Logger = None,
    ):
        self.selection_config = selection_config
        self.logger = logger or logging.getLogger(__name__)

        # CMS ECAL detector constants
        self.BARREL_MAX_ETA = 1.4442
        self.ENDCAP_MIN_ETA = 1.566
        self.ENDCAP_MAX_ETA = 2.5

    def process_stream(
        self, data_stream: Generator[pd.DataFrame, None, None]
    ) -> Generator[pd.DataFrame, None, None]:
        """Process data stream with physics cuts and transformations.

        Parameters
        ----------
        data_stream : Generator[pd.DataFrame, None, None]
            Input data stream

        Yields
        ------
        pd.DataFrame
            Processed data chunks
        """
        for chunk in data_stream:
            processed_chunk = self.process_chunk(chunk)
            if len(processed_chunk) > 0:
                yield processed_chunk

    def process_chunk(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply modern physics cuts to data chunk.

        Parameters
        ----------
        data : pd.DataFrame
            Input data chunk

        Returns
        -------
        pd.DataFrame
            Processed data with cuts applied
        """
        # Apply detector geometry cuts
        data = self._apply_detector_acceptance(data)

        # Apply energy cuts
        data = self._apply_energy_cuts(data)

        # Apply invariant mass window
        data = self._apply_mass_window(data)

        # Apply additional selection cuts if configured
        if self.selection_config:
            data = self._apply_selection_cuts(data)

        self.logger.debug(f"Processed chunk: {len(data)} events after cuts")
        return data

    def _apply_detector_acceptance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply CMS ECAL detector acceptance cuts."""
        if "etaEle" not in data.columns:
            return data

        # Convert to absolute eta for cuts
        abs_eta = np.abs(data["etaEle"].values)

        # Accept barrel (|η| < 1.4442) or endcap (1.566 < |η| < 2.5)
        barrel_mask = abs_eta < self.BARREL_MAX_ETA
        endcap_mask = (abs_eta > self.ENDCAP_MIN_ETA) & (abs_eta < self.ENDCAP_MAX_ETA)

        acceptance_mask = barrel_mask | endcap_mask
        return data[acceptance_mask]

    def _apply_energy_cuts(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply minimum energy cuts."""
        if "energy_ECAL_ele" in data.columns:
            # Standard minimum energy for electron calibration
            energy_mask = data["energy_ECAL_ele"] > 25.0
            data = data[energy_mask]

        return data

    def _apply_mass_window(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Z boson mass window cut."""
        if "invMass_ECAL_ele" in data.columns:
            # Z peak region for calibration
            mass_mask = (data["invMass_ECAL_ele"] > 60.0) & (
                data["invMass_ECAL_ele"] < 120.0
            )
            data = data[mass_mask]

        return data

    def _apply_selection_cuts(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional configurable selection cuts."""
        # R9 shower shape cuts
        if "R9Ele" in data.columns and self.selection_config:
            r9_mask = (data["R9Ele"] >= self.selection_config.r9_min) & (
                data["R9Ele"] <= self.selection_config.r9_max
            )
            data = data[r9_mask]

        # Additional eta cuts
        if "etaEle" in data.columns and self.selection_config:
            eta_mask = np.abs(data["etaEle"]) <= self.selection_config.eta_max
            data = data[eta_mask]

        # Transverse energy cuts
        if "etEle" in data.columns and self.selection_config:
            et_mask = data["etEle"] >= self.selection_config.et_min
            data = data[et_mask]

        return data
