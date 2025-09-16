from dataclasses import dataclass
from typing import Optional

import numpy as np
import numba

from pymin.core.models.calibration import Scale, Smearing
from pymin.core.models.histogram import Histogram
from pymin.config.defaults import (
    UNSET,
    MIN_EVENTS_DATA,
    MIN_EVENTS_MC_DIAG,
    MIN_EVENTS_MC_OFFDIAG,
)

EPSILON = 1e-15


@numba.njit(fastmath=True)
def compute_emd_loss(binned_data, binned_mc):
    """
    Compute Earth Mover's Distance (EMD) between two histograms.

    This function computes the Earth Mover's Distance (EMD) between two histograms.
    It is weighted by the number of events in each bin in data in order to strongly
    target the bins with the most events.

    Args:
        binned_data (np.array): binned data histogram
        binned_mc (np.array): binned mc histogram
    Returns:
        float: EMD loss
    """
    # Pre-normalize to avoid division
    sum_data = np.sum(binned_data)
    sum_mc = np.sum(binned_mc)

    return np.sum(
        binned_data**2
        * np.sqrt(  # approximation of abs for differentiability
            (
                np.cumsum(binned_data / (sum_data + EPSILON))  # normalize
                - np.cumsum(binned_mc / (sum_mc + EPSILON))
            )
            ** 2  # normalize
            + EPSILON
        )
        ** 0.5
    ) / np.sum(binned_data**2)


@dataclass
class TwoParticleCategory:
    """
    Class for two-particle categories.
    """

    # scales and smearings for leading and subleading particles
    lead_scale: Scale
    sublead_scale: Scale
    lead_smearing: Smearing
    sublead_smearing: Smearing

    # data and mc events, weights, and histograms
    data_events: np.ndarray
    mc_events: np.ndarray
    weights: np.ndarray

    # not needed outside class or at initialization
    _data_hist: Histogram = UNSET
    _mc_hist: Histogram = UNSET
    _bins: np.ndarray = UNSET

    # tracking and validation
    loss: float = 0.0
    valid: bool = True
    history: Optional[list] = UNSET

    @classmethod
    def from_events(
        cls,
        lead_scale: Scale,
        sublead_scale: Scale,
        lead_smearing: Smearing,
        sublead_smearing: Smearing,
        data_events: np.ndarray,
        mc_events: np.ndarray,
        weights: np.ndarray,
    ) -> "TwoParticleCategory":
        """
        Create a TwoParticleCategory from event arrays.

        Parameters
        ----------
        lead_scale : Scale
            Scale object for leading particle
        sublead_scale : Scale
            Scale object for subleading particle
        lead_smearing : Smearing
            Smearing object for leading particle
        sublead_smearing : Smearing
            Smearing object for subleading particle
        data_events : np.ndarray
            Array of data events (invariant masses)
        mc_events : np.ndarray
            Array of MC events (invariant masses)
        weights : np.ndarray
            Array of weights for MC events

        Returns
        -------
        TwoParticleCategory
            Initialized TwoParticleCategory object
        """
        return cls(
            lead_scale=lead_scale,
            sublead_scale=sublead_scale,
            lead_smearing=lead_smearing,
            sublead_smearing=sublead_smearing,
            data_events=data_events,
            mc_events=mc_events,
            weights=weights,
        )

    def __post_init__(self):
        """Initialize history and perform initial update."""
        if self.history is UNSET:
            self.history = []

        # Clean incoming data but preserve ALL events
        self.data_events = self.data_events[~np.isnan(self.data_events)]
        self.mc_events = self.mc_events[~np.isnan(self.mc_events)]

        # Apply corresponding mask to weights
        mc_nan_mask = ~np.isnan(self.mc_events)
        self.weights = self.weights[mc_nan_mask]

        # Initialize filter masks (will be updated in _update_filter_masks)
        self._data_filter_mask = np.ones(len(self.data_events), dtype=bool)
        self._mc_filter_mask = np.ones(len(self.mc_events), dtype=bool)

        # Store original events for scaling/smearing operations
        self._original_data_events = self.data_events.copy()
        self._original_mc_events = self.mc_events.copy()
        self._original_weights = self.weights.copy()

        # Initialize bins using all data (before filtering)
        self.bins = Histogram.determine_bin_size(self.data_events, self.mc_events)

        # Initialize with identity scales and minimal smearing
        self.update(1.0, 1.0, 0.001, 0.001)

        # Validate sufficient statistics
        if not self._check_statistics():
            print(
                "[INFO] TwoParticleCategory was deactivated due to insufficient statistics"
            )
            self._invalidate()

    def _update_filter_masks(
        self,
        scaled_data: np.ndarray,
        smeared_mc: np.ndarray,
        z_peak_min: float = 80.0,
        z_peak_max: float = 100.0,
    ) -> None:
        """
        Update filter masks based on current scaled/smeared events.

        This method dynamically updates which events are within the Z-peak
        window after applying the current scale and smearing corrections.

        Parameters
        ----------
        scaled_data : np.ndarray
            Data events after applying current scale corrections
        smeared_mc : np.ndarray
            MC events after applying current smearing corrections
        z_peak_min : float, default=80.0
            Lower bound of Z-peak invariant mass window (GeV)
        z_peak_max : float, default=100.0
            Upper bound of Z-peak invariant mass window (GeV)
        """
        # Update data filter mask
        self._data_filter_mask = (scaled_data >= z_peak_min) & (
            scaled_data <= z_peak_max
        )

        # Update MC filter mask
        self._mc_filter_mask = (smeared_mc >= z_peak_min) & (smeared_mc <= z_peak_max)

    def update(
        self,
        lead_scale_value: float,
        sublead_scale_value: float,
        lead_smear_value: Optional[float] = UNSET,
        sublead_smear_value: Optional[float] = UNSET,
    ) -> None:
        """
        Update the category with new scale and smearing values.

        This method applies scale corrections to both data and MC events,
        applies smearing corrections to MC events, updates histograms,
        and computes the loss function.

        Parameters
        ----------
        lead_scale_value : float
            Scale factor for leading particle (0 treated as 1.0)
        sublead_scale_value : float
            Scale factor for subleading particle (0 treated as 1.0)
        lead_smear_value : Optional[float]
            Smearing value for leading particle (if None, no change)
        sublead_smear_value : Optional[float]
            Smearing value for subleading particle (if None, no change)
        """
        if not self.valid:
            return

        # Apply scales (treat 0 as 1.0)
        lead_scale_value = 1.0 if lead_scale_value == 0 else lead_scale_value
        sublead_scale_value = 1.0 if sublead_scale_value == 0 else sublead_scale_value

        # Always apply scales to original data events
        scaled_data = Scale.apply_two_scales(
            self._original_data_events,
            lead_scale_value,
            sublead_scale_value,
        )

        # Update scale values
        self.lead_scale.value = lead_scale_value
        self.sublead_scale.value = sublead_scale_value

        # Apply smearing to MC events if values provided
        smeared_mc = self._original_mc_events.copy()
        current_weights = self._original_weights.copy()

        if (
            lead_smear_value is not UNSET
            and lead_smear_value != 0
            and abs(self.lead_smearing.value - lead_smear_value)
            > self.lead_smearing.tolerance
        ):
            # Transform existing smearing distribution
            self.lead_smearing.random_energy_factors = (
                self.lead_smearing.transform_smearing(lead_smear_value)
            )
            self.lead_smearing.value = lead_smear_value

        if (
            sublead_smear_value is not UNSET
            and sublead_smear_value != 0
            and abs(self.sublead_smearing.value - sublead_smear_value)
            > self.sublead_smearing.tolerance
        ):
            # Transform existing smearing distribution
            self.sublead_smearing.random_energy_factors = (
                self.sublead_smearing.transform_smearing(sublead_smear_value)
            )
            self.sublead_smearing.value = sublead_smear_value

        # Apply smearing to MC events
        smeared_mc = Smearing.apply_two_smearings(
            self._original_mc_events,
            self.lead_smearing.random_energy_factors,
            self.sublead_smearing.random_energy_factors,
        )

        # Apply correction factor for correlated smearing
        correction_factor = 1.0
        if (
            lead_smear_value is not UNSET
            and sublead_smear_value is not UNSET
            and lead_smear_value != 0
            and sublead_smear_value != 0
        ):
            correction_factor = 1 - (lead_smear_value * sublead_smear_value / 8)

        smeared_mc = smeared_mc / correction_factor

        # Update filter masks based on current scaled/smeared events
        self._update_filter_masks(scaled_data, smeared_mc)

        # Apply masks to get events within Z-peak window
        filtered_data = scaled_data[self._data_filter_mask]
        filtered_mc = smeared_mc[self._mc_filter_mask]
        filtered_weights = current_weights[self._mc_filter_mask]

        # Create histograms with filtered events
        self.data_hist = Histogram.from_array(
            filtered_data,
            auto_bin=(
                True if not hasattr(self, "data_hist") else self.data_hist.auto_bin
            ),
        )
        self.mc_hist = Histogram.from_array(
            filtered_mc,
            filtered_weights,
            auto_bin=True if not hasattr(self, "mc_hist") else self.mc_hist.auto_bin,
        )

        # Store current filtered events for statistics checking
        self.data_events = filtered_data
        self.mc_events = filtered_mc
        self.weights = filtered_weights

        # Validate sufficient statistics
        if not self._check_statistics():
            print(
                "[INFO] TwoParticleCategory was deactivated due to insufficient statistics"
            )
            self._invalidate()
            return

        # Compute loss using Earth Mover's Distance
        self.loss = compute_emd_loss(self.data_hist.values, self.mc_hist.values)

        # Update history
        if self.history is not None:
            self.history.append(
                {
                    "lead_scale": lead_scale_value,
                    "sublead_scale": sublead_scale_value,
                    "lead_smear": lead_smear_value,
                    "sublead_smear": sublead_smear_value,
                    "nll": self.loss,
                    "bin_size": self.data_hist.bin_width,
                }
            )

        # Check for NaN in loss function
        if np.isnan(self.loss):
            print(
                "[INFO] TwoParticleCategory was deactivated due to NaN in loss function"
            )
            self._invalidate()

    def _check_statistics(self) -> bool:
        """Check if category has sufficient statistics for analysis."""
        data_events = np.sum(self.data_hist.values)
        mc_events = np.sum(self.mc_hist.values)

        # Determine if diagonal or off-diagonal category
        is_diagonal = self.lead_scale.index == self.sublead_scale.index
        min_mc_events = MIN_EVENTS_MC_DIAG if is_diagonal else MIN_EVENTS_MC_OFFDIAG

        return data_events >= MIN_EVENTS_DATA and mc_events >= min_mc_events

    def _invalidate(self) -> None:
        """Mark category as invalid and clean up resources."""
        self.valid = False
        self.loss = 1e30
        # Clear large arrays to free memory
        self.data_events = np.array([])
        self.mc_events = np.array([])
        self.weights = np.array([])
