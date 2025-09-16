from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
import numpy as np

from pymin.config.defaults import UNSET


@dataclass
class Calibration:
    """
    Base class for calibrations (scales and smearings).
    """

    type: str
    index: int
    eta_min: float
    eta_max: float
    r9_min: float
    r9_max: float
    gain: int
    et_min: float
    et_max: float
    value: float = UNSET
    uncertainty: float = UNSET

    @classmethod
    def from_dataframe_row(cls, row: pd.Series) -> "Calibration":
        """
        Create a Calibration object from a dataframe row.

        Parameters
        ----------
        row : dict
            Row from a dataframe containing calibration parameters.

        Returns
        -------
        Calibration
            The created Calibration object.
        """
        return cls(
            type=row["type"],
            index=row.index[0],
            eta_min=row["eta_min"],
            eta_max=row["eta_max"],
            r9_min=row["r9_min"],
            r9_max=row["r9_max"],
            gain=row["gain"],
            et_min=row["et_min"],
            et_max=row["et_max"],
        )


@dataclass
class Scale(Calibration):
    """
    Class for energy scale calibration categories.
    """

    type: str = "scale"
    value: float = 1.0

    @classmethod
    def apply_two_scales(
        cls, energy: np.ndarray, scale1: float, scale2: float
    ) -> np.ndarray:
        """
        Apply two scale factors to the energy.

        Parameters
        ----------
        energy : np.ndarray
            Original energy values.
        scale1 : float
            First scale factor.
        scale2 : float
            Second scale factor.

        Returns
        -------
        np.ndarray
            Scaled energy values.
        """
        new_scale = np.sqrt(scale1 * scale2)
        return energy * new_scale


@dataclass
class Smearing(Calibration):
    """
    Class for energy smearing calibration categories.
    """

    type: str = "smearing"
    value: float = 0.001
    tolerance: float = 1e-6
    seed: int = 3543136929
    random_generator: np.random.Generator = np.random.Generator(np.random.PCG64(seed))
    random_energy_factors: np.ndarray = np.array([])

    @classmethod
    def apply_two_smearings(
        cls, energy: np.ndarray, rand_factors1: np.ndarray, rand_factors2: np.ndarray
    ) -> np.ndarray:
        """
        Apply two smearing factors to the energy.

        Parameters
        ----------
        energy : np.ndarray
            Original energy values.
        rand_factors1 : np.ndarray
            First set of random smearing factors.
        rand_factors2 : np.ndarray
            Second set of random smearing factors.

        Returns
        -------
        np.ndarray
            Smeared energy values.

        Example
        -------
        >>> energy = np.array([100.0, 200.0, 300.0])
        >>> smearing1 = Smearing()
        >>> smearing2 = Smearing()
        >>> smeared_energy = Smearing.apply_two_smearings(
        ...     energy,
        ...     smearing1.random_energy_factors,
        ...     smearing2.random_energy_factors
        ... )
        """
        new_smearing = np.sqrt(rand_factors1**2 + rand_factors2**2)
        return energy * (1 + new_smearing)

    def transform_smearing(self, new_smearing: float) -> np.ndarray:
        """
        Transform the current smearing to a new smearing value.

        Parameters
        ----------
        new_smearing : float
            New smearing value to transform to.

        Returns
        -------
        np.ndarray
            Transformed random energy factors.

        Example
        -------
        >>> smearing = Smearing(value=0.01)
        >>> transformed_factors = smearing.transform_smearing(new_smearing=0.02)
        """
        # avoid negative or zero smearing
        if new_smearing <= self.tolerance:
            new_smearing = self.tolerance

        return ((self.random_energy_factors - 1) * (new_smearing / self.value)) + 1
