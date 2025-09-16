"""
Collection classes for:
- Calibrations
- TwoParticleCategories
"""

from dataclasses import dataclass, field
import pandas as pd

from pymin.core.models.calibration import Calibration, Scale, Smearing
from pymin.core.models.two_particle_category import TwoParticleCategory


@dataclass
class CalibrationCollection:
    """
    Collection of calibration objects (scales or smearings).
    """

    calibrations: list[Calibration] = field(default_factory=list)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "CalibrationCollection":
        """
        Create a CalibrationCollection from a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing calibration data.

        Returns
        -------
        CalibrationCollection
            A CalibrationCollection instance.
        """
        scales = [
            Scale.from_dataframe_row(row)
            for _, row in df.iterrows()
            if row["type"] == "scale"
        ]
        smearings = [
            Smearing.from_dataframe_row(row)
            for _, row in df.iterrows()
            if row["type"] == "smearing"
        ]
        return cls(calibrations=scales + smearings)


@dataclass
class TwoParticleCategoryCollection:
    """
    Collection of TwoParticleCategory objects.
    """

    category_map: dict[tuple[int, int], "TwoParticleCategory"] = field(
        default_factory=dict
    )

    @classmethod
    def from_list(
        cls, categories: list["TwoParticleCategory"]
    ) -> "TwoParticleCategoryCollection":
        """
        Create a TwoParticleCategoryCollection from a list of categories.

        Parameters
        ----------
        categories : list[TwoParticleCategory]
            List of TwoParticleCategory objects.

        Returns
        -------
        TwoParticleCategoryCollection
            A TwoParticleCategoryCollection instance.
        """
        category_map = {(cat.lead_idx, cat.sublead_idx): cat for cat in categories}
        return cls(category_map=category_map)

    def get_active_category_indices(self) -> ["TwoParticleCategory"]:
        """
        Get a list of active (valid) categories.

        Returns
        -------
        list[TwoParticleCategory]
            List of active category indices.
        """
        return [key for key, cat in self.category_map.items() if cat.valid]

    def get_active_categories(self) -> list["TwoParticleCategory"]:
        """
        Get a list of active (valid) categories.

        Returns
        -------
        list[TwoParticleCategory]
            List of active categories.
        """
        return [cat for cat in self.category_map.values() if cat.valid]

    def total_loss(self) -> float:
        """
        Calculate the total loss across all categories.

        Returns
        -------
        float
            Total loss.
        """
        return sum(cat.loss for cat in self.category_map.values() if cat.valid)
