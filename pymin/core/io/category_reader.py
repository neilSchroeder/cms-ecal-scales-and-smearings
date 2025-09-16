import pandas as pd

from pymin.core.models.calibration import Calibration, Scale, Smearing
from pymin.config.defaults import UNSET


class CategoryReader:
    """Service for reading calibration categories from CSV files."""

    def read_categories(self, file_path: str) -> list[Calibration]:
        """Read calibration categories from a CSV file.

        Parameters
        ----------
        file_path : str
            Path to the CSV file containing calibration categories.

        Returns
        -------
        list[Calibration]
            List of Calibration objects read from the file.
        """
        df = pd.read_csv(file_path)

        categories = []
        for _, row in df.iterrows():
            if row["type"] == "scale":
                category = Scale.from_dataframe_row(row)
            elif row["type"] == "smearing":
                category = Smearing.from_dataframe_row(row)
            else:
                raise ValueError(f"Unknown calibration type: {row['type']}")

            categories.append(category)

        return categories
