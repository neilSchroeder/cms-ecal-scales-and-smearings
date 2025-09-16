import unittest
from collections import OrderedDict

import pandas as pd

from src.classes.constant_classes import CategoryConstants as cc
from src.classes.constant_classes import DataConstants as dc
from src.core.data_loader import (
    add_transverse_energy,
    apply_custom_event_selection,
    apply_standard_event_selection,
    categorize_data_and_mc,
    get_dataframe,
    get_smearing_index,
)


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Set up test data
        self.files = ["/path/to/file1.root", "/path/to/file2.root"]
        self.df = pd.DataFrame(
            {
                dc.E_LEAD: [44, 45, 46],
                dc.ETA_LEAD: [0.1, 0.2, 0.3],
                dc.R9_LEAD: [0.1, 0.2, 0.3],
                dc.E_SUB: [44, 45, 46],
                dc.ETA_SUB: [0.1, 2.0, 2.4],
                dc.R9_SUB: [0.9, 0.91, 0.98],
                dc.INVMASS: [90, 91, 92],
            }
        )
        self.mc = pd.DataFrame(
            {
                dc.E_LEAD: [44, 45, 46],
                dc.ETA_LEAD: [0.1, 0.2, 0.3],
                dc.R9_LEAD: [0.1, 0.2, 0.3],
                dc.E_SUB: [44, 45, 46],
                dc.ETA_SUB: [0.1, 2.0, 2.4],
                dc.R9_SUB: [0.9, 0.91, 0.98],
                dc.INVMASS: [90, 91, 92],
            }
        )
        self.cats_df = pd.DataFrame(
            OrderedDict(
                {
                    "eta_min": [0.0],
                    "eta_max": [0.5],
                    "r9_min": [0.0],
                    "r9_max": [10],
                    "gain": [0],
                    "et_min": [0.0],
                    "et_max": [14000],
                }
            )
        )

    def test_get_dataframe(self):
        # Test get_dataframe function
        df = get_dataframe(self.files)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)  # Assuming no data in the test files

    def test_standard_cuts(self):
        # Test standard_cuts function
        df = apply_standard_event_selection(self.df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)  # Assuming no cuts applied

    def test_custom_cuts(self):
        # Test custom_cuts function
        df = apply_custom_event_selection(self.df, eta_cuts=((0.0, 1), (1, 3)))
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)  # Assuming 1 row is removed based on eta cuts

    def test_add_transverse_energy(self):
        # Test add_transverse_energy function
        data, mc = add_transverse_energy(self.df, self.mc)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIsInstance(mc, pd.DataFrame)
        self.assertEqual(len(data), 3)
        self.assertEqual(len(mc), 3)

    def test_extract_cats(self):
        # Test extract_cats function
        zcats = categorize_data_and_mc(self.df, self.mc, self.cats_df)
        self.assertIsInstance(zcats, list)
        self.assertEqual(len(zcats), 0)

    def test_get_smearing_index(self):
        # Create a test DataFrame that matches the expected structure
        # The function requires a 'type' column and specific category columns
        cats_data = {
            cc.i_type: [
                "scale",
                "scale",
                "smear",
                "smear",
            ],  # First two are scale categories, last two are smear
            cc.i_eta_min: [0.0, 0.1, 0.0, 0.1],
            cc.i_eta_max: [0.1, 0.2, 0.2, 0.3],
            cc.i_r9_min: [0.8, 0.9, 0.8, 0.9],
            cc.i_r9_max: [0.9, 1.0, 1.0, 1.0],
            cc.i_gain: [0, 0, 0, 0],
            cc.i_et_min: [20, 30, 20, 30],
            cc.i_et_max: [30, 40, 40, 50],
        }
        cats = pd.DataFrame(cats_data)

        # Test with a scale category index of 1 (eta_min=0.1, eta_max=0.2)
        # We expect it to match with the smear category at index 3
        cat_index = 1
        smearing_index = get_smearing_index(cats, cat_index)

        # The function should find the smear category that contains the scale category
        # In this case, smear index 3 (4th row) has eta_min=0.1, eta_max=0.3, which contains scale[1]
        self.assertEqual(smearing_index, 2)

        # Test with a different category index
        cat_index = 0
        smearing_index = get_smearing_index(cats, cat_index)

        # The smear category at index 2 (3rd row) has eta_min=0.0, eta_max=0.2, which contains scale[0]
        self.assertEqual(smearing_index, 2)


if __name__ == "__main__":
    unittest.main()
