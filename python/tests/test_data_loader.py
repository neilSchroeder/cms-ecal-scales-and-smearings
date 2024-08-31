import unittest
import pandas as pd

from python.tools.data_loader import (
    get_dataframe,
    standard_cuts,
    custom_cuts,
    add_transverse_energy,
    extract_cats,
    get_smearing_index,
)


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Set up test data
        self.files = ["/path/to/file1.root", "/path/to/file2.root"]
        self.df = pd.DataFrame({"eta": [0.1, 0.2, 0.3], "et": [10, 20, 30]})
        self.mc = pd.DataFrame({"eta": [0.1, 0.2, 0.3], "et": [10, 20, 30]})
        self.cats_df = pd.DataFrame(
            {"eta_min": [0.0, 0.1, 0.2], "eta_max": [0.1, 0.2, 0.3]}
        )

    def test_get_dataframe(self):
        # Test get_dataframe function
        df = get_dataframe(self.files)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)  # Assuming no data in the test files

    def test_standard_cuts(self):
        # Test standard_cuts function
        df = standard_cuts(self.df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)  # Assuming no cuts applied

    def test_custom_cuts(self):
        # Test custom_cuts function
        df = custom_cuts(self.df, eta_cuts=(0.1, 0.2))
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
        zcats = extract_cats(self.df, self.mc, self.cats_df)
        self.assertIsInstance(zcats, list)
        self.assertEqual(len(zcats), 3)

    def test_get_smearing_index(self):
        # Test get_smearing_index function
        cats = pd.DataFrame({"eta_min": [0.0, 0.1, 0.2], "eta_max": [0.1, 0.2, 0.3]})
        cat_index = 1
        smearing_index = get_smearing_index(cats, cat_index)
        self.assertEqual(smearing_index, 1)


if __name__ == "__main__":
    unittest.main()
