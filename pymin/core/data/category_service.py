"""Efficient category initialization using Polars vectorized operations."""

import polars as pl
from typing import Dict, List, Tuple
import numpy as np


class CategoryInitializer:
    """Service for efficient category initialization using Polars vectorization."""

    def __init__(self, categories: CategoryCollection):
        self.categories = categories
        self.logger = logging.getLogger(__name__)

    def initialize_all_categories(
        self, data: pl.DataFrame, mc: pl.DataFrame
    ) -> Dict[Tuple[int, int], TwoParticleCategory]:
        """Initialize all two-particle categories efficiently.

        Uses Polars vectorized operations to avoid creating individual masks.

        Parameters
        ----------
        data : pl.DataFrame
            Data events with electron kinematics
        mc : pl.DataFrame
            MC events with electron kinematics and weights

        Returns
        -------
        Dict[Tuple[int, int], TwoParticleCategory]
            Dictionary mapping (lead_idx, sublead_idx) to categories
        """
        self.logger.info("Initializing categories using vectorized Polars operations")

        # Step 1: Add category indices to each electron using vectorized operations
        data_categorized = self._add_category_indices(data)
        mc_categorized = self._add_category_indices(mc)

        # Step 2: Create invariant mass for all electron pairs
        data_with_mass = self._compute_invariant_masses(data_categorized)
        mc_with_mass = self._compute_invariant_masses(mc_categorized)

        # Step 3: Group by category pairs efficiently
        category_groups = self._group_by_category_pairs(data_with_mass, mc_with_mass)

        # Step 4: Initialize TwoParticleCategory objects
        initialized_categories = {}
        for (lead_idx, sublead_idx), group_data in category_groups.items():
            category = self._create_category_from_group(
                lead_idx, sublead_idx, group_data
            )
            if category.valid:
                initialized_categories[(lead_idx, sublead_idx)] = category

        self.logger.info(f"Initialized {len(initialized_categories)} valid categories")
        return initialized_categories

    def _add_category_indices(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add category indices to electrons using vectorized conditions."""

        # Create category matching expressions for all categories at once
        category_expressions = []

        for cat in self.categories.get_active_categories():
            condition = (
                (pl.col("etaEle").abs() >= cat.bounds.eta_min)
                & (pl.col("etaEle").abs() < cat.bounds.eta_max)
                & (pl.col("R9Ele") >= cat.bounds.r9_min)
                & (pl.col("R9Ele") < cat.bounds.r9_max)
                & (pl.col("etEle") >= cat.bounds.et_min)
                & (pl.col("etEle") <= cat.bounds.et_max)
            )

            # Handle gain condition
            if cat.bounds.gain != -1:
                condition = condition & (pl.col("gainEle") == cat.bounds.gain)

            category_expressions.append(
                pl.when(condition)
                .then(cat.index)
                .otherwise(None)
                .alias(f"cat_{cat.index}")
            )

        # Add all category indices in one operation
        return df.with_columns(category_expressions)

    def _group_by_category_pairs(
        self, data_df: pl.DataFrame, mc_df: pl.DataFrame
    ) -> Dict:
        """Group events by category pairs efficiently."""

        category_groups = {}

        # Get all possible category combinations
        active_cats = [cat.index for cat in self.categories.get_active_categories()]

        for lead_idx in active_cats:
            for sublead_idx in active_cats:
                # Filter for this category pair using Polars efficiency
                data_pair = data_df.filter(
                    (pl.col(f"cat_{lead_idx}").is_not_null())
                    & (pl.col(f"cat_{sublead_idx}").is_not_null())
                )

                mc_pair = mc_df.filter(
                    (pl.col(f"cat_{lead_idx}").is_not_null())
                    & (pl.col(f"cat_{sublead_idx}").is_not_null())
                )

                if len(data_pair) > 0 and len(mc_pair) > 0:
                    category_groups[(lead_idx, sublead_idx)] = {
                        "data": data_pair,
                        "mc": mc_pair,
                    }

        return category_groups
