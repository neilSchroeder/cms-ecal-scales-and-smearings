import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path if needed
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.classes.constant_classes import DataConstants as dc
from python.utilities.reweight_pt_y import add_weights_to_df


def validate_weight_assignment():
    """
    Validate weight assignment using the sample dataframe values
    """
    # Sample data points from the dataframe (ptZ, rapidity, claimed_weight)
    sample_data = [
        [49.138512, 0.458971, 0.0024031],
        [1.759194, 2.375784, 0.0020342],
        [4.815937, 0.772624, 0.0028883],
        [18.787357, 2.053268, 0.0024254],
        [50.984127, 1.105904, 0.001377],
        [29.718639, 1.590914, 0.001348],
        [12.776674, 0.528377, 0.002419],
        [51.830307, 1.565565, 0.001348],
        [1.201921, 0.664620, 0.001408],
        [17.850216, 0.946954, 0.001244],
    ]

    df = pd.DataFrame(sample_data, columns=["ptZ", "rapidity", "claimed_weight"])

    # Load the weights file
    weight_file = "ptz_x_rapidity_weights_step12_UL16_preVFP_pho_v10_EtaTune_R9Tune_EtTune_SmearTuneV3_fxedSmearingsNew_TEST.tsv"
    print(f"Loading weights from: {weight_file}")
    df_weight = pd.read_csv(weight_file, delimiter="\t", dtype=np.float32)

    # Process each sample point
    results = []
    for i, (ptZ, rapidity, claimed) in enumerate(
        zip(df["ptZ"], df["rapidity"], df["claimed_weight"])
    ):
        # Find the expected weight directly from the weight file
        expected_row = df_weight[
            (df_weight[dc.YMIN] <= rapidity)
            & (rapidity < df_weight[dc.YMAX])
            & (df_weight[dc.PTMIN] <= ptZ)
            & (ptZ < df_weight[dc.PTMAX])
        ]

        if len(expected_row) == 1:
            expected_weight = expected_row[dc.WEIGHT].values[0]

            # Get the weights for this rapidity bin
            y_weights = df_weight[
                (df_weight[dc.YMIN] <= rapidity) & (rapidity < df_weight[dc.YMAX])
            ].values

            # Create a Series with just the single ptZ value
            pt_series = pd.Series([ptZ])

            # Get the weight using our function
            actual_weight = add_weights_to_df((pt_series, y_weights)).values[0]

            # Determine the ptZ bin and y bin
            pt_bin = (
                f"{expected_row[dc.PTMIN].values[0]}-{expected_row[dc.PTMAX].values[0]}"
            )
            y_bin = (
                f"{expected_row[dc.YMIN].values[0]}-{expected_row[dc.YMAX].values[0]}"
            )

            results.append(
                {
                    "index": i,
                    "ptZ": ptZ,
                    "rapidity": rapidity,
                    "pt_bin": pt_bin,
                    "y_bin": y_bin,
                    "claimed_weight": claimed,
                    "expected_weight": expected_weight,
                    "actual_weight": actual_weight,
                    "matches_expected": np.isclose(actual_weight, expected_weight),
                    "matches_claimed": np.isclose(actual_weight, claimed),
                }
            )
        else:
            print(f"Warning: No unique bin found for ptZ={ptZ}, rapidity={rapidity}")

    # Create a results dataframe
    df_results = pd.DataFrame(results)

    # Print results
    print("\nWeight Validation Results:")
    print("=" * 110)
    print(
        f"{'Index':<6} {'ptZ':<10} {'y':<10} {'pt_bin':<12} {'y_bin':<10} {'Expected':<12} {'Actual':<12} {'Claimed':<12} {'Match?':<6}"
    )
    print("-" * 110)
    for _, r in df_results.iterrows():
        print(
            f"{int(r['index']):<6d} {r['ptZ']:<10.2f} {r['rapidity']:<10.2f} {r['pt_bin']:<12} {r['y_bin']:<10} "
            f"{r['expected_weight']:<12.8f} {r['actual_weight']:<12.8f} {r['claimed_weight']:<12.8f} "
            f"{'✓' if r['matches_expected'] else '✗'}"
        )

    # Overall result
    all_match = df_results["matches_expected"].all()
    print("=" * 110)
    print(f"Overall result: {'PASS' if all_match else 'FAIL'}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(df_results))
    width = 0.3

    # Plot bars for expected and actual weights
    ax.bar(
        x - width / 2,
        df_results["expected_weight"],
        width,
        label="Expected Weight",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        df_results["actual_weight"],
        width,
        label="Actual Weight",
        alpha=0.7,
    )

    # Add match indicators
    for i, r in enumerate(df_results.itertuples()):
        color = "green" if r.matches_expected else "red"
        ax.text(
            i,
            max(r.expected_weight, r.actual_weight) * 1.05,
            "✓" if r.matches_expected else "✗",
            ha="center",
            color=color,
            fontweight="bold",
        )

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Weight Value")
    ax.set_title("Weight Assignment Validation")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{i}: pt={r.ptZ:.1f}, y={r.rapidity:.2f}"
            for i, r in enumerate(df_results.itertuples())
        ],
        rotation=45,
        ha="right",
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig("weight_validation.png")

    print(f"\nValidation plot saved to: {Path('weight_validation.png').absolute()}")

    return all_match


if __name__ == "__main__":
    validate_weight_assignment()
