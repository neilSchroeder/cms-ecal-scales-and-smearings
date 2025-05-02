import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from python.classes.constant_classes import DataConstants as dc
from python.utilities.reweight_pt_y import add_weights_to_df


def create_test_weight_file():
    """Create a simple weight file with known values for testing"""
    # Define column names
    columns = [dc.YMIN, dc.YMAX, dc.PTMIN, dc.PTMAX, dc.WEIGHT]

    # Create a grid of weights with clear pattern: y_bin*10 + pt_bin
    data = []
    # 3x3 grid of bins for y and pT
    y_bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    pt_bins = [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0)]

    for y_idx, (y_min, y_max) in enumerate(y_bins):
        for pt_idx, (pt_min, pt_max) in enumerate(pt_bins):
            weight = (y_idx + 1) * 10 + (
                pt_idx + 1
            )  # Create unique weight for each bin
            data.append([y_min, y_max, pt_min, pt_max, weight])

    df_weights = pd.DataFrame(data, columns=columns)

    # Save to temporary file
    temp_file = os.path.join(tempfile.gettempdir(), "test_weights.txt")
    df_weights.to_csv(temp_file, sep="\t", index=False)

    return temp_file, df_weights


def test_weights_for_bin(y_bin, pt_values, weights_array):
    """Test weight assignment for a specific y bin and set of pt values"""
    # Create a series of pt values
    pt_series = pd.Series(pt_values)

    # Apply the add_weights_to_df function
    result = add_weights_to_df((pt_series, weights_array))

    return result.tolist()


def validate_weight_assignment():
    """Validate the weight assignment logic"""
    # Create test weights file
    weight_file, df_weights = create_test_weight_file()

    print("Test weight matrix:")
    print(df_weights)

    # Test points - one for each bin center
    test_points = []
    expected_weights = []

    # Create test points for each bin
    for y_idx in range(3):
        y_min, y_max = (
            df_weights[dc.YMIN].unique()[y_idx],
            df_weights[dc.YMAX].unique()[y_idx],
        )
        weights_for_y = df_weights[df_weights[dc.YMIN] == y_min].values

        # Test points near the center of each pT bin
        pt_values = [15.0, 45.0, 75.0]  # Centers of pT bins
        expected = [(y_idx + 1) * 10 + (pt_idx + 1) for pt_idx in range(3)]

        # Get actual weights
        actual = test_weights_for_bin(y_idx, pt_values, weights_for_y)

        test_points.extend([(y_idx, pt) for pt in pt_values])
        expected_weights.extend(expected)

        # Print results for this y bin
        print(f"\nTesting y_bin {y_idx} ({y_min:.1f}-{y_max:.1f}):")
        for i, (pt, exp, act) in enumerate(zip(pt_values, expected, actual)):
            match = "✓" if np.isclose(exp, act) else "✗"
            print(
                f"  pT={pt:4.1f} - Expected: {exp:4.1f}, Actual: {act:4.1f} - {match}"
            )

    # Visualize results
    fig, ax = plt.subplots(figsize=(12, 8))

    # Group by y bin for visualization
    for y_idx in range(3):
        indices = [i for i, (y, _) in enumerate(test_points) if y == y_idx]

        # Get corresponding pt values, expected and actual weights
        pts = [test_points[i][1] for i in indices]
        exp = [expected_weights[i] for i in indices]
        act = [actual[i % 3] for i in range(3)]  # Take from the actual results

        # Plot as grouped bars
        x = np.arange(len(pts)) + y_idx * 4
        ax.bar(
            x - 0.2,
            exp,
            width=0.4,
            alpha=0.7,
            label=f"Expected (y_bin={y_idx})" if y_idx == 0 else "",
        )
        ax.bar(
            x + 0.2,
            act,
            width=0.4,
            alpha=0.7,
            label=f"Actual (y_bin={y_idx})" if y_idx == 0 else "",
        )

        # Add text labels
        for i, (e, a) in enumerate(zip(exp, act)):
            match = "✓" if np.isclose(e, a) else "✗"
            color = "green" if np.isclose(e, a) else "red"
            ax.text(
                x[i], max(e, a) + 1, match, ha="center", color=color, fontweight="bold"
            )

    ax.set_ylabel("Weight Value")
    ax.set_title("Weight Assignment Validation")
    ax.legend()

    # Set x-ticks
    all_x = np.array([i + j * 4 for j in range(3) for i in range(3)])
    ax.set_xticks(all_x)
    ax.set_xticklabels([f"y{p[0]},pt={p[1]}" for p in test_points], rotation=45)

    plt.tight_layout()
    plt.savefig("weight_validation.png")

    # Cleanup
    os.remove(weight_file)

    print("\nValidation complete. Results saved to 'weight_validation.png'")


if __name__ == "__main__":
    validate_weight_assignment()
