"""
emd_vs_poisson.py — Compare EMD and Poisson-NLL·chi² loss surfaces on a toy
that mimics the production pipeline as closely as possible.

Setup
-----
Two single-electron categories with distinct CB tail shapes (A=EB-like, narrow;
B=EE-like, wider) yield three di-electron pairings: AA, AB, BB. Off-diagonal AB
gets weight 0.1 (production default). Each pairing uses Freedman-Diaconis bin
sizing computed on the (injected) data array — same recipe as
zcat_class.__init__.

Truth values are stated from the MINIMIZER'S perspective: the trial scale/smear
that recovers the global minimum. Data is injected with the matching factors so
the recovered minimum lands on the stated truth.

  MC   : scale = 1.000 (both cats), smear = 0.005 (both cats), N = 100k
  Data : scale_A = 1.010, scale_B = 0.995, smear_A = 0.015, smear_B = 0.025

Both losses are evaluated from the exact same binned histograms used in
production (numba_hist edges + apply_smearing_cached), so the comparison is
faithful to the optimizer's view.

Scans
-----
- Four 1D scans (s_A, s_B, σ_A, σ_B), 21 points each, others pinned at truth.
- Two 2D contour scans: (s_A, σ_A) and (s_A, s_B), 21x21.

For each grid point: total loss = sum_{ij in {AA,AB,BB}} w_ij * loss_ij with
w_AA = w_BB = 1.0, w_AB = 0.1.

Outputs (under scripts/experiments/output/, gitignored):
  - scans_1d.png            : 2x2 grid, EMD and NLL on twin y-axes
  - scan_2d_sA_sigA.png     : two contour panels (EMD | NLL)
  - scan_2d_sA_sB.png       : two contour panels (EMD | NLL)
  - summary.txt             : curvature / well-width / min offset table

Run
---
  source .venv/bin/activate
  python -m scripts.experiments.emd_vs_poisson
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Use production hot-path functions directly
from python.classes.zcat_class import (
    apply_smearing_cached,
    _generate_smearing_randn,
    compute_earthmovers_distance,
    compute_nll_chisqr,
    build_uniform_weights,
)
from python.classes.breit_wigner import bw
from python.classes.crystal_ball import cb
from python.utilities import numba_hist


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIST_MIN, HIST_MAX = 80.0, 100.0
N_EVENTS = 100_000
RNG_SEED = 20260523

# Per-electron-category CB shape (response factor pdf around 1.0)
CAT_A = dict(alpha=1.5, n=2.0, mean=1.0, width=0.012)   # narrow / EB-like
CAT_B = dict(alpha=1.0, n=3.0, mean=1.0, width=0.022)   # wider  / EE-like

# Injection values (what we apply to DATA at construction time).
# Production convention: zcat.inject multiplies data by sqrt(s*s) = s, and
# zcat.update at fit-time multiplies data by trial_scale. So for scales the
# MINIMUM is at trial = 1/inject, not at trial = inject.
# For smears: data is pre-smeared by σ_inj. MC has NO injected smear, so the
# optimum trial smear equals σ_inj exactly.
INJECT = {
    "scale_A": 1.010,
    "scale_B": 0.995,
    "smear_A": 0.040,
    "smear_B": 0.060,
}
MC_SMEAR_INJECT = 0.0  # MC is CB-only at construction; trial-smear adds on top

# Where the minimizer SHOULD land (truth from the optimizer's perspective):
TRUTH = {
    "scale_A": 1.0 / INJECT["scale_A"],
    "scale_B": 1.0 / INJECT["scale_B"],
    "smear_A": float(np.sqrt(INJECT["smear_A"] ** 2 - MC_SMEAR_INJECT ** 2)),
    "smear_B": float(np.sqrt(INJECT["smear_B"] ** 2 - MC_SMEAR_INJECT ** 2)),
}

# Pairing weights (production default: diag=1.0, off-diag=0.1)
W_AA, W_AB, W_BB = 1.0, 0.1, 1.0


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def _inverse_cdf_sample(x_grid, pdf_vals, n, rng):
    """Inverse-CDF sampling on a fine grid."""
    cdf = np.cumsum(pdf_vals)
    cdf /= cdf[-1]
    u = rng.uniform(0, 1, n)
    return np.interp(u, cdf, x_grid)


def sample_bw_masses(n, rng):
    """Sample true m_ee from a relativistic Z Breit-Wigner."""
    x = np.linspace(HIST_MIN, HIST_MAX, 4000)
    pdf = bw(x).y
    return _inverse_cdf_sample(x, pdf, n, rng)


def sample_cb_factors(n, params, rng):
    """Sample per-electron response factors from a Crystal Ball pdf."""
    x = np.linspace(0.8, 1.2, 4000)
    pdf = cb(x, [params["alpha"], params["n"], params["mean"], params["width"]]).y
    return _inverse_cdf_sample(x, pdf, n, rng)


def sample_pairing(cat_lead, cat_sublead, n, rng):
    """
    Generate n di-electron masses for one pairing:
      m_ee = m_ee_true * sqrt(f_lead * f_sublead)
    where m_ee_true ~ BW and f_e ~ CB(cat_e).
    """
    m_true = sample_bw_masses(n, rng)
    f_lead = sample_cb_factors(n, cat_lead, rng)
    f_sub = sample_cb_factors(n, cat_sublead, rng)
    return (m_true * np.sqrt(f_lead * f_sub)).astype(np.float32)


def inject_data(masses, scale, smear, rng):
    """
    Mirror zcat.inject: data <- data * sqrt(scale*scale) * sqrt((1+ε_lead)(1+ε_sublead)).
    Here `scale` and `smear` are PER-ELECTRON values (lead == sublead since the
    pairing is single-category for the injection step).
    """
    out = masses * float(scale)  # sqrt(scale*scale) for matched lead/sublead
    if smear > 0:
        eps_lead = rng.normal(0.0, smear, len(out)).astype(np.float32)
        eps_sub = rng.normal(0.0, smear, len(out)).astype(np.float32)
        out = out * np.sqrt((1.0 + eps_lead) * (1.0 + eps_sub))
    return out.astype(np.float32)


def inject_mixed_data(masses, scale_lead, scale_sublead, smear_lead, smear_sublead, rng):
    """Off-diagonal injection (AB pairing)."""
    out = masses * float(np.sqrt(scale_lead * scale_sublead))
    eps_lead = rng.normal(0.0, smear_lead, len(out)).astype(np.float32) if smear_lead > 0 else 0.0
    eps_sub = rng.normal(0.0, smear_sublead, len(out)).astype(np.float32) if smear_sublead > 0 else 0.0
    if smear_lead > 0 or smear_sublead > 0:
        out = out * np.sqrt((1.0 + eps_lead) * (1.0 + eps_sub))
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Pairing container (mimics the subset of zcat we need)
# --------------------------------------------------------------------------- #

@dataclass
class Pairing:
    name: str
    weight: float
    data: np.ndarray         # injected toy data
    mc: np.ndarray           # untouched toy mc
    bin_edges: np.ndarray
    num_bins: int
    randn_lead: np.ndarray
    randn_sublead: np.ndarray
    emd_weights: np.ndarray

    def evaluate(self, s_lead, s_sublead, sigma_lead, sigma_sublead):
        """Apply trial (scale, smear), histogram, return (emd, nll_chisqr)."""
        scale_factor = np.float32(np.sqrt(s_lead * s_sublead))
        d = self.data * scale_factor
        if sigma_lead == 0 and sigma_sublead == 0:
            m = self.mc
        else:
            m = apply_smearing_cached(
                self.mc, sigma_lead, sigma_sublead, self.randn_lead, self.randn_sublead
            )

        # window + sentinels (match zcat.update)
        mask_d = (d >= HIST_MIN) & (d <= HIST_MAX)
        mask_m = (m >= HIST_MIN) & (m <= HIST_MAX)
        sent = np.array([HIST_MIN, HIST_MAX], dtype=np.float32)
        d_w = np.concatenate([d[mask_d], sent])
        m_w = np.concatenate([m[mask_m], sent])
        w_w = np.concatenate(
            [np.ones(mask_m.sum(), dtype=np.float32), np.zeros(2, dtype=np.float32)]
        )

        binned_d = numba_hist.numba_histogram_with_edges(d_w, self.bin_edges).astype(np.float64)
        binned_m = numba_hist.numba_weighted_histogram_with_edges(m_w, w_w, self.bin_edges).astype(np.float64)
        binned_m[binned_m == 0] = 1e-15
        norm_m = (binned_m / np.sum(binned_m)).astype(np.float64)

        emd = compute_earthmovers_distance(binned_d, norm_m, self.emd_weights)
        nll = compute_nll_chisqr(binned_d, norm_m, num_bins=self.num_bins)
        return float(emd), float(nll)


def make_pairing(name, weight, data, mc):
    """Set up Freedman-Diaconis binning & cached randn for one pairing."""
    d_win = data[(data >= HIST_MIN) & (data <= HIST_MAX)]
    m_win = mc[(mc >= HIST_MIN) & (mc <= HIST_MAX)]
    iqr_d = stats.iqr(d_win, rng=(25, 75)) if len(d_win) > 1 else 0.25
    iqr_m = stats.iqr(m_win, rng=(25, 75)) if len(m_win) > 1 else 0.25
    fd_d = 2.0 * iqr_d / (len(d_win) ** (1.0 / 3.0))
    fd_m = 2.0 * iqr_m / (len(m_win) ** (1.0 / 3.0))
    bin_size = max(fd_d, fd_m, 0.05)  # floor to keep n_bins reasonable
    num_bins = int(round((HIST_MAX - HIST_MIN) / bin_size))
    bin_edges = numba_hist.make_bin_edges(HIST_MIN, HIST_MAX, num_bins)
    # cached gaussians (use a fixed seed per pairing — production uses _derive_cat_seed)
    randn_lead, randn_sublead = _generate_smearing_randn(len(mc), abs(hash(name)) % (2**31))
    emd_weights = build_uniform_weights(num_bins)
    print(f"  [{name}] N_data={len(data)} N_mc={len(mc)} bin_size={bin_size:.3f} num_bins={num_bins}")
    return Pairing(
        name=name, weight=weight, data=data, mc=mc,
        bin_edges=bin_edges, num_bins=num_bins,
        randn_lead=randn_lead, randn_sublead=randn_sublead,
        emd_weights=emd_weights,
    )


# --------------------------------------------------------------------------- #
# Total loss over pairings
# --------------------------------------------------------------------------- #

def total_loss(pairings, s_A, s_B, sig_A, sig_B):
    """Return (total_emd, total_nll) summed (weighted) over AA, AB, BB."""
    # Pairing parameters: which (lead, sublead) scale/smear applies?
    cfg = {
        "AA": (s_A, s_A, sig_A, sig_A),
        "AB": (s_A, s_B, sig_A, sig_B),
        "BB": (s_B, s_B, sig_B, sig_B),
    }
    emd_tot = 0.0
    nll_tot = 0.0
    for p in pairings:
        params = cfg[p.name]
        emd, nll = p.evaluate(*params)
        emd_tot += p.weight * emd
        nll_tot += p.weight * nll
    return emd_tot, nll_tot


# --------------------------------------------------------------------------- #
# Scan analysis
# --------------------------------------------------------------------------- #

def parabolic_summary(x, y):
    """Fit parabola near the minimum; return (x_min, curvature d2y/dx2, well_width@1%)."""
    i_min = int(np.argmin(y))
    # local fit using 5 nearest points
    lo, hi = max(0, i_min - 2), min(len(x), i_min + 3)
    xs, ys = x[lo:hi], y[lo:hi]
    if len(xs) < 3:
        return float(x[i_min]), 0.0, float("nan")
    a, b, c = np.polyfit(xs, ys, 2)
    x_min = -b / (2 * a) if a != 0 else float(x[i_min])
    curvature = 2 * a  # d²y/dx²
    # well width at Δy = 1% of (max-min)
    delta = 0.01 * (np.max(y) - np.min(y))
    if a > 0 and delta > 0:
        well_width = 2.0 * np.sqrt(delta / a)
    else:
        well_width = float("nan")
    return float(x_min), float(curvature), float(well_width)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_1d_scans(scans, truth_dict):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("1D loss scans — EMD (blue) vs Poisson NLL·χ² (red)", fontsize=13)
    for ax, (name, data) in zip(axes.flat, scans.items()):
        xs, emd_vals, nll_vals = data["x"], data["emd"], data["nll"]
        ax2 = ax.twinx()
        ax.plot(xs, emd_vals, "b-o", ms=3, label="EMD")
        ax2.plot(xs, nll_vals, "r-s", ms=3, label="NLL·χ²")
        ax.axvline(truth_dict[name], color="k", ls="--", lw=1, alpha=0.6, label=f"truth = {truth_dict[name]:.4f}")
        ax.set_xlabel(name)
        ax.set_ylabel("EMD", color="b")
        ax2.set_ylabel("NLL·χ²", color="r")
        ax.tick_params(axis="y", labelcolor="b")
        ax2.tick_params(axis="y", labelcolor="r")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "scans_1d.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  wrote {out}")


def plot_2d_scan(scan, xlabel, ylabel, truth_x, truth_y, fname):
    X, Y = scan["X"], scan["Y"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, key, title in zip(axes, ("emd", "nll"), ("EMD", "NLL·χ²")):
        Z = scan[key]
        # normalize to (Z - Zmin) for readability
        Zn = Z - Z.min()
        cs = ax.contourf(X, Y, Zn, levels=20, cmap="viridis")
        ax.contour(X, Y, Zn, levels=10, colors="white", linewidths=0.4, alpha=0.6)
        ax.plot(truth_x, truth_y, "rx", ms=12, mew=2, label=f"truth ({truth_x:.4f}, {truth_y:.4f})")
        # mark the discrete grid minimum
        idx = np.unravel_index(np.argmin(Z), Z.shape)
        ax.plot(X[idx], Y[idx], "wo", ms=8, mfc="none", mew=2, label=f"argmin ({X[idx]:.4f}, {Y[idx]:.4f})")
        ax.set_title(f"{title}  (Z - Z_min)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(cs, ax=ax)
    plt.tight_layout()
    out = OUT_DIR / fname
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    rng = np.random.default_rng(RNG_SEED)
    print("[1/4] sampling toy MC and data...")

    # ----- MC (CB-only, no injected scale or smear) -----
    mc_AA = inject_data(sample_pairing(CAT_A, CAT_A, N_EVENTS, rng), 1.0, MC_SMEAR_INJECT, rng)
    mc_AB = inject_mixed_data(sample_pairing(CAT_A, CAT_B, N_EVENTS, rng), 1.0, 1.0,
                              MC_SMEAR_INJECT, MC_SMEAR_INJECT, rng)
    mc_BB = inject_data(sample_pairing(CAT_B, CAT_B, N_EVENTS, rng), 1.0, MC_SMEAR_INJECT, rng)

    # ----- Data: inject the imperfect scales and smears -----
    data_AA = inject_data(sample_pairing(CAT_A, CAT_A, N_EVENTS, rng),
                          INJECT["scale_A"], INJECT["smear_A"], rng)
    data_AB = inject_mixed_data(sample_pairing(CAT_A, CAT_B, N_EVENTS, rng),
                                INJECT["scale_A"], INJECT["scale_B"],
                                INJECT["smear_A"], INJECT["smear_B"], rng)
    data_BB = inject_data(sample_pairing(CAT_B, CAT_B, N_EVENTS, rng),
                          INJECT["scale_B"], INJECT["smear_B"], rng)
    print(f"  INJECT (data): {INJECT}")
    print(f"  TRUTH (= expected optimum): {TRUTH}")

    print("[2/4] building pairings...")
    pairings = [
        make_pairing("AA", W_AA, data_AA, mc_AA),
        make_pairing("AB", W_AB, data_AB, mc_AB),
        make_pairing("BB", W_BB, data_BB, mc_BB),
    ]

    # ---- evaluate at truth as sanity check ----
    e0, n0 = total_loss(pairings, TRUTH["scale_A"], TRUTH["scale_B"],
                                  TRUTH["smear_A"], TRUTH["smear_B"])
    print(f"  truth-point loss: EMD={e0:.4f}  NLL·χ²={n0:.4f}")

    # diagnostic dump: mean/std at truth point vs MC, after applying trial
    print("  diagnostic (data after trial-scale vs mc after trial-smear, AA pairing):")
    p = pairings[0]
    s = np.float32(np.sqrt(TRUTH["scale_A"] * TRUTH["scale_A"]))
    d_t = p.data * s
    m_t = apply_smearing_cached(p.mc, TRUTH["smear_A"], TRUTH["smear_A"],
                                p.randn_lead, p.randn_sublead)
    d_w = d_t[(d_t >= HIST_MIN) & (d_t <= HIST_MAX)]
    m_w = m_t[(m_t >= HIST_MIN) & (m_t <= HIST_MAX)]
    print(f"    AA  data: mean={np.mean(d_w):.4f}  std={np.std(d_w):.4f}  N={len(d_w)}")
    print(f"    AA  mc  : mean={np.mean(m_w):.4f}  std={np.std(m_w):.4f}  N={len(m_w)}")
    # also at "wrong" smear=0
    m0 = p.mc
    m0_w = m0[(m0 >= HIST_MIN) & (m0 <= HIST_MAX)]
    print(f"    AA  mc (σ_trial=0): mean={np.mean(m0_w):.4f}  std={np.std(m0_w):.4f}  N={len(m0_w)}")

    # ---- 1D scans ----
    print("[3/4] running 1D scans...")
    n_pts = 21
    scan_ranges = {
        "scale_A": np.linspace(TRUTH["scale_A"] - 0.020, TRUTH["scale_A"] + 0.020, n_pts),
        "scale_B": np.linspace(TRUTH["scale_B"] - 0.020, TRUTH["scale_B"] + 0.020, n_pts),
        "smear_A": np.linspace(max(0.0, TRUTH["smear_A"] - 0.030), TRUTH["smear_A"] + 0.030, n_pts),
        "smear_B": np.linspace(max(0.0, TRUTH["smear_B"] - 0.030), TRUTH["smear_B"] + 0.030, n_pts),
    }
    scans_1d = {}
    for name, xs in scan_ranges.items():
        emd_vals, nll_vals = [], []
        for v in xs:
            kwargs = dict(s_A=TRUTH["scale_A"], s_B=TRUTH["scale_B"],
                          sig_A=TRUTH["smear_A"], sig_B=TRUTH["smear_B"])
            kwargs[{"scale_A": "s_A", "scale_B": "s_B",
                    "smear_A": "sig_A", "smear_B": "sig_B"}[name]] = float(v)
            e, n = total_loss(pairings, **kwargs)
            emd_vals.append(e)
            nll_vals.append(n)
        scans_1d[name] = dict(x=xs, emd=np.array(emd_vals), nll=np.array(nll_vals))
        print(f"  {name}: EMD argmin x={xs[np.argmin(emd_vals)]:.5f}  NLL argmin x={xs[np.argmin(nll_vals)]:.5f}  truth={TRUTH[name]:.5f}")

    plot_1d_scans(scans_1d, TRUTH)

    # ---- 2D scans ----
    print("[4/4] running 2D scans...")
    n2 = 21

    def run_2d(varx_name, vary_name, x_range, y_range):
        X, Y = np.meshgrid(x_range, y_range)
        EMD = np.zeros_like(X)
        NLL = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                kwargs = dict(s_A=TRUTH["scale_A"], s_B=TRUTH["scale_B"],
                              sig_A=TRUTH["smear_A"], sig_B=TRUTH["smear_B"])
                kwargs[{"scale_A": "s_A", "scale_B": "s_B",
                        "smear_A": "sig_A", "smear_B": "sig_B"}[varx_name]] = float(X[i, j])
                kwargs[{"scale_A": "s_A", "scale_B": "s_B",
                        "smear_A": "sig_A", "smear_B": "sig_B"}[vary_name]] = float(Y[i, j])
                e, n = total_loss(pairings, **kwargs)
                EMD[i, j] = e
                NLL[i, j] = n
        return dict(X=X, Y=Y, emd=EMD, nll=NLL)

    scan_sA_sigA = run_2d("scale_A", "smear_A",
                          np.linspace(TRUTH["scale_A"] - 0.012, TRUTH["scale_A"] + 0.012, n2),
                          np.linspace(max(0.0, TRUTH["smear_A"] - 0.025), TRUTH["smear_A"] + 0.025, n2))
    plot_2d_scan(scan_sA_sigA, "scale_A", "smear_A",
                 TRUTH["scale_A"], TRUTH["smear_A"], "scan_2d_sA_sigA.png")

    scan_sA_sB = run_2d("scale_A", "scale_B",
                        np.linspace(TRUTH["scale_A"] - 0.012, TRUTH["scale_A"] + 0.012, n2),
                        np.linspace(TRUTH["scale_B"] - 0.012, TRUTH["scale_B"] + 0.012, n2))
    plot_2d_scan(scan_sA_sB, "scale_A", "scale_B",
                 TRUTH["scale_A"], TRUTH["scale_B"], "scan_2d_sA_sB.png")

    # ---- summary table ----
    lines = []
    lines.append("EMD vs Poisson-NLL·χ² loss-surface comparison")
    lines.append("=" * 60)
    lines.append(f"N_events/pairing: {N_EVENTS}")
    lines.append(f"Truth: {TRUTH}")
    lines.append(f"Truth-point loss: EMD={e0:.4f}  NLL·χ²={n0:.4f}")
    lines.append("")
    lines.append(f"{'param':<10} {'loss':<6} {'argmin':<10} {'truth':<10} {'bias':<10} {'curvature':<14} {'well_width@1%':<14}")
    lines.append("-" * 80)
    for name, sc in scans_1d.items():
        for label, key in (("EMD", "emd"), ("NLL", "nll")):
            x_min, curv, ww = parabolic_summary(sc["x"], sc[key])
            bias = x_min - TRUTH[name]
            lines.append(
                f"{name:<10} {label:<6} {x_min:<10.6f} {TRUTH[name]:<10.6f} {bias:<+10.6f} {curv:<14.3e} {ww:<14.3e}"
            )
    summary = "\n".join(lines)
    print()
    print(summary)
    out = OUT_DIR / "summary.txt"
    out.write_text(summary + "\n")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
