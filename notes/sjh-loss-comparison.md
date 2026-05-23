# ecal-ss-sjh — EMD vs Poisson-NLL·χ² loss-surface comparison

Empirical comparison of the two candidate loss functions used in
[python/utilities/minimizer.py](../python/utilities/minimizer.py)
(`compute_earthmovers_distance` vs `compute_nll_chisqr`) on a controlled toy
problem. Experiment lives in
[scripts/experiments/emd_vs_poisson.py](../scripts/experiments/emd_vs_poisson.py)
and writes plots + summary to `scripts/experiments/output/` (gitignored).

## Setup

- Two Voigt-like categories (`BW_CB ⊗ BW_CB`) labelled `A` (EB-like, narrow)
  and `B` (EE-like, wider), three pairings `AA / AB / BB`, 100k events each.
- "Data" injected with imperfect scales (1.010 / 0.995) and per-electron
  smearings (0.040 / 0.060). "MC" generated at perfect scale (1.000) and zero
  smearing — the fitter must recover scales by undoing the data scale and
  smearings via `apply_smearing_cached` on MC.
- Truth values for the trial parameters are therefore
  `s_X = 1/inject_scale_X` and `σ_X = sqrt(inject_smear_X² − mc_smear_X²)`.
- Off-diagonal pairing weight 0.1 (matches production post-e31), uniform EMD
  weights (matches production post-9bi), Freedman–Diaconis binning (matches
  zcat).

## 1D scans (`scans_1d.png`)

| param   | loss | argmin     | truth      | bias       |
| ------- | ---- | ---------- | ---------- | ---------- |
| scale_A | EMD  | 0.990137   | 0.990099   | +0.000038  |
| scale_A | NLL  | 0.989896   | 0.990099   | −0.000203  |
| scale_B | EMD  | 1.003271   | 1.005025   | −0.001754  |
| scale_B | NLL  | 1.004139   | 1.005025   | −0.000886  |
| smear_A | EMD  | 0.038983   | 0.040000   | −0.001017  |
| smear_A | NLL  | **0.013474** | 0.040000 | **−0.026526** |
| smear_B | EMD  | 0.054270   | 0.060000   | −0.005730  |
| smear_B | NLL  | 0.043720   | 0.060000   | −0.016280  |

- **Scales:** both losses are well-behaved, smooth, and recover truth within
  ~0.2% with a single category fixed. NLL has slightly smaller bias on the
  scales here.
- **Smears:** EMD recovers smear_A within 0.001 of truth and smear_B within
  0.006. NLL produces a **noticeably biased and noisy** scan in both smear
  parameters — bin-to-bin fluctuations in the NLL curve are 10–20% of the
  total scan range, and on smear_A its argmin lands at 0.013 (a stray local
  minimum) instead of the truth value 0.040.

## 2D scans

- `scan_2d_sA_sigA.png` (scale_A × smear_A): EMD shows a clean unimodal
  banana-shaped well centred near truth. NLL·χ² shows **multiple local
  minima** along the smear axis — the global argmin is at smear_A=0.015 (deep
  blue lobe at the bottom) while a second comparable basin exists near the
  truth value. A naive minimiser would converge to the wrong basin.
- `scan_2d_sA_sB.png` (scale_A × scale_B): both losses are unimodal and
  centred near truth. EMD's contours are sharper (diamond-like); NLL's are
  rounder and noisier but qualitatively correct.

## Verdict

For the scale dimension EMD and NLL·χ² behave comparably, but along the
**smear axis** EMD is dramatically smoother and more accurate at this
statistic level (100k events/pairing). NLL·χ² is noise-dominated in σ at
fixed N, producing spurious local minima that could trap a gradient-based
minimiser. The 2D `(s, σ)` surface confirms this: NLL has a bimodal landscape
where EMD has a single basin.

**Recommendation:** keep `compute_earthmovers_distance` as the production
loss for this problem. The Poisson-NLL·χ² implementation should not be
promoted to default; its scan is acceptable for scales but unreliable for
smearings at the per-category statistics typical in production.

The asymmetric sensitivity of EMD on the smear axis (sharp rise above truth,
gentler approach from below) is a real feature of the loss, not a noise
artifact — see `scans_1d.png`, bottom row.

## Reproduce

```bash
python -m scripts.experiments.emd_vs_poisson
# → scripts/experiments/output/{scans_1d.png, scan_2d_sA_sigA.png, scan_2d_sA_sB.png, summary.txt}
```
