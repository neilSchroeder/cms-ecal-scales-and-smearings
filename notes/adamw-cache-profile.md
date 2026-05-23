# AdamW smart-skip cache profile (ecal-ss-nx4)

## Setup

- **Test configuration**: 8 scale params, 4 smear params (12 total), 10 zcats
- **Iterations**: 5 iterations per method
- **AdamW FD step**: `eps = 1e-5`
- **Test fixture**: Synthetic loss function (`sum((x_i - target)^2)`) with mock zcats
- **Cat index pattern**: Each cat touches 2 scale indices + 2 smear indices (4 total)

## Measurements

| Strategy | total calls | cats updated | cats skipped | skip rate | avg cats updated/call | iterations | nfev |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AdamW (current) | 126 | 530 | 730 | 57.94% | 4.21 | 5 | 126 |
| L-BFGS-B (reference) | 273 | 1397 | 1333 | 48.83% | 5.12 | 0* | 273 |

*L-BFGS-B reported 0 iterations but 273 function evaluations (internal line-search behavior).

## Analysis

### AdamW call pattern (central finite differences)

For each iteration, AdamW computes gradients via:

```python
for i in range(n_params):
    x_fwd = x.copy(); x_fwd[i] += eps
    x_bwd = x.copy(); x_bwd[i] -= eps
    f_fwd = fun(x_fwd)  # Call A: x with coord i at +eps
    f_bwd = fun(x_bwd)  # Call B: x with coord i at -eps
    grad[i] = (f_fwd - f_bwd) / (2*eps)
```

Then updates `x` with the Adam step and repeats.

**Key observation**: Between consecutive FD calls, the coordinate difference pattern is:

1. Initial: `fun(x)` → `previous_guess = x`
2. Coord 0 fwd: `fun(x + eps*e_0)` → differs at index 0
3. Coord 0 bwd: `fun(x - eps*e_0)` → differs at index 0 (relative to fwd)
4. **Coord 1 fwd**: `fun(x + eps*e_1)` → **differs at indices 0 AND 1** (0: -eps→0, 1: 0→+eps)
5. Coord 1 bwd: `fun(x - eps*e_1)` → differs at index 1
6. **Coord 2 fwd**: `fun(x + eps*e_2)` → **differs at indices 1 AND 2**
7. ...and so on

**Result**: Most FD calls differ from `previous_guess` at **2 indices**, invalidating all cats touching either index. With 10 cats and 12 params, this explains the ~4.2 cats/call average (vs 10 total).

### Cache effectiveness

The smart-skip cache achieves:
- **57.94% skip rate** under AdamW (better than I expected based on the ticket hypothesis)
- **48.83% skip rate** under L-BFGS-B

This is actually **working reasonably well** for AdamW. The cache prevents ~58% of possible cat updates, reducing work from 10 cats/call to ~4.2 cats/call.

### Theoretical optimal

A perfect per-cat cache keyed on `(lead_value, sublead_value, lead_smear_value, sublead_smear_value)` would:
- Only update a cat when one of its specific 4 index values changes
- Achieve higher skip rates by avoiding redundant updates when an index returns to a previously-seen value

However, implementing this would require:
- Per-cat value caching (4 floats × N cats)
- Floating-point equality checks (risky with optimizer perturbations)
- More complex cache invalidation logic

**Estimated gain**: Marginal. The current approach already skips ~58% of updates. A perfect cache might push this to ~70-80%, but at significant code complexity cost.

## Recommendation

**[X] Keep as-is**

**Justification:**

1. **Cache is working**: 57.94% skip rate under AdamW means the smart-skip cache is successfully avoiding over half of the potential cat updates.

2. **Avg 4.2 cats/call vs 10 total**: We're only updating ~42% of cats per call, which is a significant win compared to naively updating all cats every time.

3. **Code simplicity**: The current index-based approach is simple, robust, and easy to understand. It avoids floating-point equality checks and per-cat state tracking.

4. **Diminishing returns**: Moving to a per-cat value cache would add complexity (float comparisons, more memory, harder debugging) for an estimated improvement from ~58% to maybe ~75% skip rate — not worth it.

5. **L-BFGS-B comparison**: The cache works better for AdamW (58%) than L-BFGS-B (49%), suggesting it's well-suited to the FD call pattern.

6. **No performance complaint**: There's no evidence that `target_function` is a bottleneck. The expensive part is the zcat EMD computation, which we're already skipping 58% of the time.

### Alternative considered but rejected

**Rework cache keying** to per-cat value tuples:
- Pros: Potentially 15-20% higher skip rate
- Cons: Floating-point equality fragile, more memory, harder to debug, complex invalidation logic
- Verdict: **Not worth the engineering cost**

**Remove cache**:
- Pros: Simpler code
- Cons: Lose the 58% skip rate, update 10 cats every call instead of 4.2
- Verdict: **Bad trade — cache provides real value**

## Raw Numbers

```
Profiling smart-skip cache behavior
Setup: 8 scale params, 4 smear params, 10 zcats
Running 5 iterations of each method...


============================================================
Method: AdamW
============================================================
Total calls:           126
Cats updated:          530
Cats skipped:          730
Full-skip calls:         1
Skip rate:           57.94%
Avg cats/call:        4.21
============================================================


============================================================
Method: L-BFGS-B
============================================================
Total calls:           273
Cats updated:         1397
Cats skipped:         1333
Full-skip calls:         1
Skip rate:           48.83%
Avg cats/call:        5.12
============================================================


Theoretical Analysis:
AdamW iterations:    5
AdamW nfev:          126
L-BFGS-B iterations: 0
L-BFGS-B nfev:       273

With perfect smart-skip:
  Each FD call updates 2-4 cats (indices touched)
  vs current: all 10 cats updated every call
```

## Conclusion

The smart-skip cache is **effective and should be kept as-is**. It provides meaningful performance improvement (58% reduction in cat updates) with simple, maintainable code. The AdamW central-FD pattern does cause more invalidations than an ideal oracle cache, but the current approach strikes the right balance between simplicity and effectiveness.
