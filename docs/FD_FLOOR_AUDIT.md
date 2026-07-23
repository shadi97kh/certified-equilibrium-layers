# FD Floor Audit: RNAi closed-form vs finite-difference verification

Scope: the code producing the per-microRNA "Max rel. err." column of the
theory-verification table (McGeary GSE140217/GSE140218 HEK293, 17 microRNAs,
reported values 8.396e-08 to 2.075e-06). This is the RNAi single-pool
comparative-static check, NOT the IGNN gate in /home/shadi/certified_ignn.
Read-only audit; no experiment was re-run.

## STEP 1: Location of the verification code

Single function computing the reported max relative error:
`verify_mirna` in
[scripts/theory_realdata_verification.py](../scripts/theory_realdata_verification.py).

Full source (scripts/theory_realdata_verification.py:86-119):

```python
86  def verify_mirna(K, x, M, occ_name):
87      """Verify the certified ceRNA comparative static (closed-form vs numerical) for one miRNA."""
88      occ = OCC[occ_name]
89      f0 = solve_f(M, x, K) if occ_name == "langmuir" else solve_general(M, x, K, occ)
90      th0, dth0 = occ(f0, K)
91      D = 1.0 + float(np.sum(x * dth0))  # reduced residual Jacobian (1x1); D > 0 => M-matrix
92
93      # Perturb the single highest-abundance competitor s; measure de-repression of all partners.
94      s = int(np.argmax(x))
95      eps = 1e-5 * max(x[s], 1.0)
96      x2 = x.copy()
97      x2[s] += eps
98      f1 = solve_f(M, x2, K) if occ_name == "langmuir" else solve_general(M, x2, K, occ)
99      th1, _ = occ(f1, K)
100     dtheta_num = (th1 - th0) / eps  # numerical dtheta_j/dx_s
101     dfdx = -th0[s] / D  # closed form df/dx_s
102     dtheta_cf = dth0 * dfdx  # closed form dtheta_j/dx_s
103
104     partners = np.arange(len(K)) != s
105     certified_sign = bool(np.all(dtheta_num[partners] <= 1e-15))  # all partners de-repressed
106     m_matrix = bool(D > 0.0)  # reduced Jacobian M-matrix
107     # closed form matches numerical (relative error on the response vector)
108     denom = np.abs(dtheta_cf) + 1e-30
109     max_rel = float(np.max(np.abs(dtheta_num - dtheta_cf) / denom))
110     sign_match = bool(np.all(np.sign(dtheta_num[partners]) == np.sign(dtheta_cf[partners])))
111     return {
112         "n_targets": int(len(K)),
113         "f": f0,
114         "D": D,
115         "m_matrix": m_matrix,
116         "certified_sign": certified_sign,
117         "closed_form_sign_match": sign_match,
118         "max_rel_err_cf_vs_num": max_rel,
119         "mean_derepression": float(np.mean(dtheta_num[partners])),
```

Supporting definitions in the same file:
- `solve_general` (scripts/theory_realdata_verification.py:69): `def solve_general(M, x, K, occ, tol=1e-13, max_iter=200)`.
- `solve_f` is imported from an external module
  (scripts/theory_realdata_verification.py:43,
  `from offtarget_phase_probe import solve_f`); its signature is
  `def solve_f(M, x, K, tol: float = 1e-14, max_iter: int = 200)`
  (/home/shadi/mirna_baselines/scripts/offtarget_phase_probe.py:31).

The reported 17-microRNA table is the `langmuir` occupancy, so the equilibrium
re-solve on both the base and perturbed point goes through `solve_f`.

## STEP 2: Verbatim report

a) Magnitude floor / resolvable mask applied to the closed-form values before
   division: **ABSENT.** There is no resolvable-magnitude mask (no
   `cf.abs() > threshold`, no drop of small-magnitude entries). The only literal
   applied before the division is an **additive** term in the denominator
   (line 108): `denom = np.abs(dtheta_cf) + 1e-30`. `1e-30` is a
   division-by-zero guard, not a magnitude threshold that excludes any entry.

b) Finite-difference step size h (line 95): `eps = 1e-5 * max(x[s], 1.0)`,
   where `s = int(np.argmax(x))` (line 94, the single highest-abundance
   competitor). The difference is **forward / one-sided**: `x2[s] += eps`
   (line 97) and `dtheta_num = (th1 - th0) / eps` (line 100). Not central.

c) Equilibrium re-solve tolerance passed to the solver:
   - langmuir path (the reported table): `solve_f(...)` default `tol = 1e-14`
     (offtarget_phase_probe.py:31). `solve_f` is called with no tol override
     (lines 89, 98).
   - other occupancies: `solve_general(..., tol=1e-13)` default (line 69).
   The solver is a guarded/bisection Newton on the scalar monotone residual, not
   a separate bisection stage; the tolerance above is the only re-solve
   tolerance in the path.

d) Form of the relative error (lines 108-109):
   `np.abs(dtheta_num - dtheta_cf) / (np.abs(dtheta_cf) + 1e-30)`, i.e.
   `|fd - cf| / (|cf| + 1e-30)`. This is `|cf - fd| / |cf|` with an **additive
   1e-30 guard** on the denominator. It is NOT `/ max(|cf|, eps)` and NOT a
   masked ratio.

e) Entries dropped / masked / nan-filtered anywhere in the path:
   - `partners = np.arange(len(K)) != s` (line 104) excludes the perturbed
     self-coordinate `s`, but only for `certified_sign` (105), `sign_match`
     (110), and `mean_derepression` (119).
   - The reported `max_rel` (line 109) is taken over the **full** response
     vector `dtheta_num`/`dtheta_cf` (all `len(K)` coordinates, including the
     self-coordinate `s`); no partner mask, no magnitude mask.
   - No `nan` filtering anywhere in the function.
   - Upstream corpus filtering (restriction to canonical site types and a
     minimum target count) occurs in the driver, not inside this metric; it does
     not drop individual response-vector entries from the max_rel computation.

## STEP 3: Empirical magnitude distribution of |closed-form| sensitivities

**Cannot determine without re-run.**

The existing artifact
[results/theory_realdata_verification.json](../results/theory_realdata_verification.json)
stores, per microRNA (`per_mirna_langmuir`, 17 entries), only scalar summaries:
`n_targets`, `f`, `D`, `m_matrix`, `certified_sign`, `closed_form_sign_match`,
`max_rel_err_cf_vs_num`, `mean_derepression`. A structural scan confirms there
is no list/array anywhere in the JSON payload. The per-entry closed-form
sensitivities `dtheta_cf` (`|dtheta_j/dx_s|` for each transcript j) are computed
inside `verify_mirna` and never persisted.

Therefore the requested distribution of `|closed-form|` across the 17
microRNAs (min, 1st percentile, median, max, and counts/fractions below 1e-3,
1e-5, 1e-7, 1e-9) cannot be read from existing artifacts. Per the task
constraint, no re-run was performed to reconstruct it.

## STEP 4: Consistency check

- Because 2(a) is an **additive 1e-30 denominator guard** and not a masking
  floor, it excludes zero entries. The question "would floor X exclude every
  entry?" does not apply: there is no magnitude mask in this code path. The
  guard is trivially compatible with the reported per-microRNA maxima
  (8.396e-08 to 2.075e-06): those maxima are `|fd - cf| / |cf|` ratios, so the
  underlying `|cf|` values are resolved far above 1e-30 and the +1e-30 term is
  numerically inert at the reported magnitudes.

- Smallest `|closed-form|` magnitude and its relation to the finite-difference
  resolution floor: cannot be stated exactly, because the per-entry `|cf|`
  values are not stored (STEP 3). Structurally, the difference here is
  **forward** (one-sided), so its resolution floor is `~ solver_tol / h`, not
  the central-difference `tol / (2h)`. With `solver_tol = 1e-14` and
  `h = eps = 1e-5 * max(x[s], 1.0) >= 1e-5`, that floor is at most
  `1e-14 / 1e-5 = 1e-9`, and smaller when `x[s] > 1`. The reported max relative
  errors (~1e-8 to ~2e-6) are consistent with forward-difference truncation
  error O(h) dominating and sitting above this solver-limited floor, rather than
  with float64 round-off. This is a structural bound only; confirming that the
  smallest actual `|cf|` sits above the floor requires the per-entry data and
  hence a re-run.

## Summary

The RNAi verification applies **no magnitude floor / resolvable mask** before
computing relative error. The only literal near the division is an additive
`1e-30` denominator guard (line 108). The metric is
`|fd - cf| / (|cf| + 1e-30)` over the full response vector, using a **forward**
difference with step `eps = 1e-5 * max(x[argmax(x)], 1.0)` and equilibrium
re-solve tolerance `1e-14` (langmuir `solve_f`). This differs from the IGNN
gate's `cf.abs() > 1e-3` resolvable mask; no such 1e-3 (or any) threshold exists
here, and none was carried over. The per-entry `|closed-form|` magnitude
distribution is not recoverable from existing artifacts: cannot determine
without re-run.

---

# Addendum: gradient check for the TRAINABLE equilibrium layer

Scope: the gradient check reporting relative L2 error ~2.1e-07 across
lsy6 / mir139 / mir144. This is a **different** function from `verify_mirna`
(already audited above). Read-only; nothing was re-run.

## Location

`gradient_gate` in
[scripts/learned_layer.py](../scripts/learned_layer.py), lines 222-275. It checks
the custom `torch.autograd.Function` IFT backward `dL/dK` (L = sum_s theta_s)
against a finite-difference reference on real per-microRNA K.

Relevant source (scripts/learned_layer.py:245-266):

```python
245         h = 1e-4
246
247         def cdiff(s: int, hh: float, kb=kbase, xn=xnp, mm=m) -> float:
248             kp = kb.copy()
249             kp[s] += hh
250             km = kb.copy()
251             km[s] -= hh
252             fp = solve_f(mm, xn, kp)
253             fm = solve_f(mm, xn, km)
254             return (float(np.sum(fp / (kp + fp))) - float(np.sum(fm / (km + fm)))) / (2 * hh)
255
256         a = np.empty(len(coords))
257         n_fd = np.empty(len(coords))
258         for i, s in enumerate(coords):
259             n_fd[i] = (
260                 4.0 * cdiff(s, h) - cdiff(s, 2.0 * h)
261             ) / 3.0  # Richardson-extrapolated (O(h^4))
262             a[i] = g_auto[s]
263         rel_l2 = float(np.linalg.norm(a - n_fd) / (np.linalg.norm(n_fd) + 1e-30))
264         floor = 1e-6 * float(np.max(np.abs(n_fd)))
265         max_elt = float(np.max(np.abs(a - n_fd) / (np.abs(n_fd) + floor)))
266         passed = rel_l2 < 1e-5
```

## Verbatim answers

a) One-sided or two-sided? **Two-sided (central).** `cdiff` forms both
   `kp[s] += hh` and `km[s] -= hh` and returns
   `(sum(fp/(kp+fp)) - sum(fm/(km+fm))) / (2 * hh)` (lines 248-254).

b) Richardson extrapolation applied, and at what order? **Yes.** The reference
   derivative is not a single central difference; it is the Richardson
   combination `n_fd = (4.0 * cdiff(s, h) - cdiff(s, 2.0 * h)) / 3.0`
   (lines 259-261). This is one level of Richardson extrapolation on a
   second-order central difference: it cancels the leading O(h^2) term, giving
   **fourth-order, O(h^4)** accuracy (matching the inline comment).

c) Step size h: **`h = 1e-4`** (line 245). The Richardson combination evaluates
   the central difference at `h = 1e-4` and at `2*h = 2e-4` (line 260).

d) Exact relative-error formula (line 263):
   `rel_l2 = np.linalg.norm(a - n_fd) / (np.linalg.norm(n_fd) + 1e-30)`, i.e.
   `||g_auto - g_fd||_2 / (||g_fd||_2 + 1e-30)` — a **vector** L2 relative error
   over the sampled coordinates, where `g_auto` is the custom-IFT gradient and
   `g_fd` is the Richardson-extrapolated reference. (Line 265 additionally
   reports a floored elementwise max for transparency; the gate/reported metric
   is `rel_l2`.)

## Is "central finite difference" an accurate description?

**Partially, and incomplete.** The differencing stencil is genuinely central
(two-sided), so "central finite difference" correctly describes the *direction*
of the difference. But the reference derivative actually compared against the
custom backward is **not** a single central difference: it is a
Richardson-extrapolated combination of two central differences (at h and 2h),
which is fourth-order O(h^4) rather than the second-order O(h^2) of a plain
central difference. A precise description is "Richardson-extrapolated central
finite difference." Calling it simply "central finite difference" understates
the reference scheme's accuracy but is not wrong about the stencil being
two-sided/central.

