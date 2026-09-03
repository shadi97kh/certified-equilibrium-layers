# Project contract

This file is the authoritative project contract. Every future change MUST
conform to the rules below. If a request conflicts with this file, stop and
surface the conflict before acting.

## Project scope

This repository is the AAAI project: a shared CASCADE core plus four certified
theorems on REAL siRNA data. The core is implemented once and reused; the
experiments never duplicate it.

- **CASCADE core** (`src/cascade/`). An interpretable GNN for siRNA
  seed-mediated off-target prediction. It extends a prior model (CCMP:
  FiLM-conditioned per-concept heterogeneous message passing, K=9 concept
  channels) and preserves CCMP's by-construction zero cross-concept leakage
  property (L_k = 0). The CASCADE forward is: per-concept features to a CCMP
  backbone that predicts per-transcript direct seed repression x0, then a
  certified propagation head that diffuses x0 over the GRN with the
  bounded-alpha operator P_alpha to predict off-target log2 fold-change.

- **AAAI certification** (`src/certification/`). Four theorems on REAL siRNA data
  (no generated data on any reported-number path, see the hard data policy):
  - **T1, leakage-free per-concept identifiability.** Under L_k = 0 each concept
    channel is identifiable in isolation, so its contribution to x0 is recovered
    without cross-channel confounding.
  - **T2, P_alpha contraction / stability on signed, directed GRNs.** The
    bounded-alpha propagation operator P_alpha = alpha (I - (1-alpha) S)^{-1} is
    a contraction (||P_alpha||_2 <= 1, eigenvalues in [alpha/(2-alpha), 1]) and
    is stable to GRN perturbation, including signed and directed edges.
  - **T3, KKT-projected mass-action / thermodynamic constraint.** The
    seed-binding / repression map is projected onto the mass-action and
    thermodynamic feasible set via its KKT conditions, so x0 respects the
    physical binding constraints.
  - **T4, certified off-target effect bound.** A probabilistic certified L2
    region for y = P_alpha(S) x0 under joint seed-error and GRN-edge uncertainty
    (matrix-Bernstein, radius ((1-alpha)/alpha) b(delta) ||x0||).

## CASCADE contributions (priority order)

1. **Per-concept compositional PAC-Bayes generalization bound.** A bound that
   is channel-separated across the K concept channels, plus a theorem that
   L_k = 0 isolation strictly tightens the bound by zeroing the cross-channel
   covariance term.
2. **Certified off-target containment.** The architecture constrains GRN
   propagation to a bounded-alpha personalized-PageRank operator, so predicted
   off-target signal at graph distance r decays at a certified rate. The
   upgrade target for this contribution is an effective-resistance /
   spectral-gap distance bound.
3. **Causal-abstraction consistency corollary.** Under L_k = 0, the seed
   concept channel is a consistent causal abstraction of the propagation
   submodule.

## Hard engineering rules

These rules are non-negotiable. Violating any of them is grounds to reject a
change.

1. **Theorem / verification parity.** Every theorem MUST have a matching
   numerical verification script under `verify/` (top level) that empirically
   confirms the bound on real data. A theorem without a passing verification
   script does not ship.
2. **By-construction containment.** Containment is enforced architecturally
   (bounded teleport alpha, spectral-norm-clipped diffusion). It is NEVER
   enforced via a loss penalty or a soft regularizer.
3. **Seeded determinism.** Every script accepts `--seed` and logs the seed it
   ran with. Default seed values are not silently chosen.
4. **No em-dashes.** Anywhere. Not in code comments, docstrings, README, the
   paper, commit messages, or any other artifact in this repo. Use a comma,
   a colon, parentheses, or split the sentence.
5. **Reproducibility.** A single `make repro` reproduces all headline numbers
   from cached data without network access.
6. **Commit cadence.** Commit after each completed unit of work. Use
   Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`,
   `chore:`, `build:`, `ci:`).
7. **Output discipline.** Experiment scripts and agents emit results as JSON
   under `results/` and figures as PNG or SVG. They NEVER produce LaTeX, `.tex`,
   or paper prose. A plain markdown summary is allowed. The `paper/` directory
   is authored by humans; agents do not write there.
8. **Shared-core reuse.** The AAAI experiments import the core from
   `src/cascade/`. They MUST reuse the P_alpha operator, the data loaders, and
   the graph builders, and MUST NOT fork or re-implement them.

## Hard data policy (AAAI reported numbers)

The AAAI paper reports its four theorems on REAL siRNA data. Generated data may
NEVER sit on the path of a reported AAAI number. This policy is non-negotiable
and is enforced mechanically.

1. **No generated data on a reported-number path.** No generated data may live
   in `src/cascade/data/` (the loaders) or `experiments/aaai/` (the experiments
   and the aggregator). This means no random construction of features, labels,
   or edges; no signal injection; no fabricated edge confidences; no fallback
   that stands in for absent real data.
2. **Loaders raise, never fabricate.** Every loader RAISES `FileNotFoundError`
   with a clear message when its real file is absent. A real loader that is not
   yet wired RAISES `NotImplementedError` (it does not return a stand-in). No
   experiment produces a number without the real data behind it.
3. **Mechanical enforcement.** `scripts/assert_no_synthetic.py` greps
   `src/cascade/data/` and `experiments/aaai/` for generation patterns
   (`np.random` / `default_rng` / `standard_normal` / `rng.*` construction, and
   the words `synthetic`, `plant`, `proxy`, `toy`) and exits non-zero on any
   hit. It runs in CI and is a precondition of `make repro-aaai`. The only
   exemptions are committed unit-test fixtures under `tests/` and a line marked
   `# real-data-policy: allow`.
4. **Where generated data is allowed.** Certificate-soundness verification
   (coverage, tightness, the C5 breakdown) legitimately runs on random-graph
   ensembles: it verifies the math, not biology, and is never a reported
   biological number. It lives under `experiments/soundness/` (NOT scanned) and
   its results are labeled "random-graph verification, no biological claim".
5. **Tracked data gaps stay visible.** When real data (GSE184929 / STRING join,
   OligoFormer CSVs, microarray supplements) is absent, the affected AAAI
   sections are reported as PENDING real data, never backfilled with a stand-in.

## Stack

Python 3.11. Pinned in `pyproject.toml`.

- PyTorch, PyTorch Geometric
- numpy, scipy, pandas, scikit-learn
- ViennaRNA (Python bindings)
- pytest for tests
- ruff for lint and format

The certification package is pure numpy / scipy (no torch) so it runs in the
lightweight CI image. Only the CASCADE model and the experiment training loops
require torch; tests that need it are gated with `pytest.importorskip("torch")`.

## Repository layout

```
src/cascade/          shared core (REUSE, never duplicate)
  data/               REAL-data loaders + caching: STRING, TRRUST, GSE184929, OligoFormer, off-target
  features/           seed-match (SeedMatchR logic) + thermodynamics + per-concept features
  models/             ccmp_backbone, propagation_head (P_alpha), cascade
  verify/             containment, pac_bayes, abstraction audits (library)
  eval/               metrics + figures
src/certification/    AAAI: probabilistic certified L2 region
experiments/aaai/     REAL-data experiments (exp_baselines, exp_t1..t4, export_paper_numbers)
experiments/soundness/  certificate-soundness on random graphs (math only; NOT scanned by the policy)
results/aaai/         AAAI JSON + figures
tests/                pytest
verify/               runnable theorem-verification scripts (rule 1)
scripts/              data download + assert_no_synthetic.py (hard data policy enforcer)
paper/                human-authored LaTeX (agents never write here)
```

The split between `src/cascade/verify/` and top-level `verify/` is deliberate:
the former holds library code (audits, estimators), the latter holds the
runnable scripts that each correspond to a theorem.

## Shared core contract

The AAAI experiments depend on these imports from `src/cascade/`. They are kept
stable; breaking them is a contract change.

Importable and implemented:

- **P_alpha operator.** `cel.models.propagation_head.PropagationHead`
  (torch) and `cel.verify.containment.propagate` /
  `cel.verify.containment.dense_propagation_operator` (numpy).
- **STRING graph builder.** `cel.data.load_string_grn`.
- **GSE184929 off-target loader** (SeedMatchR seed logic).
  `cel.data.load_gse184929` with `cel.features.seedmatch`.

Tracked gaps. These are stubbed and each has a visible failing (xfail-strict)
gap test in `tests/test_shared_core.py`. When a gap is implemented, the test
flips to XPASS and the strict marker forces removal, so the gap cannot close
silently.

- **Reactome graph builder.** `cel.data.reactome.load_reactome_graph`
  raises `NotImplementedError`. TODO: parse the Reactome functional-interaction
  / pathway graph into the same symmetric-normalized sparse `S` form as STRING.

## Workflow

- Experiment scripts are argparse CLIs that accept `--seed` / `--seeds` and log
  the seed. No hard-coded paths.
- Tests live next to behavior, not next to code. Put unit tests in `tests/`.
- Lint and tests run in CI on every push (see `.github/workflows/ci.yml`).
- Pre-commit hooks run ruff. Install with `pre-commit install`.
