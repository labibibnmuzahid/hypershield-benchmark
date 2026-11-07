# HyperShield benchmark: space–ground sensor fusion for tracking maneuvering hypersonic glide targets

This is the simulation code behind *"Uncertainty-aware space–ground sensor fusion for tracking maneuvering hypersonic targets: A reproducible simulation benchmark"* (A. Al Junayed, L. Ibn Muzahid). It generates every table, figure and number in the paper from fixed seeds.

**Scope.** The benchmark covers kinematic observation, multi-sensor state estimation and how credible the reported uncertainty is. It has no engagement, fire-control, interceptor or guidance model, which matches the paper's stated scope (Sections 1 and 8.4).

## What it simulates

| Paper section | Module | Contents |
|---|---|---|
| 3, Table 1, Fig. 1 | `hsbench/observability.py` | Stagnation temperature, Wien peaks, plasma critical density, 4/3-Earth radar line of sight, LEO horizon, warning time |
| 4.1 | `hsbench/trajectory.py` | Mach 5–8 glide trajectories on a spherical Earth: skip-like altitude profile, piecewise-constant lateral command with a 4 s lag, heading hold, 0.05 s Euler integration, 1 Hz output |
| 4.2 | `hsbench/sensors.py` | Walker 48/6/1 constellation at 1000 km and 55°. Visibility (Earth occlusion, under 4500 km), K = 3 greedy D-optimal tasking (Eq. 7), nadir-frame angle measurements (Eq. 8), Pd 0.9, Markov outages (20%, mean 25 s), Poisson clutter in a ±2 mrad box |
| 4.3 | `hsbench/sensors.py` | Ground radar at 30° N 0° E with a 10 m mast: range/azimuth/elevation, 4/3-Earth horizon and under 1500 km, Pd 0.9, Poisson clutter in a ±3 km × ±6 mrad × ±6 mrad box |
| 5.1–5.2 | `hsbench/filters.py` | CV and CA models (Eqs. 9–10). EKF-CV, UKF-CV (2n points, α=1, β=0, κ=0, which is the cubature rule), EKF-CA, and a two-model IMM in nine-state space with the CV mode resetting acceleration. Joseph-form updates |
| 5.3–5.4 | `hsbench/tracker.py` | χ²(0.999) gating, nearest-neighbour association, sequential IR-then-radar updates. Radar or IR-stereo initiation (closest approach, under 2 km, 5–120 km altitude, 3 Gauss–Newton iterations, inverse-Fisher covariance). Confirmation after 3 hits in 10 s. Deletion on a 20 s coast, √tr(P_pos) > 30 km, or association ratio < 0.4 |
| 6 | `hsbench/metrics.py` | OSPA (c = 10 km), custody, continuity, TTFT, RMSE, NEES/ANEES, coverage C_α, run-level bootstrap (2000 resamples), Wilcoxon signed-rank, Holm correction |
| 6–7 | `run_experiments.py` | Tuning (seed 11, 200 runs), main evaluation (seeds 221–223, 500 runs per level, 36 cells = 18,000 tracker runs), sensitivity sweeps (seed 33, 300 runs), illustrative example (seed 44) |

Every Monte Carlo run is vectorised: one time loop advances all runs of a cell together. Detections are generated once, before any tracker runs, so all estimators and sensing configurations see identical data (common random numbers). The sweeps reuse the same trajectories and the same underlying random draws, so only the parameter being swept changes.

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q tests                 # 31 verification tests, ~10 s
python run_experiments.py all             # tune -> main -> sweeps -> example, ~6 min on 2 cores
python analysis_tables.py                 # tables + results/key_numbers.json
python analysis_conditional.py            # credibility diagnostics (Section 7.4)
python make_figs.py                       # figures/Figure_1..9 (.png 300 dpi + .pdf)
```

`python run_experiments.py all --quick` runs a smoke test with small run counts. The main evaluation reads the process-noise values picked by the tuning step from `results/tuned.json`. Nothing is hard-coded from the paper.

## Outputs

- `results/main/*.npz`: per-epoch error, NEES, confirmation and false-association flags for every cell
- `results/tables/`: Table 5 (`table5_medium.md/.tex`), Table 6 (`table6_paired.md`), Table A.1 (`tableA1_tuning.md`), all cells with confidence intervals (`summary_all_cells.csv`), every paired test with Holm-adjusted p-values (`paired_all.json`)
- `results/key_numbers.json`, `results/conditional.json`, `results/sweeps.json`: every number quoted in the text
- `results/defaults.json`: the complete configuration of the main run
- `figures/Figure_1` … `Figure_9`: ready to upload as separate files

## Verification (`tests/`)

- Section 3 values (596/727/1023 km, 3707 km, 1372/2469/3155 K, Wien peaks, critical densities, warning times)
- Analytic IR and radar Jacobians against finite differences; the polar-to-Cartesian inverse
- Walker geometry; Q matrices against numerical integration of the continuous-time model
- EKF and UKF against the textbook Kalman filter on a linear model; UKF points equal the cubature rule
- ANEES ≈ 1 for EKF-CV, UKF-CV and the IMM on matched models
- Outage-chain fraction and mean duration; detection and clutter rates; common random numbers across sweeps
- Noise-free stereo triangulation; reproducibility of datasets and trackers; OSPA, Holm and bootstrap checks

## Choices the paper's text leaves open

These are the interpretations made where the manuscript does not specify an implementation detail. Each one is visible in the code.

1. **Heading-hold reference ψ_ref.** The heading of the initial great circle at the vehicle's current position.
2. **Altitude phase φ.** Drawn from U[0, 2π).
3. **Tasking.** Uses the true target position, because detections are generated before any tracker runs.
4. **Outage chain.** Steps once per second. During an outage no IR satellite produces any detection, clutter included.
5. **Radar clutter.** Generated only while the target is inside radar coverage, centred on its true return.
6. **Initiation.** The initiating detection counts as the first of the three confirmation hits. The initiation cycle performs no further update.
7. **Association ratio.** Starts at 1 at initiation. The √tr(P_pos) deletion rule also applies to tentative tracks. The association-ratio rule applies to confirmed tracks only.
8. **False-association flag.** A track started from a clutter detection counts as a false association at its start.
9. **IMM details.** Initial mode probabilities are 0.5/0.5, and the CV mode's acceleration block carries a variance of 1e-12.
10. **Estimator test family.** The 36 comparisons are {IMM vs EKF-CV, IMM vs UKF-CV, IMM vs EKF-CA, UKF-CV vs EKF-CV} × 3 sensing configurations × 3 maneuver levels.
11. **TTFT median.** Runs that are never tracked count as infinite.
12. **Sweep grids.** Revisit {1, 2, 5, 10} s; outage {0, 10, 20, 35, 50}%; angle noise {50, 100, 150, 200} µrad; radar Pd {0.5, …, 0.9}; false alarms {0, 0.5, 1, 2, 3} per scan, applied to both sensors.
13. **Example run.** The first of 50 seed-44 medium runs that enters radar coverage between 250 and 350 s with at least three IR outages before entry.

## Licence and citation

Code, data and figures are released under CC BY 4.0 (see `LICENSE`). If you use the benchmark, please cite it as described in `CITATION.cff`.

## Software

Python 3.13, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2. NumPy `SeedSequence` entropy values are in `hsbench/config.py` (`SEEDS`).
