Supplementary data: the supplementary tables as CSV files
==========================================================

Paper: "Uncertainty-aware space-ground sensor fusion for tracking maneuvering
hypersonic targets: A reproducible simulation benchmark"
(A. Al Junayed, L. Ibn Muzahid).
Code and full results: https://github.com/labibibnmuzahid/hypershield-benchmark

Every file is written by make_supplement.py from the released result files.
Values are unrounded. Units: km, s, g = 9.80665 m/s^2. Shares are fractions
(0-1), not percentages. Estimators: EKF-CV, UKF-CV, EKF-CA, IMM-CV/CA.
Sensing: ir (IR only), radar (radar only), fused. Maneuver: low (0.5 g),
medium (2 g), high (4 g); 500 held-out runs per maneuver level.

S1-S3_all_cells.csv  (Tables S1-S3)
    One row per maneuver x sensing x estimator (36 rows).
    OSPA_km, OSPA_CI_low/high   mean OSPA (cut-off 10 km) and 95% bootstrap interval
    custody                     share of epochs with a confirmed track within 1 km
    continuity                  share of epochs with a confirmed track
    segments_per_run            confirmed track segments per run
    TTFT_median_s               median time to first track within 5 km (inf = never)
    ever_tracked                share of runs tracked at least once
    RMSE_km                     position RMSE over epochs with a track
    ANEES, ANEES_CI_low/high    normalized ANEES (ideal 1) and 95% bootstrap interval
    C50, C95                    empirical coverage of the nominal 50% / 95% ellipsoids

S4_paired_estimators.csv  (Table S4)
    All 36 paired estimator comparisons of per-run OSPA (one Holm family).
    delta_OSPA_km = mean(first - second); relative = delta / mean(second);
    share_first_better = share of runs with lower OSPA for the first estimator;
    p_wilcoxon = two-sided Wilcoxon signed-rank p; p_holm = Holm-adjusted p.

S5_paired_fusion.csv  (Table S5)
    All 24 comparisons of fused against single-sensor tracking (one Holm family);
    columns as in S4, with baseline = ir or radar.

S6_credibility_fused_medium.csv  (Table S6)
    Credibility diagnostics, fused configuration, medium maneuvers, per estimator:
    false_window_share_of_epochs, ANEES_after_false, ANEES_otherwise,
    C95_after_false, C95_otherwise, false_window_share_of_NEES,
    false_update_share, median_normalized_NEES, worst5pct_share_of_NEES,
    coverage_<alpha> for alpha = 0.50, 0.68, 0.90, 0.95, 0.99.

S7_ANEES_by_acceleration.csv  (Table S7)
    Normalized ANEES by the target's instantaneous lateral acceleration (g),
    fused configuration, medium maneuvers; share_of_epochs refers to IMM-CV/CA.

S8-S9_sensitivity.csv  (Tables S8-S9)
    One-at-a-time sweeps on 300 medium-maneuver runs (trajectories fixed):
    parameter in {ir_revisit [s], outage_fraction, ir_sigma_urad [rad],
    radar_pd, false_alarms [per scan, both sensors]}; sensing in {ir, fused}.
    Columns OSPA_km, custody, ANEES (empty = undefined), C95, false_update_share.

S10_tuning.csv  (Table S10)
    Process-noise grid search, 200 fused medium-maneuver tuning runs.
    q_v_or_none [km^2/s^3], q_a_or_none [km^2/s^5], p_s (IMM only), selected.

S11_scenario_statistics.csv  (Table S11)
    Scenario statistics per maneuver level (peak lateral acceleration in g,
    altitude in km, visible satellites, outage and radar-coverage shares,
    radar entry time in s).

S12_timing.csv  (Table S12)
    Tracker CPU time per cell (s) and microseconds per filter cycle and run.
