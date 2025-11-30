• Review of latest code/results:

  What changed

  - Per-symbol chronological split added; series with <1k rows are skipped (helps avoid tiny BUSD/USDC_Kraken leakage).
  - Forecasting labels/targets still at +1h; lag features added; USDT pairs normalized to USD.
  - Anomaly evaluation added with a simple ground-truth rule |deviation_pct|>1%.

  Current results (reports/metrics/modeling_results_predictive.csv)

  - Classification (test set: 17,447 rows): High 6,774 / Medium 10,661 / Low 12. Macro F1 ≈ 0.386, accuracy ≈ 0.58. All 12 Low
    examples (FDUSD only) are misclassified (Low recall = 0). Model predicts only High/Medium.
  - Regression: Ridge RMSE ≈ 0.00185 with R² = -0.57; NN similar (still negative R²). Next-hour deviation forecasting is not
    predictive.
  - Anomalies: 1,745 flags (~2%). True depegs by threshold (|dev|>1%): 1,270; recall ≈ 0.75 (good), precision unknown because
    calculation crashed earlier; contamination fixed at 2% not tuned per symbol/event.
  - Low events in test are extremely sparse (12) and only from FDUSD, so Low performance is effectively unmeasured/failed.

  Gaps vs. intent

  - Severe class imbalance in test: almost no Low cases; High/Medium dominates. Need a validation split that contains meaningful
    depegs (e.g., by holding out known stress windows per coin).
  - No per-symbol anomaly precision/recall or per-symbol Low recall; model still ignores Low.
  - Regression still worse than baseline; features/target horizon likely too tight (+1h).

  Direct next steps to make this useful

  1. Build an event-aware test set: for each symbol, hold out the most volatile 10% window (or known drawdowns) so Low is present
     in test; or oversample/weight Low in training and enforce predictions for Low (e.g., class_weight boosting for Low).
  2. Use a longer horizon and richer features: predict +6h/+24h and include lags/rolling z-scores of deviation/volume/volatility
     and BTC/ETH returns; current +1h signal is too hard.
  3. Tune anomalies per symbol: compute precision/recall against depeg windows for each coin; adjust contamination and maybe use
     deviation velocity/volume spike thresholds to reduce false positives.
  4. Drop or separately analyze coins with almost no depegs; for FDUSD/TUSD etc., explicitly inject their depeg periods into test
     to get measurable Low recall.
  5. Document results with per-class/per-symbol metrics and plots highlighting Low performance; right now the confusion matrix is
     dominated by High/Medium and hides the miss on Low.
