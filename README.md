# scorecard-woe-logistic

Classic credit scorecard on the German Credit dataset: WOE binning, logistic regression, points table, and holdout evaluation.

## Results (holdout, 30% stratified, seed 42)

| Metric | Train | Holdout |
|---|---|---|
| AUC | 0.726 | 0.668 |
| Gini | 0.451 | 0.337 |
| KS | 0.337 | 0.310 |

The 5-fold CV search over `C` (0.01 to 10) picked `C = 1.0`, so the tuned model equals the default baseline.

Notebook: [notebooks/01_scorecard_woe_logistic.ipynb](notebooks/01_scorecard_woe_logistic.ipynb). Figures in [reports/figures/](reports/figures/).

## Method

- `Sex` is a protected attribute. It is not a model feature. It is audited in `calibration-and-fairness` (CR03).
- Numeric features: Age, Credit amount, Duration (quantile bins). Categorical: Job, Housing, Saving accounts, Checking account, Purpose (rare levels grouped as `other`).
- WOE and IV are fitted on the training split only. Features with IV outside 0.02 to 0.5 are dropped.
- Points: 20 PDO, 600 points at 50:1 good odds.

## Limitations

- German Credit has 1,000 rows and no date column, so the split is random, not time-based.
- The `uciml/german-credit` copy has no target, so this project uses the `kabure/german-credit-data-with-risk` copy (target `Risk`).

## Run

```bash
pip install -e ".[dev]"
make test
make notebook
```

Data is read from `CREDIT_DATA_DIR` (default `~/.local/share/credit-data`). See [data/README.md](data/README.md).
