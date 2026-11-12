# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CR01 - Scorecard with WOE and logistic regression (German Credit)
#
# Goal: build a classic credit scorecard. WOE binning, logistic regression, points table.
# Compare a default `C` with a `C` tuned by 5-fold CV. Report discrimination on a holdout.
#
# Data: German Credit (`kabure/german-credit-data-with-risk`), 1,000 rows. The target
# is `Risk` (1 = good). Split is stratified and random (no date column in this dataset).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from credit_core.binning import make_numeric_edges
from credit_core.datasets import load_german
from credit_core.metrics import decile_table
from credit_core.plots import plot_ks, plot_roc, plot_score_distribution
from scorecard_woe.pipeline import CATEGORICAL, FEATURES, NUMERIC, evaluate, fit_scorecard, load_split

FIG = Path("../reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Data and class balance

# %%
X_all, y_all = load_german()
print(X_all.shape, "good rate:", round(y_all.mean(), 3))
display(X_all.head())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
y_all.map({1: "good", 0: "bad"}).value_counts().plot.bar(ax=axes[0], title="Outcome")
axes[1].hist(X_all["Credit amount"], bins=40, color="tab:gray")
axes[1].set(title="Credit amount", xlabel="Amount")
fig.tight_layout()
fig.savefig(FIG / "cr01_eda.png", dpi=120)
plt.show()

# %% [markdown]
# ## 2. Split and WOE

# %%
split = load_split()
print(len(split.X_train), len(split.X_test))
print("edges Credit amount (train):", make_numeric_edges(split.X_train["Credit amount"], 5))

# %% [markdown]
# ## 3. Fit the baseline (C = 1) and the tuned scorecard (C by CV)

# %%
sc_default, _ = fit_scorecard(split, C=1.0)
sc_tuned, C_best = fit_scorecard(split)
print("tuned C:", C_best)

res = pd.DataFrame(
    {
        "model": ["C=1 baseline", f"C={C_best} tuned"],
        "test_auc": [evaluate(sc_default, split)["test"]["auc"], evaluate(sc_tuned, split)["test"]["auc"]],
        "test_gini": [evaluate(sc_default, split)["test"]["gini"], evaluate(sc_tuned, split)["test"]["gini"]],
        "test_ks": [evaluate(sc_default, split)["test"]["ks"], evaluate(sc_tuned, split)["test"]["ks"]],
    }
)
display(res.round(4))

# %% [markdown]
# ## 4. IV ranking

# %%
iv = sc_tuned.encoder.iv_table()
display(iv.round(4))

fig, ax = plt.subplots(figsize=(7, 4))
iv.sort_values("iv").plot.barh(x="feature", y="iv", ax=ax, legend=False, title="Information value (train)")
ax.axvline(0.02, color="orange", ls="--")
ax.axvline(0.5, color="red", ls="--")
fig.tight_layout()
fig.savefig(FIG / "cr01_iv.png", dpi=120)
plt.show()

# %% [markdown]
# ## 5. Points table

# %%
pts = sc_tuned.points_table()
display(pts.round(2))

# %% [markdown]
# ## 6. Holdout discrimination and plots

# %%
X_te = split.X_test[FEATURES]
p_te = sc_tuned.predict_good(X_te)
s_te = sc_tuned.score(X_te)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
plot_roc(split.y_test, p_te, label=f"C={C_best}", ax=axes[0])
plot_ks(split.y_test, p_te, ax=axes[1])
plot_score_distribution(s_te, split.y_test, ax=axes[2])
fig.tight_layout()
fig.savefig(FIG / "cr01_holdout.png", dpi=120)
plt.show()

# %%
decile_table(split.y_test, p_te)

# %% [markdown]
# ## 7. Interpretation
#
# Model is a logistic regression on WOE, so each feature has a monotone effect by bin.
# `Sex` is not in the model (protected). It is audited in CR03.

# %%
print(sc_tuned.reason_codes(X_te.head(3), top=3))
