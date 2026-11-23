"""
Mobile Usage Addiction Prediction  -  Sprints 1, 2 and 3
Regression problem: predict Addiction_Level (Innomatics 4-sprint framework)

Run:  python mobile_addiction_sprints_1_to_3.py "phone_addiction_dataset 1.csv"
Outputs go to ./outputs (plots, tables, model).
"""
import sys, os, warnings, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
SEED = 42
CSV = sys.argv[1] if len(sys.argv) > 1 else "phone_addiction_dataset 1.csv"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.append(s)


def metrics(y, p):
    return dict(MAE=mean_absolute_error(y, p),
                RMSE=np.sqrt(mean_squared_error(y, p)),
                R2=r2_score(y, p))


# =====================================================================
# SPRINT 1 : DATA UNDERSTANDING & PREPROCESSING
# =====================================================================
say("=" * 70, "\nSPRINT 1\n", "=" * 70)

# Step 1 - load & validate
df = pd.read_csv(CSV)
say("Shape:", df.shape)

# Step 2 - inspection
say("\nDtypes:\n", df.dtypes.to_string())
say("\nMissing values:", int(df.isna().sum().sum()))
say("Duplicates:", int(df.duplicated().sum()))
say("\nUnique values per column:\n", df.nunique().to_string())
say("\nDescribe:\n", df.describe().T.round(2).to_string())

# Step 3 - cleaning
df.columns = [c.strip().lower() for c in df.columns]                 # standardise names
df = df.rename(columns={"interllectual_performance": "intellectual_performance"})  # fix typo
n0 = len(df)
df = df.drop_duplicates().reset_index(drop=True)
say(f"\nRemoved {n0 - len(df)} duplicate rows")
for c in df.select_dtypes("object"):
    df[c] = df[c].str.strip()
# name / location are identifiers (2933 and 2726 distinct values) -> no predictive meaning
df = df.drop(columns=["name", "location"])
say("Dropped identifier columns: name, location")

TARGET = "addiction_level"
num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != TARGET]
cat_cols = ["gender", "phone_usage_purpose"]

# Step 4 - EDA
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(5, 4, figsize=(18, 16))
for a, c in zip(ax.ravel(), num_cols + [TARGET]):
    sns.histplot(df[c], kde=True, ax=a, color="#c62828")
    a.set_title(f"{c} (skew {df[c].skew():.2f})", fontsize=9)
for a in ax.ravel()[len(num_cols) + 1:]:
    a.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/01_univariate.png", dpi=110); plt.close()

corr = df[num_cols + [TARGET]].corr()
say("\nCorrelation with target:\n", corr[TARGET].drop(TARGET).sort_values(ascending=False).round(3).to_string())
plt.figure(figsize=(14, 11))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, annot_kws={"size": 7})
plt.title("Correlation heatmap"); plt.tight_layout()
plt.savefig(f"{OUT}/02_heatmap.png", dpi=110); plt.close()

top = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).index[:6]
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for a, c in zip(ax.ravel(), top):
    sns.scatterplot(x=df[c], y=df[TARGET], alpha=.25, s=10, ax=a, color="#1565c0")
    a.set_title(f"{c} vs {TARGET}")
plt.tight_layout(); plt.savefig(f"{OUT}/03_bivariate.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=df, x="gender", y=TARGET, ax=ax[0])
sns.boxplot(data=df, x="phone_usage_purpose", y=TARGET, ax=ax[1])
plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(f"{OUT}/04_categorical.png", dpi=110); plt.close()

say(f"\nTarget skew {df[TARGET].skew():.2f}; {(df[TARGET] == 10).mean():.1%} of rows sit at the max value 10 "
    "(target is capped / right-censored)")

# Step 5 - outliers (IQR). Values are all within valid ranges, so we CAP rather than drop
cap_bounds = {}
rep = []
for c in num_cols:
    q1, q3 = df[c].quantile([.25, .75]); i = q3 - q1
    lo, hi = q1 - 3 * i, q3 + 3 * i            # 3*IQR = only extreme values
    n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
    rep.append((c, n_out)); cap_bounds[c] = (lo, hi)
    df[c] = df[c].clip(lo, hi)
say("\nExtreme outliers (3*IQR) capped:", {k: v for k, v in rep if v})
plt.figure(figsize=(14, 5)); sns.boxplot(data=df[num_cols].apply(lambda s: (s - s.mean()) / s.std()), orient="h")
plt.title("Boxplots after capping (z-scored for comparison)"); plt.tight_layout()
plt.savefig(f"{OUT}/05_boxplots.png", dpi=110); plt.close()

# Step 6-8 - encoding, split, scaling (scaler/encoder FITTED ON TRAIN ONLY via pipeline)
X, y = df.drop(columns=TARGET), df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
say(f"\nTrain {X_tr.shape}  Test {X_te.shape}")


def make_pre(nums, cats):
    return ColumnTransformer([("num", StandardScaler(), nums),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cats)])


pre = make_pre(num_cols, cat_cols)
say("Sprint 1 deliverables: clean dataset, EDA plots (outputs/01-05), documented steps.")

# =====================================================================
# SPRINT 2 : MODEL BUILDING & EVALUATION
# =====================================================================
say("\n" + "=" * 70, "\nSPRINT 2\n", "=" * 70)
models = {
    "Linear Regression (baseline)": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
}
rows = []
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("m", m)]).fit(X_tr, y_tr)
    tr, te = metrics(y_tr, pipe.predict(X_tr)), metrics(y_te, pipe.predict(X_te))
    cv = cross_val_score(Pipeline([("pre", pre), ("m", m)]), X_tr, y_tr, cv=5, scoring="r2").mean()
    gap = tr["R2"] - te["R2"]
    rows.append({"Model": name, "Train R2": tr["R2"], "Test R2": te["R2"], "CV R2": cv,
                 "Test MAE": te["MAE"], "Test RMSE": te["RMSE"],
                 "Overfitting?": "YES" if gap > 0.05 else "No",
                 "Remarks": "baseline" if "Linear" in name else f"R2 gap {gap:.2f}"})
s2 = pd.DataFrame(rows).round(4)
s2.to_csv(f"{OUT}/sprint2_model_comparison.csv", index=False)
say("\n" + s2.to_string(index=False))
best2 = s2.sort_values("Test R2", ascending=False).iloc[0]
say(f"\nBest Sprint-2 model: {best2['Model']}  (Test R2 {best2['Test R2']})")

# =====================================================================
# SPRINT 3 : OPTIMISATION & FINAL MODEL
# =====================================================================
say("\n" + "=" * 70, "\nSPRINT 3\n", "=" * 70)


# Step 1 - feature engineering (domain ratios / aggregates)
def engineer(d):
    d = d.copy()
    d["weekend_minus_daily"] = d["weekend_usage_hours"] - d["daily_usage_hours"]
    d["usage_per_check"] = d["daily_usage_hours"] / (d["phone_checks_per_day"] + 1)
    d["checks_per_hour"] = d["phone_checks_per_day"] / (d["daily_usage_hours"] + 0.1)
    d["social_gaming_total"] = d["time_on_social_media"] + d["time_on_gaming"]
    d["leisure_share"] = d["social_gaming_total"] / (d["daily_usage_hours"] + 0.1)
    d["productive_share"] = d["time_on_education"] / (d["daily_usage_hours"] + 0.1)
    d["apps_x_usage"] = d["apps_used_daily"] * d["daily_usage_hours"]
    d["sleep_deficit"] = (8 - d["sleep_hours"]).clip(lower=0)
    d["usage_sleep_ratio"] = d["daily_usage_hours"] / (d["sleep_hours"] + 0.1)
    d["bedtime_share"] = d["screen_time_before_bed"] / (d["daily_usage_hours"] + 0.1)
    d["wellbeing_score"] = d["self_esteem"] - d["anxiety_level"] - d["depression_level"]
    d["usage_sq"] = d["daily_usage_hours"] ** 2
    return d


X_tr_f, X_te_f = engineer(X_tr), engineer(X_te)
new_feats = [c for c in X_tr_f.columns if c not in X_tr.columns]
say("Engineered features:", new_feats)
all_num = num_cols + new_feats

# Step 2 - feature selection: correlation filter + importance + RFE
c_mat = X_tr_f[all_num].corr().abs()
upper = c_mat.where(np.triu(np.ones(c_mat.shape), k=1).astype(bool))
drop_corr = [c for c in upper.columns if (upper[c] > 0.95).any()]
say("Dropped (|corr| > 0.95 with another feature):", drop_corr)
cand = [c for c in all_num if c not in drop_corr]

Z = StandardScaler().fit_transform(X_tr_f[cand])
imp = RandomForestRegressor(300, random_state=SEED, n_jobs=-1).fit(Z, y_tr)
fi = pd.Series(imp.feature_importances_, index=cand).sort_values(ascending=False)
say("\nTop feature importances:\n", fi.head(12).round(4).to_string())
fi.sort_values().tail(15).plot.barh(figsize=(8, 6), color="#c62828")
plt.title("Feature importance (RF)"); plt.tight_layout(); plt.savefig(f"{OUT}/06_importance.png", dpi=110); plt.close()

rfe = RFE(GradientBoostingRegressor(random_state=SEED, n_estimators=100), n_features_to_select=15).fit(Z, y_tr)
rfe_keep = [c for c, k in zip(cand, rfe.support_) if k]
imp_keep = fi[fi > 0.005].index.tolist()
selected = [c for c in cand if (c in rfe_keep) or (c in imp_keep)]
say("\nSelected numeric features:", selected)
sel_cat = cat_cols  # kept; negligible cost

# Compare feature sets quickly with CV so selection is justified, not assumed
def cv_r2(nums, cats):
    p = Pipeline([("pre", make_pre(nums, cats)),
                  ("m", GradientBoostingRegressor(random_state=SEED))])
    return cross_val_score(p, X_tr_f, y_tr, cv=5, scoring="r2").mean()

fs = {"original": cv_r2(num_cols, cat_cols), "original+engineered": cv_r2(all_num, cat_cols),
      "selected": cv_r2(selected, sel_cat), "selected (no categoricals)": cv_r2(selected, [])}
say("\nCV R2 by feature set (GB default):", {k: round(v, 4) for k, v in fs.items()})
best_fs = max(fs, key=fs.get)
nums_final, cats_final = {"original": (num_cols, cat_cols), "original+engineered": (all_num, cat_cols),
                          "selected": (selected, sel_cat), "selected (no categoricals)": (selected, [])}[best_fs]
say("Chosen feature set:", best_fs)

# Step 3 - hyperparameter tuning (RandomizedSearchCV, 5-fold)
kf = KFold(5, shuffle=True, random_state=SEED)
pipe_gb = Pipeline([("pre", make_pre(nums_final, cats_final)),
                    ("m", GradientBoostingRegressor(random_state=SEED))])
grid_gb = {"m__n_estimators": [200, 300, 500, 700], "m__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
           "m__max_depth": [2, 3, 4, 5], "m__subsample": [0.6, 0.8, 1.0],
           "m__min_samples_leaf": [1, 5, 10, 20], "m__max_features": [None, 0.5, 0.8]}
rs_gb = RandomizedSearchCV(pipe_gb, grid_gb, n_iter=40, cv=kf, scoring="r2", random_state=SEED, n_jobs=-1).fit(X_tr_f, y_tr)

pipe_rf = Pipeline([("pre", make_pre(nums_final, cats_final)),
                    ("m", RandomForestRegressor(random_state=SEED, n_jobs=-1))])
grid_rf = {"m__n_estimators": [300, 500], "m__max_depth": [8, 12, 16, None],
           "m__min_samples_leaf": [1, 3, 5, 10], "m__max_features": [0.3, 0.5, 0.8, 1.0]}
rs_rf = RandomizedSearchCV(pipe_rf, grid_rf, n_iter=15, cv=kf, scoring="r2", random_state=SEED, n_jobs=-1).fit(X_tr_f, y_tr)

say(f"\nTuned GB  CV R2 {rs_gb.best_score_:.4f}  params {rs_gb.best_params_}")
say(f"Tuned RF  CV R2 {rs_rf.best_score_:.4f}  params {rs_rf.best_params_}")

# Step 4/5 - final model + evaluation on unseen test data
final = rs_gb if rs_gb.best_score_ >= rs_rf.best_score_ else rs_rf
final_model = final.best_estimator_
p_tr, p_te = final_model.predict(X_tr_f), np.clip(final_model.predict(X_te_f), 1, 10)  # target range is 1-10
mt, me = metrics(y_tr, final_model.predict(X_tr_f)), metrics(y_te, p_te)
say("\nFINAL MODEL:", type(final_model.named_steps["m"]).__name__)
say("Train:", {k: round(v, 4) for k, v in mt.items()})
say("Test :", {k: round(v, 4) for k, v in me.items()})
base = s2[s2["Model"].str.contains("Linear")].iloc[0]
say(f"Improvement over baseline Test R2: {base['Test R2']:.4f} -> {me['R2']:.4f}")
say(f"Within +/-1 point of true addiction level: {(np.abs(p_te - y_te) <= 1).mean():.1%}")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(y_te, p_te, alpha=.3, s=10); ax[0].plot([1, 10], [1, 10], "r--")
ax[0].set(xlabel="Actual", ylabel="Predicted", title="Actual vs Predicted (test)")
sns.histplot(y_te - p_te, kde=True, ax=ax[1]); ax[1].set_title("Residuals")
plt.tight_layout(); plt.savefig(f"{OUT}/07_final_eval.png", dpi=110); plt.close()

# Step 6 - serialisation (full pipeline incl. feature engineering is NOT inside sklearn pipeline,
# so we save the engineer() recipe via the saved feature list + model)
joblib.dump({"model": final_model, "numeric_features": nums_final, "categorical_features": cats_final,
             "engineered_features": new_feats}, f"{OUT}/final_model.pkl")
say("\nSaved:", f"{OUT}/final_model.pkl")

summary = {"sprint2": s2.to_dict("records"), "final_test": me, "final_train": mt,
           "final_model": type(final_model.named_steps['m']).__name__, "best_params": final.best_params_ if hasattr(final, "best_params_") else {}}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2, default=str)
open(f"{OUT}/run_log.txt", "w").write("\n".join(log))
