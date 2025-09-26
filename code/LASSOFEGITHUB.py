# -*- coding: utf-8 -*-
"""
DESI LASSO Panel Pipeline — Refactored (No-Leakage + Diagnostics)
(All-sectors pages for Q–Q and Residuals vs Fitted included)

Overview
--------
End-to-end pipeline for sectoral GVA panels that:
• Uses GroupKFold by entity with train-only within-demeaning (no leakage).
• Selects features with LASSO and reports selection stability across folds.
• Fits two-way Fixed Effects (entity & time) with clustered SE, plus Driscoll–Kraay sensitivity.
• Runs panel residual diagnostics: Hausman & Mundlak, Wooldridge AR(1), Fisher-ADF (c/ct), Pesaran CD, Durbin–Watson.
• Applies VIF hygiene (pooled + within on the final estimation sample).
• Writes a clean Excel workbook (multiple sheets) and a text summary.
• Creates "all-sectors" diagnostic pages:
  - ALL_SECTORS_QQ.png
  - ALL_SECTORS_ResidVsFitted.png
  - (optional) ALL_SECTORS_COMBINED.png

Intended Data Layout
--------------------
The input Excel file must contain at least:
• Panel keys:   CountryShort (entity), Year (time)
• Targets:      Columns starting with "GVA_"                 (one per sector)
• Candidates:   Columns starting with "DESI_"           (feature pool)
• Controls:     Columns starting with "CNTRL_"               (optional)

The script filters out countries with fewer than MIN_YEARS_PER_COUNTRY unique years
and logs them in the Excel output.

No-Leakage Feature Selection
----------------------------
• GroupKFold on entities.
• Train-only within-demeaning via WithinDemeaner (fitted on the training fold).
• Pipeline: SimpleImputer → StandardScaler → LassoCV (cv=3).
• Selection stability per target is saved in the Selection_Stability sheet.

Model Estimation
----------------
• Two-way Fixed Effects (entity & time) via linearmodels.PanelOLS.
• Clustered standard errors (entity and time).
• Optional Driscoll–Kraay sensitivity (kernel='bartlett', bandwidth in DK_BANDWIDTHS).

Diagnostics
-----------
• Hausman vs Random Effects + Mundlak (entity means of X).
• Wooldridge AR(1) (approx) on FE residuals.
• Fisher-ADF (c/ct) panel test on residuals (Fisher aggregation).
• Pesaran CD test for cross-sectional dependence.
• Durbin–Watson value (informative).
• VIFs: pooled and within (on the final estimation sample).

Outputs
-------
• CSV:
  - DatasetMasterCAP_filtered.csv             (filtered panel)
• Excel (RESULTS_XLSX):
  - Regression_Results                        (FE coefficients, clustered SE, CIs, p-values, DK p-values, VIFs, R², diagnostics)
  - Diagnostics_Summary                       (one row per sector with key flags and summary stats)
  - Selection_Stability                       (per-target selection frequencies from LASSO CV)
  - Sensitivity_DK                            (coefficients/p-values under DK bandwidths)
  - Dropped_Variables_VIF                     (within-VIF drops pre-fit)
  - Excluded_Countries_Initial                (entities filtered out at load)
• Text:
  - DESILASSOPERCAP_CSE_refactored_summary.txt
• Figures (PLOTS_DIR):
  - ALL_SECTORS_QQ.png
  - ALL_SECTORS_ResidVsFitted.png
  - ALL_SECTORS_COMBINED.png  (if combined page succeeds)
  - <sector>_leave_one_entity_out_influence.csv (optional LOO influence)

How to Run
----------
1) Place the raw Excel dataset at:
   BASE_PATH / "DatasetMasterCAP.xlsx"
   (Adjust BASE_PATH and filenames in the CONFIG section below as needed.)
2) Install requirements (Python 3.10+ recommended):
   pip install numpy pandas scipy statsmodels linearmodels scikit-learn matplotlib seaborn
3) Run:
   python your_script_name.py
4) Inspect outputs:
   • Excel workbook and text summary under BASE_PATH.
   • Diagnostics images under PLOTS_DIR.

Key Configuration Knobs (see CONFIG block)
------------------------------------------
• VIF_THRESHOLD:           Within-VIF drop threshold (default 5.0).
• MIN_YEARS_PER_COUNTRY:   Minimum unique time support per entity.
• N_FOLDS:                 GroupKFold folds (by entity).
• ADD_KERNEL_COV:          Enable Driscoll–Kraay sensitivity fits (True/False).
• DK_BANDWIDTHS:           Bartlett kernel bandwidths for DK sensitivity.
• SEED:                    Reproducibility for LASSO CV and numpy.

Troubleshooting
---------------
• "LASSO selection returned no features":
  - Relax LASSO (e.g., reduce N_FOLDS), check feature variance, raise data coverage.
• Singularities / VIF issues:
  - Review highly collinear features; lower VIF_THRESHOLD or prune variables upstream.
• Missing columns:
  - Ensure expected prefixes exist: GVA_, DESI_, CNTRL_, and keys CountryShort, Year.
• Headless environments:
  - Matplotlib saves PNGs without a display; ensure write permissions on PLOTS_DIR.

Reproducibility
---------------
• Random seeds are set via SEED and LassoCV(random_state=SEED).
• No-leakage guaranteed by train-only within-demeaning and fold logic.
• All key hyperparameters live in the CONFIG section for transparent runs.

Requirements
------------
numpy, pandas, scipy, statsmodels, linearmodels, scikit-learn, matplotlib, seaborn
"""

import os
import warnings
import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson

from linearmodels.panel import PanelOLS, RandomEffects
from scipy.stats import chi2, norm, f as f_dist  # <-- F CDF for Wooldridge p-value

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- CONFIG ----------------------
# CONFIG GUIDE:
# • BASE_PATH: folder that holds the input Excel and where outputs will be written.
# • INPUT_XLSX: must include keys (CountryShort, Year), GVA_* targets, DESI_* features, optional CNTRL_*.
# • RESULTS_XLSX / RESULTS_TXT / PLOTS_DIR: change names/locations as you prefer.
# • Tune VIF_THRESHOLD, N_FOLDS, MIN_YEARS_PER_COUNTRY, ADD_KERNEL_COV, DK_BANDWIDTHS as needed.
# • Runtime scales with #entities × #years × #features × #targets; keep DK sensitivity on only if you need it.

VIF_THRESHOLD = 5.0
MIN_YEARS_PER_COUNTRY = 7
N_FOLDS = 5
SEED = 42

ADD_KERNEL_COV = True
DK_BANDWIDTHS = [2, 3, 4]   # Driscoll–Kraay sensitivity (Bartlett)

# File paths (edit as needed)
BASE_PATH = r"C:\Users\..............."
INPUT_XLSX = os.path.join(BASE_PATH, "DatasetMasterCAP.xlsx")
CSV_PATH = os.path.join(BASE_PATH, "DatasetMasterCAP_filtered.csv")
RESULTS_XLSX = os.path.join(BASE_PATH, "DESILASSOPERCAP_CSE_refactored.xlsx")
RESULTS_TXT = os.path.join(BASE_PATH, "DESILASSOPERCAP_CSE_refactored_summary.txt")
PLOTS_DIR = os.path.join(BASE_PATH, "scats_refactored")
# ---------------------------------------------------

# ---------------------- Utilities ----------------------
def safe_asinh(y):
    y = pd.to_numeric(y, errors="coerce")
    return np.arcsinh(y.astype(float))


def filter_countries_by_data_availability(df, country_col='CountryShort', year_col='Year',
                                          min_years=MIN_YEARS_PER_COUNTRY):
    country_year_counts = df.groupby(country_col)[year_col].nunique()
    countries_to_drop = country_year_counts[country_year_counts < min_years].index.tolist()
    exclusion_log = pd.DataFrame({
        'CountryShort': countries_to_drop,
        'Reason': [f'Fewer than {min_years} unique years of data'] * len(countries_to_drop)
    })
    df_filtered = df[~df[country_col].isin(countries_to_drop)].copy()
    return df_filtered, exclusion_log


# ---------------------- WithinDemeaner (manual, no-leakage) ----------------------
class WithinDemeaner:
    """
    Demeans y & X w.r.t. entity/time means. Train-only fit; can transform any df.
    """
    def __init__(self, entity_col='CountryShort', time_col='Year', y_col=None, x_cols=None):
        self.entity_col = entity_col
        self.time_col = time_col
        self.y_col = y_col
        self.x_cols = list(x_cols) if x_cols is not None else None

        self.y_g_ = None
        self.X_g_ = None
        self.y_e_means_ = None   # Series (index=entity)
        self.X_e_means_ = None   # DataFrame (index=entity, cols=x_cols)
        self.y_t_means_ = None   # Series (index=time)
        self.X_t_means_ = None   # DataFrame (index=time,   cols=x_cols)

    def fit(self, train_df):
        if self.y_col is None or self.x_cols is None:
            raise ValueError("WithinDemeaner requires y_col and x_cols.")
        cols = [self.entity_col, self.time_col, self.y_col] + self.x_cols
        d = train_df[cols].copy()

        self.y_g_ = d[self.y_col].mean()
        self.X_g_ = d[self.x_cols].mean()

        self.y_e_means_ = d.groupby(self.entity_col)[self.y_col].mean()
        self.X_e_means_ = d.groupby(self.entity_col)[self.x_cols].mean()

        self.y_t_means_ = d.groupby(self.time_col)[self.y_col].mean()
        self.X_t_means_ = d.groupby(self.time_col)[self.x_cols].mean()
        return self

    def transform(self, df):
        # y part
        y = df[self.y_col].copy()
        y_e = df[self.entity_col].map(self.y_e_means_)
        y_t = df[self.time_col].map(self.y_t_means_)
        y_e = y_e.fillna(self.y_g_)
        y_t = y_t.fillna(self.y_g_)
        y_w = y - y_e - y_t + self.y_g_

        # X part (row-wise fallback to global means)
        X = df[self.x_cols].copy()
        X_e_rows = []
        for ent in df[self.entity_col]:
            if ent in self.X_e_means_.index:
                X_e_rows.append(self.X_e_means_.loc[ent].values)
            else:
                X_e_rows.append(self.X_g_.values)
        X_e = pd.DataFrame(X_e_rows, index=df.index, columns=self.x_cols)

        X_t_rows = []
        for t in df[self.time_col]:
            if t in self.X_t_means_.index:
                X_t_rows.append(self.X_t_means_.loc[t].values)
            else:
                X_t_rows.append(self.X_g_.values)
        X_t = pd.DataFrame(X_t_rows, index=df.index, columns=self.x_cols)

        X_w = X - X_e - X_t + self.X_g_
        return y_w, X_w


# ---------------- VIF with data hygiene ----------------
def _clean_for_vif(X: pd.DataFrame):
    Xc = X.replace([np.inf, -np.inf], np.nan)
    Xc = Xc.dropna(axis=1, how='all')
    if Xc.shape[1] == 0:
        return Xc
    Xc = Xc.dropna(axis=0, how='any').copy()
    if Xc.shape[1] == 0 or Xc.shape[0] == 0:
        return Xc
    variances = Xc.var(axis=0, ddof=1)
    zero_var_cols = variances[~np.isfinite(variances) | (variances <= 0)].index.tolist()
    if zero_var_cols:
        Xc = Xc.drop(columns=zero_var_cols)
    return Xc


def calculate_vif(X):
    Xc = _clean_for_vif(X)
    if Xc.shape[1] <= 1:
        return pd.DataFrame({'Variable': Xc.columns, 'VIF': [np.nan] * Xc.shape[1]})
    vif_data = []
    X_with_const = sm.add_constant(Xc, has_constant='add')
    for i in range(1, X_with_const.shape[1]):
        try:
            vif = variance_inflation_factor(X_with_const.values, i)
        except Exception:
            vif = np.nan
        vif_data.append({'Variable': Xc.columns[i - 1], 'VIF': vif})
    return pd.DataFrame(vif_data)


def remove_high_vif(X, vif_threshold=VIF_THRESHOLD, gva_target=None, drop_log=None):
    dropped = []
    X_iter = X.copy()
    while True:
        if X_iter.shape[1] <= 2:
            break
        vif_df = calculate_vif(X_iter)
        if vif_df.empty:
            break
        high_vif = vif_df[(pd.to_numeric(vif_df['VIF'], errors='coerce') > vif_threshold)]
        if high_vif.empty:
            break
        worst_var = high_vif.sort_values('VIF', ascending=False).iloc[0]
        worst_feature = worst_var['Variable']
        if drop_log is not None and gva_target is not None:
            drop_log.append({'Sector': gva_target, 'Dropped_Variable': worst_feature, 'VIF': worst_var['VIF']})
        X_iter = X_iter.drop(columns=[worst_feature], errors='ignore')
        dropped.append(worst_feature)
    return X_iter, dropped


# ---------------------- Panel-aware LASSO (manual CV, no leakage) ----------------------
def run_lasso_variable_selection(df):
    warnings.filterwarnings('ignore', category=UserWarning)

    gva_cols = [c for c in df.columns if c.startswith('GVA_')]
    DESI_cols = [c for c in df.columns if c.startswith('DESI_')]
    control_cols = [c for c in df.columns if c.startswith('CNTRL_')]
    X_cols_all = DESI_cols + control_cols

    selected_features_by_gva = {}
    selection_freq_by_gva = {}

    entities = df['CountryShort'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    base_splits = list(gkf.split(df, groups=entities))  # indices on full df

    for gva_target in gva_cols:
        # prepare target
        df_iter = df.copy()
        df_iter[gva_target] = safe_asinh(df_iter[gva_target])
        df_iter = df_iter.dropna(subset=[gva_target]).copy()

        # re-map splits to df_iter length
        max_idx = len(df_iter) - 1
        splits = []
        for tr_idx, te_idx in base_splits:
            tr_idx = [i for i in tr_idx if i <= max_idx]
            te_idx = [i for i in te_idx if i <= max_idx]
            if len(tr_idx) > 10 and len(te_idx) > 1:
                splits.append((np.array(tr_idx), np.array(te_idx)))
        if not splits:
            selected_features_by_gva[gva_target] = []
            selection_freq_by_gva[gva_target] = pd.Series(0.0, index=X_cols_all)
            continue

        # storage for stability
        freq = pd.Series(0.0, index=X_cols_all)
        counts = pd.Series(0.0, index=X_cols_all)

        # fold loop (no leakage)
        for tr_idx, te_idx in splits:
            train = df_iter.iloc[tr_idx]
            test = df_iter.iloc[te_idx]

            # Fit within on TRAIN only; transform TRAIN and TEST
            demean = WithinDemeaner(entity_col='CountryShort', time_col='Year',
                                    y_col=gva_target, x_cols=X_cols_all).fit(train)
            y_tr_w, X_tr_w = demean.transform(train)
            y_te_w, X_te_w = demean.transform(test)

            # Build matrices
            y_tr = pd.Series(y_tr_w, index=train.index).astype(float)
            X_tr = pd.DataFrame(X_tr_w, index=train.index, columns=X_cols_all).astype(float)

            # Drop all-NaN cols in train (after within)
            X_tr = X_tr.dropna(axis=1, how='all')
            keep_cols = X_tr.columns.tolist()
            if len(keep_cols) == 0:
                continue

            # Pipeline on TRAIN within (impute+scale+lasso)
            pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler(with_mean=True, with_std=True)),
                ('lasso', LassoCV(cv=3, random_state=SEED, max_iter=20000, n_jobs=-1, tol=1e-3))
            ])
            pipe.fit(X_tr[keep_cols], y_tr.values.ravel())

            coefs = pd.Series(pipe.named_steps['lasso'].coef_, index=keep_cols)
            freq.loc[keep_cols] += (np.abs(coefs.values) > 1e-6)
            counts.loc[keep_cols] += 1.0

        # final fit on FULL sample (still avoid leakage since it's for reporting)
        demean_full = WithinDemeaner(entity_col='CountryShort', time_col='Year',
                                     y_col=gva_target, x_cols=X_cols_all).fit(df_iter)
        y_full_w, X_full_w = demean_full.transform(df_iter)
        y_full = pd.Series(y_full_w, index=df_iter.index).astype(float)
        X_full = pd.DataFrame(X_full_w, index=df_iter.index, columns=X_cols_all).astype(float)
        X_full = X_full.dropna(axis=1, how='all')

        if X_full.shape[1] == 0:
            selected_features_by_gva[gva_target] = []
            selection_freq_by_gva[gva_target] = freq.divide(counts.where(counts > 0, np.nan))
            continue

        final_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('lasso', LassoCV(cv=3, random_state=SEED, max_iter=20000, n_jobs=-1, tol=1e-3))
        ])
        final_pipe.fit(X_full, y_full.values.ravel())
        final_coefs = pd.Series(final_pipe.named_steps['lasso'].coef_, index=X_full.columns)
        selected = final_coefs.index[np.abs(final_coefs.values) > 1e-6].tolist()

        selected_features_by_gva[gva_target] = selected
        selection_freq_by_gva[gva_target] = freq.divide(counts.where(counts > 0, np.nan)).fillna(0.0)

    return selected_features_by_gva, selection_freq_by_gva


# ----------------- Hausman & Mundlak -----------------
def hausman_test(fixed, random):
    b = fixed.params
    B = random.params
    common = b.index.intersection(B.index)
    if len(common) == 0:
        return np.nan, np.nan, "No common parameters."
    diff = b[common] - B[common]
    v_b = fixed.cov.loc[common, common]
    v_B = random.cov.loc[common, common]
    try:
        inv = np.linalg.pinv(v_b - v_B)
        stat = float(diff.T @ inv @ diff)
        pval = chi2.sf(stat, len(common))
        return stat, pval, None
    except Exception as e:
        return np.nan, np.nan, f"Inversion error: {e}"


def mundlak_re(y, X, entity_index):
    X_means = X.groupby(entity_index).transform('mean')
    X_mundlak = pd.concat([X, X_means.add_suffix('_EMean')], axis=1)
    re = RandomEffects(y, X_mundlak).fit()
    pvals_means = re.pvalues[[c for c in X_mundlak.columns if c.endswith('_EMean')]]
    return re, pvals_means


# ----- Pesaran CD & Fisher-ADF on residuals (c & ct) -----
def _pesaran_cd_from_resid_df(resid_df: pd.DataFrame):
    if resid_df.empty or resid_df.shape[1] < 2:
        return np.nan, np.nan
    R = resid_df.dropna(how='any')
    T, N = R.shape
    if N < 2 or T < 2:
        return np.nan, np.nan
    C = R.corr()
    iu = np.triu_indices(N, 1)
    sum_rho = np.nansum(C.values[iu])
    CD = np.sqrt(2 * T / (N * (N - 1))) * sum_rho
    pval = 2 * (1 - norm.cdf(np.abs(CD)))
    return float(CD), float(pval)


def compute_panel_residual_tests(resids: pd.Series):
    resids = resids.copy()
    resids.index = pd.MultiIndex.from_tuples(resids.index)
    resids = resids.sort_index(level=[0, 1])

    pvals_c, pvals_ct = [], []
    for ent, s in resids.groupby(level=0):
        x = s.droplevel(0).sort_index().dropna().values
        if x.size >= 5:
            try:
                adf_c = adfuller(x, regression='c', autolag='AIC')
                pvals_c.append(adf_c[1])
            except Exception:
                pass
            try:
                adf_ct = adfuller(x, regression='ct', autolag='AIC')
                pvals_ct.append(adf_ct[1])
            except Exception:
                pass

    def fisher(pvals):
        k = len(pvals)
        if k == 0:
            return np.nan, np.nan
        stat = -2.0 * np.sum(np.log(pvals))
        p = chi2.sf(stat, 2 * k)
        return float(stat), float(p)

    fisher_c_stat, fisher_c_p = fisher(pvals_c)
    fisher_ct_stat, fisher_ct_p = fisher(pvals_ct)

    try:
        resid_df = resids.unstack(level=0)
        cd_stat, cd_p = _pesaran_cd_from_resid_df(resid_df)
    except Exception:
        cd_stat, cd_p = np.nan, np.nan

    return {
        'Fisher_ADF_resid_stat_c': fisher_c_stat,
        'Fisher_ADF_resid_p_c': fisher_c_p,
        'Fisher_ADF_resid_stat_ct': fisher_ct_stat,
        'Fisher_ADF_resid_p_ct': fisher_ct_p,
        'Pesaran_CD_stat': cd_stat,
        'Pesaran_CD_p': cd_p
    }


# ----- Wooldridge AR(1) test (approx) -----
def wooldridge_ar1_test(y, X):
    """
    Approx Drukker/Wooldridge test:
    - Fit FE (unadjusted) → residuals u_it
    - For each entity: Δu_it vs u_i,t-1 (pooled OLS w/o intercept)
    - H0: no AR(1)
    """
    fe = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(cov_type='unadjusted')
    res = fe.resids.copy()
    res.index = pd.MultiIndex.from_tuples(res.index)
    res = res.sort_index()

    rows = []
    for ent, s in res.groupby(level=0):
        s = s.droplevel(0).sort_index()
        if len(s) < 3:
            continue
        du = s.diff()
        lag = s.shift(1)
        tmp = pd.DataFrame({'du': du, 'lagu': lag}).dropna()
        if len(tmp) >= 3:
            rows.append(tmp)
    if not rows:
        return np.nan, np.nan
    Z = pd.concat(rows, axis=0)
    model = sm.OLS(Z['du'].values, Z['lagu'].values).fit()
    f_stat = float((model.params[0] ** 2) / (model.bse[0] ** 2))
    pval = float(1 - f_dist.cdf(f_stat, 1, len(Z) - 1))
    return f_stat, pval


# ------------- NEW: All-sectors diagnostic pages -------------
def save_diagnostics_pages(diag_dict, plots_dir, page_name_prefix="ALL_SECTORS", dpi=200):
    """
    Build 2 (+1 optional) figure-pages:
      • {prefix}_QQ.png              → All sectors, Q–Q subplots
      • {prefix}_ResidVsFitted.png   → All sectors, Residuals vs Fitted subplots
      • {prefix}_COMBINED.png        → (optional) Top: ResidVsFitted grid, Bottom: Q–Q grid

    diag_dict format:
      {
        "GVA_SectorName": {
            "resids": 1D array-like,
            "fitted": 1D array-like
        },
        ...
      }
    """
    import math, os
    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import matplotlib.gridspec as gridspec

    os.makedirs(plots_dir, exist_ok=True)
    sectors = sorted([s for s in diag_dict.keys() if diag_dict[s].get("resids") is not None])
    n = len(sectors)
    if n == 0:
        return

    # Grid layout to "fit nicely"
    ncols = 3 if n <= 9 else 4  # bump to 5 if you have many sectors
    nrows = int(np.ceil(n / ncols))

    # Landscape page size from per-cell size
    cell_w, cell_h = 4.0, 3.4
    fig_w = ncols * cell_w
    fig_h = nrows * cell_h

    # ---------- (A) QQ PAGE ----------
    fig_q, axes_q = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi)
    axes_q = np.atleast_2d(axes_q)
    for i, sec in enumerate(sectors):
        r, c = divmod(i, ncols)
        ax = axes_q[r, c]
        res = np.asarray(diag_dict[sec]["resids"], dtype=float)
        res = res[np.isfinite(res)]
        if res.size == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
        else:
            sm.ProbPlot(res).qqplot(line='s', ax=ax, marker='.', markersize=3.5)
        ax.set_title(sec, fontsize=10)
        ax.tick_params(labelsize=8)
    # close empty panels
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes_q[r, c].axis('off')
    fig_q.suptitle("Q–Q Plots — All GVA Sectors", y=0.995, fontsize=14)
    fig_q.tight_layout(rect=[0, 0, 1, 0.98])
    fig_q.savefig(os.path.join(plots_dir, f"{page_name_prefix}_QQ.png"), bbox_inches="tight")
    plt.close(fig_q)

    # ---------- (B) Residuals vs Fitted PAGE ----------
    fig_r, axes_r = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi)
    axes_r = np.atleast_2d(axes_r)
    for i, sec in enumerate(sectors):
        r, c = divmod(i, ncols)
        ax = axes_r[r, c]
        fitted = np.asarray(diag_dict[sec].get("fitted"), dtype=float)
        resids = np.asarray(diag_dict[sec].get("resids"), dtype=float)
        m = np.isfinite(fitted) & np.isfinite(resids)
        fitted, resids = fitted[m], resids[m]
        if fitted.size == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
        else:
            ax.scatter(fitted, resids, s=8, alpha=0.45)
            ax.axhline(0, lw=0.8, alpha=0.6, ls='--')
        ax.set_title(sec, fontsize=10)
        ax.tick_params(labelsize=8)
        if r == nrows - 1: ax.set_xlabel("Fitted", fontsize=9)
        if c == 0:          ax.set_ylabel("Residuals", fontsize=9)
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes_r[r, c].axis('off')
    fig_r.suptitle("Residuals vs Fitted — All GVA Sectors", y=0.995, fontsize=14)
    fig_r.tight_layout(rect=[0, 0, 1, 0.98])
    fig_r.savefig(os.path.join(plots_dir, f"{page_name_prefix}_ResidVsFitted.png"), bbox_inches="tight")
    plt.close(fig_r)

    # ---------- (C) OPTIONAL: COMBINED PAGE ----------
    try:
        fig_c = plt.figure(figsize=(fig_w, fig_h * 2 + 1), dpi=dpi)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.25)

        # Top: Residuals vs Fitted grid
        gs_top = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs[0], wspace=0.25, hspace=0.35)
        for i, sec in enumerate(sectors):
            r, c = divmod(i, ncols)
            ax = fig_c.add_subplot(gs_top[r, c])
            fitted = np.asarray(diag_dict[sec].get("fitted"), dtype=float)
            resids = np.asarray(diag_dict[sec].get("resids"), dtype=float)
            m = np.isfinite(fitted) & np.isfinite(resids)
            fitted, resids = fitted[m], resids[m]
            if fitted.size == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
            else:
                ax.scatter(fitted, resids, s=7, alpha=0.42)
                ax.axhline(0, lw=0.7, alpha=0.6, ls='--')
            ax.set_title(sec, fontsize=9)
            ax.tick_params(labelsize=7)

        # Bottom: Q–Q grid
        gs_bot = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs[1], wspace=0.25, hspace=0.35)
        for i, sec in enumerate(sectors):
            r, c = divmod(i, ncols)
            ax = fig_c.add_subplot(gs_bot[r, c])
            res = np.asarray(diag_dict[sec]["resids"], dtype=float)
            res = res[np.isfinite(res)]
            if res.size == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
            else:
                sm.ProbPlot(res).qqplot(line='s', ax=ax, marker='.', markersize=3.2)
            ax.set_title(sec, fontsize=9)
            ax.tick_params(labelsize=7)

        fig_c.suptitle("Diagnostics — All GVA Sectors (Top: Residuals vs Fitted • Bottom: Q–Q)",
                       y=0.995, fontsize=14)
        fig_c.tight_layout(rect=[0, 0, 1, 0.985])
        fig_c.savefig(os.path.join(plots_dir, f"{page_name_prefix}_COMBINED.png"), bbox_inches="tight")
        plt.close(fig_c)
    except Exception:
        pass


# ------------- (optional utility kept for backward-compat) -------------
def leave_one_entity_out_influence(y, X):
    """
    Leave-one-entity-out Δβ (L2 norm). Returns DataFrame sorted desc.
    """
    entities = y.index.get_level_values(0).unique()
    base = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    beta0 = base.params
    rows = []
    for ent in entities:
        mask = y.index.get_level_values(0) != ent
        if mask.sum() < len(X.columns) + 5:
            continue
        fit_i = PanelOLS(y[mask], X[mask], entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
        b_i = fit_i.params
        common = beta0.index.intersection(b_i.index)
        d = np.linalg.norm((beta0.loc[common] - b_i.loc[common]).values, 2)
        rows.append({'Entity': ent, 'DeltaBeta_L2': d})
    if not rows:
        return pd.DataFrame(columns=['Entity', 'DeltaBeta_L2'])
    return pd.DataFrame(rows).sort_values('DeltaBeta_L2', ascending=False)


# ---------------- FE models (with within-VIF, FDR, DK sensitivity) ----------------
def run_fixed_effects_models(df_csv_path, selected_features_dict, vif_threshold, plots_dir, results_txt_path):
    warnings.filterwarnings('ignore', category=FutureWarning)
    df_panel = pd.read_csv(df_csv_path)

    with open(results_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Panel Regression & Diagnostics (Two-Way Clustered; within-VIF; FDR; DK_sensitivity={ADD_KERNEL_COV})\n"
                f"Generated on: {pd.Timestamp.now()}\n")

    all_rows, drop_log = [], []
    dk_rows = []

    # NEW: collect residuals & fitted for all sectors → pages
    diag_collect = {}

    for gva_target, features in selected_features_dict.items():
        if not features:
            continue

        df_iter = df_panel.copy()
        df_iter[gva_target] = safe_asinh(df_iter[gva_target])

        model_vars = [gva_target] + features
        df_subset = df_iter[['CountryShort', 'Year'] + model_vars].dropna(subset=[gva_target]).copy()
        df_subset = df_subset.set_index(['CountryShort', 'Year'])
        if df_subset.empty:
            continue

        y = df_subset[gva_target]
        X = df_subset[features]

        # Impute (post-selection)
        imputer = SimpleImputer(strategy='mean')
        X_imp = pd.DataFrame(imputer.fit_transform(X), index=X.index, columns=X.columns)

        # ----- Within-VIF matrix for DROP (pre-fit; informative for FE) -----
        def within_full(df0, y_col, X_cols, entity='CountryShort', time='Year'):
            y_g = df0[y_col].mean()
            X_g = df0[X_cols].mean()
            y_e = df0.groupby(entity)[y_col].transform('mean')
            X_e = df0.groupby(entity)[X_cols].transform('mean')
            y_t = df0.groupby(time)[y_col].transform('mean')
            X_t = df0.groupby(time)[X_cols].transform('mean')
            y_w = df0[y_col] - y_e - y_t + y_g
            X_w = df0[X_cols] - X_e - X_t + X_g
            return y_w, X_w

        df_tmp = df_panel[['CountryShort', 'Year'] + features + [gva_target]].dropna(subset=[gva_target]).copy()
        _, Xw_for_vif = within_full(df_tmp.reset_index(drop=True), gva_target, features)
        Xw_for_vif = Xw_for_vif.reindex(columns=features).dropna(axis=1, how='all')
        Xw_for_vif_clean = _clean_for_vif(Xw_for_vif)

        # Drop high within-VIF vars (pre-fit)
        if Xw_for_vif_clean.shape[1] >= 2:
            X_final_w, dropped = remove_high_vif(Xw_for_vif_clean, vif_threshold, gva_target, drop_log)
            keep_cols = list(X_final_w.columns)
            X_imp = X_imp[keep_cols].copy()
        else:
            keep_cols, dropped = list(X_imp.columns), []

        # Align
        y, X_imp = y.align(X_imp, join='inner', axis=0)
        if len(y) < len(X_imp.columns) + 5:
            continue

        # FE model
        fe_model = PanelOLS(y, X_imp, entity_effects=True, time_effects=True)
        robust_results = fe_model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)

        # --- collect residuals & fitted for pages
        try:
            diag_collect[gva_target] = {
                "resids": robust_results.resids.values.squeeze(),
                "fitted": robust_results.fitted_values.values.squeeze()
            }
        except Exception:
            diag_collect[gva_target] = {"resids": None, "fitted": None}

        # DK sensitivity
        kernel_results_dict = {}
        if ADD_KERNEL_COV:
            for bw in DK_BANDWIDTHS:
                try:
                    kernel_results_dict[bw] = fe_model.fit(cov_type='kernel', kernel='bartlett', bandwidth=bw)
                except Exception:
                    kernel_results_dict[bw] = None

        # RE & Mundlak
        entity_index = y.index.get_level_values(0)
        try:
            re_model = RandomEffects(y, X_imp).fit()
        except Exception:
            re_model = None
        try:
            re_mundlak, mundlak_p_means = mundlak_re(y, X_imp, entity_index)
        except Exception:
            re_mundlak, mundlak_p_means = None, pd.Series(dtype=float)

        # Hausman
        if re_model is not None:
            chi2_stat, haus_pvalue, haus_note = hausman_test(robust_results, re_model)
        else:
            chi2_stat, haus_pvalue, haus_note = np.nan, np.nan, "RE failed."

        # Residual diagnostics
        resid_tests = compute_panel_residual_tests(robust_results.resids)
        try:
            wool_F, wool_p = wooldridge_ar1_test(y, X_imp)
        except Exception:
            wool_F, wool_p = np.nan, np.nan

        # DW (informative)
        try:
            dw_value = float(durbin_watson(robust_results.resids.values))
        except Exception:
            dw_value = np.nan

        # ---------------- Within-VIF REPORT on FINAL estimation sample ----------------
        if keep_cols:
            X_imp_final = X_imp[keep_cols].copy()
            # within transform on final estimation sample (entity idx=level 0, time idx=level 1)
            Xe_means = X_imp_final.groupby(level=0).transform('mean')
            Xt_means = X_imp_final.groupby(level=1).transform('mean')
            Xg_means = X_imp_final.mean()
            Xw_final_report = X_imp_final - Xe_means - Xt_means + Xg_means
            within_vif_df = calculate_vif(Xw_final_report) if Xw_final_report.shape[1] >= 2 else pd.DataFrame()
        else:
            within_vif_df = pd.DataFrame()

        # Pooled-VIF (indicative) on the final sample
        pooled_vif_df = calculate_vif(X_imp) if X_imp.shape[1] >= 2 else pd.DataFrame()

        # Text report
        with open(results_txt_path, 'a', encoding='utf-8') as f:
            f.write(f"\n\n{'='*80}\nRegression Results for: {gva_target}\n{'='*80}\n{robust_results}\n")
            if ADD_KERNEL_COV:
                f.write("\n-- Driscoll–Kraay sensitivity (kernel=bartlett) --\n")
                for bw, kr in kernel_results_dict.items():
                    f.write(f"\n[bw={bw}]\n{kr}\n" if kr is not None else f"\n[bw={bw}] DK fit failed.\n")
            f.write(f"\n{'-'*30} Diagnostics {'-'*30}\n")
            f.write(f"\nHausman p: {haus_pvalue if np.isfinite(haus_pvalue) else 'NA'}"
                    f" | Note: {haus_note if haus_note else 'OK'}")
            f.write(f"\nMundlak means p (min): {mundlak_p_means.min():.4f}" if not mundlak_p_means.empty else "\nMundlak: NA")
            f.write(f"\nWooldridge AR(1) F={wool_F:.4f}, p={wool_p:.4f}" if np.isfinite(wool_F) else "\nWooldridge: NA")
            f.write(f"\nFisher-ADF (c) p={resid_tests['Fisher_ADF_resid_p_c']:.4f}, "
                    f"(ct) p={resid_tests['Fisher_ADF_resid_p_ct']:.4f}")
            f.write(f"\nPesaran CD p={resid_tests['Pesaran_CD_p']:.4f}")
            f.write(f"\nDurbin–Watson (informative): {dw_value:.4f}\n")

            if not within_vif_df.empty:
                f.write("\nWithin-VIF (final estimation sample) range: "
                        f"{within_vif_df['VIF'].min():.3f} – {within_vif_df['VIF'].max():.3f}")
            if not pooled_vif_df.empty:
                f.write("\nPooled-VIF (indicative, final sample) range: "
                        f"{pooled_vif_df['VIF'].min():.3f} – {pooled_vif_df['VIF'].max():.3f}\n")

        # Optional: leave-one-entity-out influence
        try:
            loo = leave_one_entity_out_influence(y, X_imp)
            if not loo.empty:
                loo.to_csv(os.path.join(plots_dir, f"{gva_target}_leave_one_entity_out_influence.csv"), index=False)
        except Exception:
            pass

        # Collect rows (clustered-robust)
        params = robust_results.params
        robust_std_errors = robust_results.std_errors
        pvalues = robust_results.pvalues
        conf_int = robust_results.conf_int()

        for var in params.index:
            row = {
                'Sector': gva_target,
                'Variable': var,
                'Coefficient': float(params[var]),
                'Robust_Std_Error': float(robust_std_errors[var]),
                'P_Value': float(pvalues[var]),
                'CI_Lower': float(conf_int.loc[var, 'lower']),
                'CI_Upper': float(conf_int.loc[var, 'upper']),
                'R-squared': float(robust_results.rsquared),
                'R-squared (Between)': float(robust_results.rsquared_between),
                'R-squared (Within)': float(robust_results.rsquared_within),
                'R-squared (Overall)': float(robust_results.rsquared_overall),
                'DurbinWatson': dw_value,
                'Hausman_p': float(haus_pvalue) if np.isfinite(haus_pvalue) else np.nan,
                'Mundlak_min_p_EMeans': float(mundlak_p_means.min()) if not mundlak_p_means.empty else np.nan,
                'Wooldridge_F': wool_F,
                'Wooldridge_p': wool_p,
                'Fisher_ADF_resid_stat_c': resid_tests['Fisher_ADF_resid_stat_c'],
                'Fisher_ADF_resid_p_c': resid_tests['Fisher_ADF_resid_p_c'],
                'Fisher_ADF_resid_stat_ct': resid_tests['Fisher_ADF_resid_stat_ct'],
                'Fisher_ADF_resid_p_ct': resid_tests['Fisher_ADF_resid_p_ct'],
                'Pesaran_CD_stat': resid_tests['Pesaran_CD_stat'],
                'Pesaran_CD_p': resid_tests['Pesaran_CD_p'],
                'Within_VIF': np.nan,
                'Pooled_VIF': np.nan
            }
            try:
                row['Within_VIF'] = float(within_vif_df.set_index('Variable').loc[var, 'VIF'])
            except Exception:
                pass
            try:
                row['Pooled_VIF'] = float(pooled_vif_df.set_index('Variable').loc[var, 'VIF'])
            except Exception:
                pass
            all_rows.append(row)

        # DK sensitivity sheet rows
        if ADD_KERNEL_COV:
            for bw, kr in kernel_results_dict.items():
                if kr is None:
                    continue
                for var in kr.params.index:
                    dk_rows.append({
                        'Sector': gva_target,
                        'Bandwidth': bw,
                        'Variable': var,
                        'DK_Coefficient': float(kr.params[var]),
                        'DK_Std_Error': float(kr.std_errors[var]),
                        'DK_P_Value': float(kr.pvalues[var])
                    })

        for d in dropped:
            drop_log.append({'Sector': gva_target, 'Dropped_Variable': d, 'VIF': np.nan})

    # --- Build all-sectors pages ---
    try:
        if len(diag_collect) > 0:
            save_diagnostics_pages(diag_collect, plots_dir, page_name_prefix="ALL_SECTORS", dpi=200)
    except Exception as e:
        print("save_diagnostics_pages failed:", e)

    if not all_rows:
        return None, None, None

    full_results_df = pd.DataFrame(all_rows)

    # FDR per sector
    full_results_df['P_FDR'] = np.nan
    for sec, g in full_results_df.groupby('Sector'):
        try:
            _, p_adj, _, _ = multipletests(g['P_Value'].values, method='fdr_bh')
            full_results_df.loc[g.index, 'P_FDR'] = p_adj
        except Exception:
            pass

    drop_log_df = pd.DataFrame(drop_log) if drop_log else pd.DataFrame(columns=['Sector', 'Dropped_Variable', 'VIF'])
    dk_df = pd.DataFrame(dk_rows) if dk_rows else pd.DataFrame(columns=['Sector','Bandwidth','Variable','DK_Coefficient','DK_Std_Error','DK_P_Value'])
    return full_results_df, drop_log_df, dk_df


def build_diagnostics_summary(results_df):
    # Helper
    def flag(p):
        try:
            return float(p) < 0.05
        except Exception:
            return False

    agg = []
    for sec, g in results_df.groupby("Sector"):
        k = g["Variable"].nunique()
        r2_within = g["R-squared (Within)"].dropna().iloc[0] if g["R-squared (Within)"].notna().any() else np.nan
        dw = g["DurbinWatson"].dropna().iloc[0] if g["DurbinWatson"].notna().any() else np.nan
        haus_p = g["Hausman_p"].dropna().iloc[0] if g["Hausman_p"].notna().any() else np.nan
        wool_p = g["Wooldridge_p"].dropna().iloc[0] if g["Wooldridge_p"].notna().any() else np.nan
        cd_p = g["Pesaran_CD_p"].dropna().iloc[0] if g["Pesaran_CD_p"].notna().any() else np.nan
        fadf_c_p = g["Fisher_ADF_resid_p_c"].dropna().iloc[0] if g["Fisher_ADF_resid_p_c"].notna().any() else np.nan
        fadf_ct_p = g["Fisher_ADF_resid_p_ct"].dropna().iloc[0] if g["Fisher_ADF_resid_p_ct"].notna().any() else np.nan
        mundlak_min_p = g["Mundlak_min_p_EMeans"].dropna().iloc[0] if g["Mundlak_min_p_EMeans"].notna().any() else np.nan

        # Core counts
        sig_coef = (g["P_Value"] < 0.05).sum()
        sig_fdr = (g["P_FDR"] < 0.05).sum() if "P_FDR" in g else np.nan

        # ADF: Nonstationary flags using the rule p>=0.05
        nonstat_c = (float(fadf_c_p) >= 0.05) if not pd.isna(fadf_c_p) else np.nan
        nonstat_ct = (float(fadf_ct_p) >= 0.05) if not pd.isna(fadf_ct_p) else np.nan

        agg.append({
            "Sector": sec,
            "Num_Coefficients": int(k),
            "R2_Within": r2_within,
            "DW_info": dw,
            "Hausman_p": haus_p,
            "Reject_RE (Hausman)": flag(haus_p),
            "Mundlak_min_p_EMeans": mundlak_min_p,
            "Reject_RE (Mundlak)": flag(mundlak_min_p),
            "Wooldridge_AR1_p": wool_p,
            "AR1_present": flag(wool_p),
            "Pesaran_CD_p": cd_p,
            "CrossSectional_Dependence": flag(cd_p),
            "Fisher_ADF_c_p": fadf_c_p,
            "Nonstationary_resid_c": nonstat_c,
            "Nonstationary_resid_c_01": (1 if nonstat_c is True else (0 if nonstat_c is False else np.nan)),
            "Fisher_ADF_ct_p": fadf_ct_p,
            "Nonstationary_resid_ct": nonstat_ct,
            "Nonstationary_resid_ct_01": (1 if nonstat_ct is True else (0 if nonstat_ct is False else np.nan)),
            "Sig_Coeff_p<0.05": int(sig_coef),
            "Sig_Coeff_FDR<0.05": int(sig_fdr) if not pd.isna(sig_fdr) else np.nan
        })
    return pd.DataFrame(agg).sort_values("Sector").reset_index(drop=True)


# ------------------------------ MAIN ------------------------------
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Diagnostic plots will be saved in '{os.path.abspath(PLOTS_DIR)}'.")

    print("\n--- Step 1: Load & Filter Data ---")
    df0 = pd.read_excel(INPUT_XLSX)

    # Keep only countries with enough time support
    df_filtered, exclusion_log = filter_countries_by_data_availability(df0)
    df_filtered.to_csv(CSV_PATH, index=False)

    print("\n--- Step 2: LASSO Feature Selection (Panel-aware, No-Leakage) ---")
    selected_features, selection_freqs = run_lasso_variable_selection(df_filtered)

    if any(selected_features.values()):
        print("\n--- Selected Features Summary ---")
        for gva_var, features in selected_features.items():
            print(f"{gva_var}: {len(features)} vars -> {', '.join(features) if features else 'None'}")
            if gva_var in selection_freqs and not selection_freqs[gva_var].empty:
                top_stable = selection_freqs[gva_var].sort_values(ascending=False).head(10)
                if (top_stable > 0).any():
                    print("  Top stability:")
                    for k, v in top_stable.items():
                        if v > 0:
                            print(f"    {k}: {v:.2f}")

        print("\n--- Step 3: FE Models + Diagnostics + DK Sensitivity ---")
        full_results_df, drop_log_df, dk_df = run_fixed_effects_models(
            CSV_PATH, selected_features,
            vif_threshold=VIF_THRESHOLD,
            plots_dir=PLOTS_DIR,
            results_txt_path=RESULTS_TXT
        )

        print("\n--- Step 4: Save Results ---")
        if full_results_df is not None:
            diag_df = build_diagnostics_summary(full_results_df)

            # Build Selection_Stability sheet from selection_freqs
            rows = []
            for sec, ser in selection_freqs.items():
                if ser is None or len(ser) == 0:
                    continue
                s = ser.dropna()
                for var, freq in s.items():
                    rows.append({'Sector': sec, 'Variable': var, 'Selection_Frequency': float(freq)})
            selection_df = pd.DataFrame(rows).sort_values(['Sector','Selection_Frequency'],
                                                          ascending=[True, False])

            with pd.ExcelWriter(RESULTS_XLSX) as writer:
                full_results_df.to_excel(writer, sheet_name="Regression_Results", index=False)
                diag_df.to_excel(writer, sheet_name="Diagnostics_Summary", index=False)
                if not selection_df.empty:
                    selection_df.to_excel(writer, sheet_name="Selection_Stability", index=False)
                if dk_df is not None and not dk_df.empty:
                    dk_df.to_excel(writer, sheet_name="Sensitivity_DK", index=False)
                if drop_log_df is not None and not drop_log_df.empty:
                    drop_log_df.to_excel(writer, sheet_name="Dropped_Variables_VIF", index=False)
                if exclusion_log is not None and not exclusion_log.empty:
                    exclusion_log.to_excel(writer, sheet_name="Excluded_Countries_Initial", index=False)

            print(f"Excel saved: '{RESULTS_XLSX}'")
            print(f"Text summary saved: '{RESULTS_TXT}'")
            print(f"All-sectors pages saved to: '{PLOTS_DIR}' -> "
                  f"ALL_SECTORS_QQ.png, ALL_SECTORS_ResidVsFitted.png (+ COMBINED)")
        else:
            print("Finished, but no numerical results were generated.")
    else:
        print("LASSO selection returned no features. Halting process.")

    print("\nProcess completed.")
