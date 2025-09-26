# -*- coding: utf-8 -*-
"""
Block-SPCA → Two-way Fixed Effects (no lags) — One workbook, many sheets (Global SPCA once)
+ LP-IRFs small multiples (2×3) per sector for the 6 base factors with gray shading on
segments where p<0.10 (clustered).

Overview
--------
This script:
1) Builds block-specific Sparse PCA (SPCA) factors for the Digital Economy (DE) and
   Entrepreneurship (ENT) indicator blocks (global, unsupervised, once).
2) (Optional) Creates DE×ENT interaction terms (not used as IRF shocks).
3) Runs sector-by-sector Two-way FE panel regressions of GVA sectors on the factors.
   - Clustered SE (entity & time), DK kernel sensitivity (Bartlett bandwidths).
   - Diagnostics: Durbin–Watson, Breusch–Pagan, Pesaran CD, Fisher-ADF on residuals.
   - VIFs: pooled and within (on within-transformed matrix).
   - FDR-adjusted p-values and significance stars.
4) (Optional) Estimates Local Projections (LP) IRFs for each sector to shocks in the 6 base factors.
   - Shock definition: AR(1) innovation with entity FE (NO time FE).
   - LP controls: lags of the SAME shock (NOT level); optionally lags of OTHER shocks.
   - Specs kept: Baseline & WithCtrls.
   - Plots: 2×3 small-multiple grids per sector with gray shading on segments where p<0.10.

Fixes / Choices
---------------
- Anchoring with fallback when anchors are pruned (to keep DE/ENT factor orientation stable).
- Shock defined as AR(1) residual with entity FE only.
- LP includes lags of the shock itself; NOT lags of the level.
- Only two LP specs are kept: Baseline and WithCtrls (lags of OTHER shocks). The older WithCtrls_FEh0 is removed.
- IRF summary: Sector × Shock × Spec → Peak_beta, h_peak, P_at_peak, #Sig_horizons_p<0.05, Cum_0_5, Nobs_min.

Inputs & Structure
------------------
• Dataset Excel: must include panel keys CountryShort, Year and GVA_* sector variables.
• Indicator table Excel: must include the mapping of indicators to blocks with columns:
  - 'Indicator (abbr.)'
  - 'Indicator Category'  (values expected: 'Digital Economy' or 'Entrepreneurship')
• The dataset must contain the indicator columns referenced by the indicator table.

Outputs
-------
• Excel workbook (OUT_XLSX) with sheets:
  - all_regressions           : FE results (coefficients, clustered SEs, CIs, p-values, FDR, VIFs, R², diagnostics)
  - Sensitivity_DK            : Driscoll–Kraay (Bartlett) sensitivity across bandwidths
  - Diagnostics_Summary       : one row per sector with compact diagnostics
  - Loadings_DigitalEconomy   : SPCA loadings for DE indicators
  - Loadings_Entrepreneurship : SPCA loadings for ENT indicators
  - Variance_Explained        : SPCA "energy" shares and total explained energy
  - Block_Vars_Selected       : indicators kept after variance and correlation pruning
  - IRF_LP_CLUSTERED          : LP-IRF estimates (if IRF_ENABLE=True)
  - IRF_Summary_CLUSTERED     : compact IRF summary (if IRF_ENABLE=True)
• Figures:
  - IRF_PLOTS_CLUSTERED/IRF_<Sector>_<Spec>_GRID_{raw|cum}.{png,svg}  (if IRF_ENABLE=True)

Configuration Guide
-------------------
• BASE_PATH / file names: set paths below to your local folders.
• SPCA:
  - N_FACTORS      : number of components per block (min(n_features, N_FACTORS))
  - VAR_THRESHOLD  : variance threshold for pruning (VarianceThreshold)
  - CORR_THRESHOLD : pairwise absolute correlation pruning threshold
  - SPCA_ALPHA     : SparsePCA L1 weight
• DK_BWS          : Bartlett bandwidths for Driscoll–Kraay sensitivity on FE
• Anchoring:
  - ORIENT_FACTORS : set True to orient factors by anchor correlations (with fallback)
• LP-IRF:
  - IRF_ENABLE           : master switch (True to run LP)
  - IRF_HORIZONS         : list of horizons
  - IRF_LAGS_Y           : # of lags of Y in LP
  - IRF_LAGS_SHOCK       : # of lags of the SAME shock in LP
  - IRF_WINSORIZE_SHOCKS : winsorize shock innovations at IRF_WINSOR_Q tails
  - IRF_BASE_SHOCKS_ONLY : if True, use only 6 base factors as shocks
  - IRF_MIN_COVERAGE     : min share of rows with valid shock & lag to keep a factor as shock
  - IRF_SIG_ALPHA        : alpha for gray shading (segment significant if either endpoint p<alpha)
  - IRF_PLOTS_DIR        : output folder for IRF plots
• Randomness: SEED controls SparsePCA reproducibility.

Dependencies
------------
numpy, pandas, scikit-learn, statsmodels, linearmodels, scipy, matplotlib

How to Run
----------
1) Set BASE_PATH below so the DATASET_XLSX and INDICATOR_XLSX are found.
2) (Optional) Toggle IRF_ENABLE to False if you want a fast pass without IRFs/plots.
3) Run with Python 3.10+:
   python your_script_name.py
4) Check OUT_XLSX and (if enabled) IRF plots in IRF_PLOTS_DIR.

Notes
-----
• The indicator table drives which columns (present in the dataset) are used per block.
• Interaction terms are kept for completeness but are NOT used as shocks in LP.
• FDR uses Benjamini–Hochberg across coefficients per sector.

"""

import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparsePCA
from sklearn.feature_selection import VarianceThreshold

from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm, chi2

import matplotlib.pyplot as plt
import textwrap

# ---------------------- PATHS & CONFIG ----------------------
#Edit respectively
BASE_PATH = r"C:\Users\.................."

DATASET_XLSX   = os.path.join(BASE_PATH, "data", "DatasetMasterCAP.xlsx")
INDICATOR_XLSX = os.path.join(BASE_PATH, "PhD", "data", "Indicator_Table.xlsx")
OUT_XLSX       = os.path.join(BASE_PATH, "data", "SPCA_Panel_ALL_IN_ONE_WITH_TABS8.xlsx")

for p in [DATASET_XLSX, INDICATOR_XLSX]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing file: {p}")

# SPCA / Reduction
N_FACTORS      = 3
VAR_THRESHOLD  = 0.01
CORR_THRESHOLD = 0.90
SPCA_ALPHA     = 1.0

# DK bandwidths (Bartlett) — for main FE sensitivity
DK_BWS = [2, 3, 4]

# Anchoring
ORIENT_FACTORS = True

# --- LP-IRF ---
IRF_ENABLE             = True
IRF_HORIZONS           = [0, 1, 2, 3, 4, 5]
IRF_LAGS_Y             = 1
IRF_LAGS_SHOCK         = 1   # lags of the SHOCK (not level)
IRF_WINSORIZE_SHOCKS   = True
IRF_WINSOR_Q           = 0.01
IRF_BASE_SHOCKS_ONLY   = True
IRF_MIN_COVERAGE       = 0.10
IRF_SIG_ALPHA          = 0.10
IRF_PLOTS_DIR          = os.path.join(BASE_PATH, "data", "IRF_PLOTS_CLUSTERED")
os.makedirs(IRF_PLOTS_DIR, exist_ok=True)

# LP specs: Baseline (only main shock) & WithCtrls (lags of other shocks)
IRF_SPECS = [
    {"name": "Baseline",   "with_ctrls_shock_lags": False},
    {"name": "WithCtrls",  "with_ctrls_shock_lags": True},
]

SEED = 0
np.random.seed(SEED)

# ---------------------- Helpers ----------------------
def _clean_for_vif(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.replace([np.inf, -np.inf], np.nan)
    Xc = Xc.dropna(axis=1, how='all')
    if Xc.shape[1] == 0: return Xc
    Xc = Xc.dropna(axis=0, how='any')
    if Xc.shape[1] == 0 or Xc.shape[0] == 0: return Xc
    variances = Xc.var(axis=0, ddof=1)
    zero_var_cols = variances[~np.isfinite(variances) | (variances <= 0)].index.tolist()
    if zero_var_cols: Xc = Xc.drop(columns=zero_var_cols)
    return Xc

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    Xc = _clean_for_vif(X)
    if Xc.shape[1] <= 1:
        return pd.DataFrame({'Variable': Xc.columns, 'VIF': [np.nan] * Xc.shape[1]})
    rows = []
    X_const = sm.add_constant(Xc, has_constant='add')
    for i in range(1, X_const.shape[1]):
        try:
            vif = variance_inflation_factor(X_const.values, i)
        except Exception:
            vif = np.nan
        rows.append({'Variable': Xc.columns[i-1], 'VIF': float(vif)})
    return pd.DataFrame(rows)

def pesaran_cd_from_resids(resids: pd.Series):
    """
    Pesaran CD test from FE residuals Series with MultiIndex (entity,time).
    """
    try:
        R = resids.copy()
        R.index = pd.MultiIndex.from_tuples(R.index)  # (entity, time)
        R = R.sort_index()
        Rdf = R.unstack(level=0)  # rows=time, cols=entity
        Rdf = Rdf.dropna(how='any')
        if Rdf.shape[1] < 2 or Rdf.shape[0] < 2:
            return np.nan, np.nan
        C = Rdf.corr()
        N = C.shape[0]
        vals = C.values
        s = 0.0; k = 0
        for i in range(N):
            for j in range(i+1, N):
                if np.isfinite(vals[i, j]): s += vals[i, j]; k += 1
        T = Rdf.shape[0]
        if k == 0: return np.nan, np.nan
        CD = np.sqrt(2 * T / (N * (N - 1))) * s
        p = 2 * (1 - norm.cdf(abs(CD)))
        return float(CD), float(p)
    except Exception:
        return np.nan, np.nan

def reduce_block_vars(block_df: pd.DataFrame, vars_list, var_thr=0.01, corr_thr=0.9):
    """
    1) Mean-impute → 2) Standardize → 3) VarianceThreshold → 4) Corr-pruning.
    Returns (info_dict, X_reduced).
    """
    info = {'original_vars': list(vars_list), 'after_varthr_vars': [], 'after_corr_vars': []}
    if not vars_list: return info, pd.DataFrame()
    imp = SimpleImputer(strategy='mean')
    Xi = pd.DataFrame(imp.fit_transform(block_df[vars_list]), columns=vars_list, index=block_df.index)
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(Xi), columns=vars_list, index=block_df.index)
    sel = VarianceThreshold(threshold=var_thr)
    Xv = sel.fit_transform(Xs)
    keep_idx = np.where(sel.get_support())[0].tolist()
    after_varthr = [vars_list[i] for i in keep_idx]
    info['after_varthr_vars'] = after_varthr
    if not after_varthr: return info, pd.DataFrame()
    Xv_df = pd.DataFrame(Xv, columns=after_varthr, index=block_df.index)
    corr = Xv_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > corr_thr)]
    final_vars = [c for c in after_varthr if c not in to_drop]
    info['after_corr_vars'] = final_vars
    Xfinal = Xv_df[final_vars].copy()
    return info, Xfinal

def spca_block_named(Xblock: pd.DataFrame, n_factors=3, seed=0, name_prefix="DE"):
    """
    Run SparsePCA for a block; return (F_df, loadings_df, variance_explained_df).
    """
    if Xblock.shape[1] == 0:
        return pd.DataFrame(index=Xblock.index), pd.DataFrame(), pd.DataFrame()
    n_components = min(n_factors, Xblock.shape[1])
    sp = SparsePCA(n_components=n_components, alpha=SPCA_ALPHA, random_state=seed)
    T = sp.fit_transform(Xblock.values)         # (n × k) factors
    C = sp.components_                          # (k × p) loadings'
    X = Xblock.values
    Xhat = T @ C
    denom = float(np.linalg.norm(X, 'fro')**2); denom = max(denom, 1e-12)
    ev_total = float(np.linalg.norm(Xhat, 'fro')**2 / denom)
    shares = []
    for k in range(n_components):
        Xhat_k = np.outer(T[:, k], C[k, :])
        shares.append(float(np.linalg.norm(Xhat_k, 'fro')**2 / denom))
    factor_names = [f"{name_prefix}_Factor_{i+1}" for i in range(n_components)]
    F_df = pd.DataFrame(T, index=Xblock.index, columns=factor_names)
    loadings = pd.DataFrame(C.T, index=Xblock.columns, columns=factor_names)
    block_label = "Digital Economy" if name_prefix == "DE" else "Entrepreneurship"
    var_rows = []
    for nm, sh in zip(factor_names, shares):
        var_rows.append({'Block': block_label, 'Factor': nm,
                         'SPCA_Energy_Share': sh, 'SPCA_EV_Total': np.nan})
    var_rows.append({'Block': block_label, 'Factor': '(total)',
                     'SPCA_Energy_Share': np.nan, 'SPCA_EV_Total': ev_total})
    var_exp = pd.DataFrame(var_rows)
    return F_df, loadings, var_exp

def stars_from_p(p_fdr):
    if pd.isna(p_fdr): return ""
    if p_fdr < 0.01: return "***"
    if p_fdr < 0.05: return "**"
    if p_fdr < 0.10: return "*"
    return ""

def fisher_adf_on_resids(resids: pd.Series):
    """
    Fisher-aggregated panel ADF on residuals (c and ct).
    """
    try:
        r = resids.copy()
        r.index = pd.MultiIndex.from_tuples(r.index)
        r = r.sort_index(level=[0,1])
    except Exception:
        return np.nan, np.nan, np.nan, np.nan
    pvals_c, pvals_ct = [], []
    for ent, s in r.groupby(level=0):
        x = s.droplevel(0).sort_index().dropna().values
        if x.size >= 5:
            try: pvals_c.append(adfuller(x, regression='c',  autolag='AIC')[1])
            except Exception: pass
            try: pvals_ct.append(adfuller(x, regression='ct', autolag='AIC')[1])
            except Exception: pass
    def fisher(pv):
        k = len(pv)
        if k == 0: return np.nan, np.nan
        stat = -2.0 * float(np.sum(np.log(pv)))
        p = chi2.sf(stat, 2*k)
        return float(stat), float(p)
    stat_c,  p_c  = fisher(pvals_c)
    stat_ct, p_ct = fisher(pvals_ct)
    return stat_c, p_c, stat_ct, p_ct

# ---------------------- Anchoring utils ----------------------
def _zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    z = (df - df.mean())/df.std(ddof=0)
    return z.replace([np.inf, -np.inf], np.nan)

def orient_factors_and_loadings(F_block: pd.DataFrame,
                                L_block: pd.DataFrame,
                                X_block: pd.DataFrame,
                                anchor_candidates: list,
                                df_full: pd.DataFrame = None):
    """
    Stabilize factor signs against an "anchor" (z-mean across available anchors).
    Fallback: if anchors were pruned in the reduced block, use anchors from df_full.
    """
    if not ORIENT_FACTORS or F_block.empty:
        return F_block, L_block

    anchors = [c for c in (anchor_candidates or []) if (not X_block.empty and c in X_block.columns)]
    X_anchor = X_block[anchors] if anchors else None

    if (X_anchor is None or X_anchor.empty) and (df_full is not None):
        anchors = [c for c in (anchor_candidates or []) if c in df_full.columns]
        X_anchor = df_full[anchors] if anchors else None

    if X_anchor is None or X_anchor.empty:
        return F_block, L_block

    A = _zscore_df(X_anchor).mean(axis=1)
    F2 = F_block.copy(); L2 = L_block.copy()
    for col in F2.columns:
        pair = pd.concat([F2[col], A], axis=1).dropna()
        if pair.shape[0] == 0:
            continue
        corr = pair.corr().iloc[0,1]
        if pd.notna(corr) and corr < 0:
            F2[col] = -F2[col]
            if col in L2.columns:
                L2[col] = -L2[col]
    return F2, L2

# ---------------------- LOAD ----------------------
df = pd.read_excel(DATASET_XLSX)
imap = pd.read_excel(INDICATOR_XLSX)
imap = imap[imap['Indicator (abbr.)'].notna()].copy()
imap['Indicator Category'] = imap['Indicator Category'].astype(str)

INDICATOR_NAME_COL     = 'Indicator (abbr.)'
INDICATOR_CATEGORY_COL = 'Indicator Category'

digital_cols = [c for c in imap.loc[imap[INDICATOR_CATEGORY_COL]=='Digital Economy', INDICATOR_NAME_COL] if c in df.columns]
entre_cols   = [c for c in imap.loc[imap[INDICATOR_CATEGORY_COL]=='Entrepreneurship', INDICATOR_NAME_COL] if c in df.columns]

# ---------------------- SPCA once (GLOBAL, unsupervised) ----------------------
dig_info, Xdig = reduce_block_vars(df, digital_cols, VAR_THRESHOLD, CORR_THRESHOLD)
ent_info, Xent = reduce_block_vars(df, entre_cols,   VAR_THRESHOLD, CORR_THRESHOLD)

F_dig, L_dig, V_dig = spca_block_named(Xdig, N_FACTORS, seed=SEED, name_prefix="DE")
F_ent, L_ent, V_ent = spca_block_named(Xent, N_FACTORS, seed=SEED, name_prefix="ENT")

DE_ANCHORS  = ['GII_Infrastr','GII_Businsoph','EIS_Intellectualassind','EIS_Emplinknowl','EIS_Knowledgintens','EIS_Patentappl']
ENT_ANCHORS = ['GEM_TotaEarlEntr','GEM_EntrInte','GEM_PercCapa','EIS_Venturcapexp','GEM_PostSchoEntr','GEM_PercOppo']

# Anchor orientation with fallback
F_dig, L_dig = orient_factors_and_loadings(F_dig, L_dig, Xdig, DE_ANCHORS,  df_full=df)
F_ent, L_ent = orient_factors_and_loadings(F_ent, L_ent, Xent, ENT_ANCHORS, df_full=df)

# Interactions after anchoring (kept for completeness; not used as IRF shocks)
interactions = {}
for c1 in F_dig.columns:
    for c2 in F_ent.columns:
        interactions[f"{c1}_X_{c2}"] = F_dig[c1] * F_ent[c2]
F_int = pd.DataFrame(interactions, index=df.index) if interactions else pd.DataFrame(index=df.index)

F_all = pd.concat([F_dig, F_ent, F_int], axis=1)

# ---------------------- Loop sectors (MAIN FE Regressions) ----------------------
gva_cols = [c for c in df.columns if c.startswith("GVA_")]

rows_all, dk_rows_all, diag_rows = [], [], []

for sector in gva_cols:
    d = df[['CountryShort','Year',sector]].dropna().copy().set_index(['CountryShort','Year'])
    if d.empty:
        continue
    Xf = F_all.copy()
    Xf['CountryShort'] = df['CountryShort']; Xf['Year'] = df['Year']
    Xf = Xf.set_index(['CountryShort','Year']).loc[d.index].copy()
    y = np.arcsinh(d[sector].astype(float))
    X = Xf.copy()
    nunq = X.nunique(); to_drop = nunq[nunq <= 1].index.tolist()
    if to_drop:
        X = X.drop(columns=to_drop)
    if X.shape[1] == 0 or len(y) < X.shape[1] + 5:
        continue

    fe = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
    res = fe.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)

    # DK sensitivity
    dk_res = {}
    for bw in DK_BWS:
        try:
            dk_res[bw] = fe.fit(cov_type='kernel', kernel='bartlett', bandwidth=bw)
        except Exception:
            dk_res[bw] = None

    # Diagnostics
    try:
        dw_val = float(sm.stats.stattools.durbin_watson(res.resids.values))
    except Exception:
        try: dw_val = float(sm.stats.stattools.durbin_watson(res.resids))
        except Exception: dw_val = np.nan

    try:
        X_bp = sm.add_constant(X, has_constant='add')
        if X_bp.shape[1] >= 2:
            bp_stat, bp_p, _, _ = het_breuschpagan(res.resids, X_bp)
            bp_pval = float(bp_p)
        else:
            bp_pval = np.nan
    except Exception:
        bp_pval = np.nan

    try:
        cd_stat, cd_p = pesaran_cd_from_resids(res.resids)
    except Exception:
        cd_stat, cd_p = np.nan, np.nan

    fadf_stat_c, fadf_p_c, fadf_stat_ct, fadf_p_ct = fisher_adf_on_resids(res.resids)
    nonstat_c  = (fadf_p_c  >= 0.05) if np.isfinite(fadf_p_c)  else np.nan
    nonstat_ct = (fadf_p_ct >= 0.05) if np.isfinite(fadf_p_ct) else np.nan
    nonstat_c_01  = (1 if nonstat_c  is True else (0 if nonstat_c  is False else np.nan))
    nonstat_ct_01 = (1 if nonstat_ct is True else (0 if nonstat_ct is False else np.nan))

    pooled_vif_df = calculate_vif(X) if X.shape[1] >= 2 else pd.DataFrame()
    try:
        Xe = X.groupby(level=0).transform('mean')
        Xt = X.groupby(level=1).transform('mean')
        Xg = X.mean()
        Xw_final = X - Xe - Xt + Xg
        within_vif_df = calculate_vif(Xw_final) if Xw_final.shape[1] >= 2 else pd.DataFrame()
    except Exception:
        within_vif_df = pd.DataFrame()

    params, ses, pvals, ci = res.params, res.std_errors, res.pvalues, res.conf_int()
    try:
        _, p_adj, _, _ = multipletests(pvals.values, method='fdr_bh')
        p_fdr = pd.Series(p_adj, index=pvals.index)
    except Exception:
        p_fdr = pd.Series([np.nan]*len(pvals), index=pvals.index)

    # DK long
    for bw, r in dk_res.items():
        if r is None:
            continue
        for var in params.index:
            dk_rows_all.append({
                'Sector': sector, 'Bandwidth': bw, 'Variable': var,
                'DK_Coefficient': float(r.params.get(var, np.nan)),
                'DK_Std_Error': float(r.std_errors.get(var, np.nan)),
                'DK_P_Value': float(r.pvalues.get(var, np.nan))
            })

    r2_w = float(res.rsquared_within)  if res.rsquared_within  is not None else np.nan
    r2_b = float(res.rsquared_between) if res.rsquared_between is not None else np.nan
    r2_o = float(res.rsquared_overall) if res.rsquared_overall is not None else np.nan

    for var in params.index:
        rows_all.append({
            'Sector': sector, 'Variable': var,
            'Coefficient': float(params[var]),
            'Std_Error_Clustered': float(ses[var]),
            'P_Value_Clustered': float(pvals[var]),
            'P_FDR': float(p_fdr[var]) if not pd.isna(p_fdr[var]) else np.nan,
            'Stars_FDR': stars_from_p(p_fdr[var]) if var in p_fdr.index else "",
            'CI_Lower': float(ci.loc[var, 'lower']), 'CI_Upper': float(ci.loc[var, 'upper']),
            'DK_p_bw2': float(dk_res[2].pvalues.get(var, np.nan)) if dk_res.get(2) is not None else np.nan,
            'DK_p_bw3': float(dk_res[3].pvalues.get(var, np.nan)) if dk_res.get(3) is not None else np.nan,
            'DK_p_bw4': float(dk_res[4].pvalues.get(var, np.nan)) if dk_res.get(4) is not None else np.nan,
            'Pooled_VIF':  float(pooled_vif_df.set_index('Variable').loc[var, 'VIF']) if (not pooled_vif_df.empty and var in pooled_vif_df['Variable'].values) else np.nan,
            'Within_VIF':  float(within_vif_df.set_index('Variable').loc[var, 'VIF']) if (not within_vif_df.empty and var in within_vif_df['Variable'].values) else np.nan,
            'R2_Within': r2_w, 'R2_Between': r2_b, 'R2_Overall': r2_o,
            'DurbinWatson': dw_val, 'BreuschPagan_p': bp_pval, 'PesaranCD_p': float(cd_p) if cd_p is not None else np.nan,
            'Fisher_ADF_resid_stat_c': float(fadf_stat_c) if np.isfinite(fadf_stat_c) else np.nan,
            'Fisher_ADF_resid_p_c':   float(fadf_p_c)    if np.isfinite(fadf_p_c)    else np.nan,
            'Nonstationary_resid_c':   nonstat_c,  'Nonstationary_resid_c_01': nonstat_c_01,
            'Fisher_ADF_resid_stat_ct':float(fadf_stat_ct) if np.isfinite(fadf_stat_ct) else np.nan,
            'Fisher_ADF_resid_p_ct':  float(fadf_p_ct)   if np.isfinite(fadf_p_ct)   else np.nan,
            'Nonstationary_resid_ct':  nonstat_ct, 'Nonstationary_resid_ct_01': nonstat_ct_01,
        })

    sig_fdr = (p_fdr < 0.05).sum() if p_fdr.notna().any() else np.nan
    diag_rows.append({
        'Sector': sector, 'Num_Coefficients': int(len(params)),
        'R2_Within': r2_w, 'R2_Between': r2_b, 'R2_Overall': r2_o,
        'DurbinWatson': dw_val, 'BreuschPagan_p': bp_pval,
        'PesaranCD_p': float(cd_p) if cd_p is not None else np.nan,
        'Fisher_ADF_resid_p_c':  float(fadf_p_c)  if np.isfinite(fadf_p_c)  else np.nan,
        'Nonstationary_resid_c': nonstat_c,
        'Fisher_ADF_resid_p_ct': float(fadf_p_ct) if np.isfinite(fadf_p_ct) else np.nan,
        'Nonstationary_resid_ct': nonstat_ct,
        'Sig_Coeff_FDR<0.05': int(sig_fdr) if not pd.isna(sig_fdr) else np.nan
    })

# ---------------------- LP-IRFs (ONLY CLUSTERED P) ----------------------
def _panel_index(df, ei='CountryShort', ti='Year'):
    return df.dropna(subset=[ei, ti]).set_index([ei, ti]).sort_index()

def _lag(df, col, k):  return df.groupby(level=0)[col].shift(k)
def _lead(df, col, k): return df.groupby(level=0)[col].shift(-k)

def _winsorize_ser(s, q=0.01):
    if s.empty: return s
    lo, hi = s.quantile(q), s.quantile(1-q)
    return s.clip(lo, hi)

def _effective_rows_for_shock(pnl, s_col):
    tmp = pnl[[s_col]].copy()
    tmp[f"L1_{s_col}"] = _lag(pnl, s_col, 1)
    return int(tmp.dropna().shape[0])

def _shock_residual(pnl, s_col):
    """
    Shock innovation for factor s_col: AR(1) residual with entity FE ONLY.
    """
    base = pnl[[s_col]].copy()
    base[f"L1_{s_col}"] = _lag(pnl, s_col, 1)
    base = base.dropna(subset=[s_col, f"L1_{s_col}"])
    if base.empty:
        return pd.Series(dtype=float, name=f"shock_{s_col}")
    yS = base[s_col].astype(float); XS = base[[f"L1_{s_col}"]].astype(float)
    try:
        m = PanelOLS(yS, XS, entity_effects=True, drop_absorbed=True)
        r = m.fit(cov_type='clustered', cluster_entity=True)
        u = (yS - r.predict(XS)).rename(f"shock_{s_col}")
        return u
    except Exception:
        X = sm.add_constant(XS.astype(float))
        ols = sm.OLS(yS.astype(float), X, missing='drop').fit()
        yhat = ols.predict(X)
        return (yS - yhat).rename(f"shock_{s_col}")

def _lp_irf_clustered(pnl, y_col, s_col, shock_series, horizons,
                      Ly=1, Ls=1, spec_dict=None,
                      ctrl_shock_series_map=None):
    """
    Jordà LP: Y_{i,t+h} = beta_h * shock_{i,t} + lags(Y) + lags(shock)
    + if spec_dict['with_ctrls_shock_lags']: lags of OTHER shocks.
    FE: entity + time. SE: clustered (entity,time) when available.
    """
    rows = []
    common = pnl.join(shock_series, how='inner')
    if common.empty:
        return rows

    shname = shock_series.name  # "shock_<factor>"

    for h in horizons:
        work = common.copy()
        work['Y_lead'] = _lead(work, y_col, h).astype(float) if h>0 else work[y_col].astype(float)

        # lags of Y
        ctrl_dyn = []
        for l in range(1, Ly+1):
            lc = f"L{l}_{y_col}"
            work[lc] = _lag(work, y_col, l)
            ctrl_dyn.append(lc)

        # lags of the SAME shock
        for l in range(1, Ls+1):
            lc = f"L{l}_{shname}"
            work[lc] = _lag(work, shname, l)
            ctrl_dyn.append(lc)

        X_cols = [shname] + ctrl_dyn

        # lags of OTHER shocks (if requested)
        if spec_dict and spec_dict.get("with_ctrls_shock_lags", False) and ctrl_shock_series_map:
            for nm, ser in ctrl_shock_series_map.items():
                work = work.join(ser, how='left')
                for l in range(1, Ls+1):
                    lc = f"L{l}_{nm}"
                    work[lc] = _lag(work, nm, l)
                    X_cols.append(lc)

        need = ['Y_lead', shname] + [c for c in X_cols if c != shname]
        need = list(dict.fromkeys(need))
        work = work.dropna(subset=need)
        if work.empty:
            continue

        yH  = work['Y_lead'].astype(float)
        XH  = work[X_cols].apply(pd.to_numeric, errors='coerce').astype(float)
        XH  = XH.dropna(axis=1, how='any')

        try:
            m = PanelOLS(yH, XH, entity_effects=True, time_effects=True, drop_absorbed=True)
            res = m.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
        except Exception:
            m = PanelOLS(yH, XH, entity_effects=True, drop_absorbed=True)
            res = m.fit(cov_type='clustered', cluster_entity=True)

        b  = float(res.params.get(shname, np.nan))
        se = float(res.std_errors.get(shname, np.nan))
        p  = float(res.pvalues.get(shname, np.nan))
        ci = res.conf_int()
        lo = float(ci.loc[shname, 'lower']) if shname in ci.index else np.nan
        hi = float(ci.loc[shname, 'upper']) if shname in ci.index else np.nan

        rows.append({
            'Sector': y_col,
            'Shock': s_col,
            'Horizon': h,
            'Spec': spec_dict['name'] if spec_dict else 'Baseline',
            'Beta': b,
            'SE_Clustered': se,
            'P_Clustered': p,
            'CI_Lower': lo,
            'CI_Upper': hi,
            'N_Obs': int(len(yH))
        })
    return rows

# --- Build panel & IRFs ---
IRF_df = pd.DataFrame(); IRF_summ_df = pd.DataFrame()

if IRF_ENABLE:
    base = _panel_index(df[['CountryShort', 'Year'] + gva_cols].copy())
    Fx = F_all.copy()
    Fx['CountryShort'] = df['CountryShort']; Fx['Year'] = df['Year']
    Fx = _panel_index(Fx)
    panel = base.join(Fx, how='left')

    # Only base DE/ENT factors as shocks if requested
    candidates = [c for c in F_all.columns if c.startswith("DE_Factor_") or c.startswith("ENT_Factor_")] \
                 if IRF_BASE_SHOCKS_ONLY else list(F_all.columns)

    total_rows = len(panel)
    eligible = []
    for s in candidates:
        if s not in panel.columns:
            continue
        eff = _effective_rows_for_shock(panel, s)
        if eff / max(1, total_rows) >= IRF_MIN_COVERAGE:
            eligible.append(s)
    if not eligible:
        eligible = [c for c in candidates if _effective_rows_for_shock(panel, c) > 0]

    # shocks
    shock_series_map = {}  # key: shock_<factor>
    factor_of_shock  = {}  # inverse map
    for s in eligible:
        u = _shock_residual(panel, s)
        if not u.empty:
            if IRF_WINSORIZE_SHOCKS:
                u = _winsorize_ser(u, q=IRF_WINSOR_Q)
            shock_name = f"shock_{s}"
            u.name = shock_name
            shock_series_map[shock_name] = u
            factor_of_shock[shock_name]  = s

    # LP
    IRF_rows = []
    for sector in gva_cols:
        if sector not in panel.columns:
            continue
        pnl = panel.copy()
        pnl[sector] = np.arcsinh(pnl[sector].astype(float))
        for shock_name, shock_u in shock_series_map.items():
            s_factor = factor_of_shock[shock_name]
            other_shocks = {nm: ser for nm, ser in shock_series_map.items() if nm != shock_name}

            for spec in IRF_SPECS:
                rows = _lp_irf_clustered(
                    pnl, y_col=sector, s_col=s_factor, shock_series=shock_u,
                    horizons=IRF_HORIZONS, Ly=IRF_LAGS_Y, Ls=IRF_LAGS_SHOCK,
                    spec_dict=spec,
                    ctrl_shock_series_map=(other_shocks if spec.get("with_ctrls_shock_lags", False) else None),
                )
                IRF_rows.extend(rows)

    IRF_df = pd.DataFrame(IRF_rows) if IRF_rows else pd.DataFrame(
        columns=['Sector','Shock','Horizon','Spec','Beta','SE_Clustered','P_Clustered','CI_Lower','CI_Upper','N_Obs'])
    if not IRF_df.empty:
        IRF_df = IRF_df.sort_values(['Sector','Shock','Spec','Horizon'])
        IRF_df['Beta_cum'] = IRF_df.groupby(['Sector','Shock','Spec'])['Beta'].cumsum()

# ---------------------- IRF Small-Multiples (2×3) per sector ----------------------
GRID_ROWS, GRID_COLS = 2, 3
IRF_IMAGE_DPI   = 300
IRF_FONT_SIZE   = 12
IRF_LINEWIDTH   = 3.2
IRF_MARKERSIZE  = 7.0
IRF_NOTE        = "Notes: Gray shading along line segments indicates p<0.10."

BASE_FACTORS = [
    "DE_Factor_1","DE_Factor_2","DE_Factor_3",
    "ENT_Factor_1","ENT_Factor_2","ENT_Factor_3"
]
FACTOR_LABELS = {
    "DE_Factor_1":  "DE 1. Institutional Readiness & Access Foundations",
    "DE_Factor_2":  "DE 2. Digital Inclusion & Affordability Constraint",
    "DE_Factor_3":  "DE 3. Digital Diffusion & Usage Intensity",
    "ENT_Factor_1": "ENT 1. Knowledge Capital, R&D Investment & Innovation Linkages",
    "ENT_Factor_2": "ENT 2. Business Climate, Regulatory Burden & Market Openness",
    "ENT_Factor_3": "ENT 3. Entrepreneurial Drive, Early-Stage Activity & SME Dynamism",
}

def _wrap_title(txt, width=46):
    return "\n".join(textwrap.wrap(txt, width=width, break_long_words=False))

def _plot_sector_grid(irf_df_sector, sector_name, spec_label="Baseline", use_cumulative=True):
    d = irf_df_sector.copy()
    d = d[(d["Spec"]==spec_label) & (d["Shock"].isin(BASE_FACTORS))]
    if d.empty:
        return

    if use_cumulative and "Beta_cum" in d.columns:
        d["Yplot"] = d["Beta_cum"]; ylbl_extra = " (cumulative)"
    else:
        d["Yplot"] = d["Beta"];     ylbl_extra = ""

    plt.rcParams.update({"font.size": IRF_FONT_SIZE})
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    y_clean = sector_name.replace("GVA_", "").replace("_", " ").strip()
    y_label = f"GVA {y_clean} Response{ylbl_extra}"

    y_min = d["Yplot"].min(skipna=True)
    y_max = d["Yplot"].max(skipna=True)
    if np.isfinite(y_min) and np.isfinite(y_max):
        pad = 0.05*(y_max - y_min if y_max>y_min else max(abs(y_max),1e-6))
        y_lims = (y_min - pad, y_max + pad)
    else:
        y_lims = None

    for ax, shock in zip(axes, BASE_FACTORS):
        ds = d[d["Shock"]==shock].sort_values("Horizon")
        ax.set_title(_wrap_title(FACTOR_LABELS.get(shock, shock)), fontsize=IRF_FONT_SIZE+1)
        ax.grid(True, which="major", alpha=0.25, linestyle="--", linewidth=0.6)

        if not ds.empty:
            H = ds["Horizon"].to_numpy()
            Y = ds["Yplot"].to_numpy()
            P = ds["P_Clustered"].to_numpy()

            # Gray shading on segments where at least one endpoint is significant at alpha
            for i in range(len(H)-1):
                if (P[i] < IRF_SIG_ALPHA) or (P[i+1] < IRF_SIG_ALPHA):
                    ax.plot([H[i], H[i+1]], [Y[i], Y[i+1]],
                            color="0.70", linewidth=IRF_LINEWIDTH+8.5,
                            alpha=1.0, solid_capstyle="round", zorder=1)

            # Main black line
            ax.plot(H, Y, marker="x", color="black",
                    linewidth=IRF_LINEWIDTH, markersize=IRF_MARKERSIZE, zorder=2)

            # Zero line
            ax.axhline(0, color="black", linewidth=0.9, zorder=0)

            # Rings at significant points
            sig = ds["P_Clustered"] < IRF_SIG_ALPHA
            if sig.any():
                ax.scatter(ds.loc[sig,"Horizon"], ds.loc[sig,"Yplot"],
                           s=IRF_MARKERSIZE*16, facecolors="none",
                           edgecolors="black", linewidths=1.2, zorder=3)

        ax.set_xlim(min(IRF_HORIZONS), max(IRF_HORIZONS))
        ax.set_xticks(list(range(int(min(IRF_HORIZONS)), int(max(IRF_HORIZONS))+1)))
        if y_lims: ax.set_ylim(*y_lims)

    for k in range(len(BASE_FACTORS), len(axes)):
        axes[k].axis("off")

    fig.supxlabel("Horizon", y=0.04, fontsize=IRF_FONT_SIZE+1)
    fig.supylabel(y_label, x=0.005, fontsize=IRF_FONT_SIZE+1)
    fig.text(0.5, 0.01, IRF_NOTE, ha="center", va="bottom")
    fig.tight_layout(rect=[0.03, 0.05, 1.0, 0.97])

    safe_sector = sector_name.replace(":", "-").replace("/", "-")
    tag = "cum" if use_cumulative else "raw"
    fname = f"IRF_{safe_sector}_{spec_label}_GRID_{tag}"
    fig.savefig(os.path.join(IRF_PLOTS_DIR, f"{fname}.png"), dpi=IRF_IMAGE_DPI, bbox_inches="tight")
    fig.savefig(os.path.join(IRF_PLOTS_DIR, f"{fname}.svg"), dpi=IRF_IMAGE_DPI, bbox_inches="tight")
    plt.close(fig)

# ---------------------- Create IRF Plots (Baseline & WithCtrls) ----------------------
if IRF_ENABLE and not IRF_df.empty:
    if "Beta_cum" not in IRF_df.columns:
        IRF_df = IRF_df.sort_values(["Sector","Shock","Spec","Horizon"])
        IRF_df["Beta_cum"] = IRF_df.groupby(["Sector","Shock","Spec"])["Beta"].cumsum()
    for sec, dsec in IRF_df.groupby("Sector"):
        for spec in [s["name"] for s in IRF_SPECS]:
            _plot_sector_grid(dsec, sector_name=sec, spec_label=str(spec), use_cumulative=True)
            _plot_sector_grid(dsec, sector_name=sec, spec_label=str(spec), use_cumulative=False)

# ---------------------- Assemble SPCA metadata sheets ----------------------
loadings_dig = L_dig.copy()
loadings_ent = L_ent.copy()
var_exp_df = pd.concat([V_dig, V_ent], axis=0, ignore_index=True) if (not V_dig.empty or not V_ent.empty) else pd.DataFrame()

def _block_info_to_long(block_name, info_dict):
    rows = []
    rows.append({'Block': block_name, 'Stage': 'Original',          'Count': len(info_dict.get('original_vars', [])), 'Variables': ", ".join(info_dict.get('original_vars', []))})
    rows.append({'Block': block_name, 'Stage': 'After_VarThreshold','Count': len(info_dict.get('after_varthr_vars', [])), 'Variables': ", ".join(info_dict.get('after_varthr_vars', []))})
    rows.append({'Block': block_name, 'Stage': 'After_CorrPruning', 'Count': len(info_dict.get('after_corr_vars', [])), 'Variables': ", ".join(info_dict.get('after_corr_vars', []))})
    return rows

block_sel_rows = []
block_sel_rows += _block_info_to_long('DigitalEconomy',  dig_info)
block_sel_rows += _block_info_to_long('Entrepreneurship', ent_info)
block_sel_df = pd.DataFrame(block_sel_rows)

# ---------------------- SAVE ----------------------
with pd.ExcelWriter(OUT_XLSX) as writer:
    # Main regressions
    if rows_all:
        all_df = pd.DataFrame(rows_all)
        main_cols = [
            'Sector','Variable',
            'Coefficient','Std_Error_Clustered','P_Value_Clustered','P_FDR','Stars_FDR',
            'CI_Lower','CI_Upper',
            'DK_p_bw2','DK_p_bw3','DK_p_bw4',
            'Pooled_VIF','Within_VIF',
            'R2_Within','R2_Between','R2_Overall',
            'DurbinWatson','BreuschPagan_p','PesaranCD_p',
            'Fisher_ADF_resid_stat_c','Fisher_ADF_resid_p_c','Nonstationary_resid_c','Nonstationary_resid_c_01',
            'Fisher_ADF_resid_stat_ct','Fisher_ADF_resid_p_ct','Nonstationary_resid_ct','Nonstationary_resid_ct_01'
        ]
        cols_final = [c for c in main_cols if c in all_df.columns] + [c for c in all_df.columns if c not in main_cols]
        all_df = all_df[cols_final]
        all_df.to_excel(writer, sheet_name="all_regressions", index=False)
    else:
        pd.DataFrame(columns=['Sector','Variable']).to_excel(writer, sheet_name="all_regressions", index=False)

    # DK sensitivity
    dk_df = pd.DataFrame(dk_rows_all) if dk_rows_all else pd.DataFrame(
        columns=['Sector','Bandwidth','Variable','DK_Coefficient','DK_Std_Error','DK_P_Value'])
    dk_df.to_excel(writer, sheet_name="Sensitivity_DK", index=False)

    # Diagnostics
    diag_df = pd.DataFrame(diag_rows) if diag_rows else pd.DataFrame(columns=['Sector'])
    diag_df.to_excel(writer, sheet_name="Diagnostics_Summary", index=False)

    # Loadings
    if not loadings_dig.empty:
        loadings_dig.to_excel(writer, sheet_name="Loadings_DigitalEconomy")
    if not loadings_ent.empty:
        loadings_ent.to_excel(writer, sheet_name="Loadings_Entrepreneurship")

    # SPCA variance explained
    if not var_exp_df.empty:
        var_exp_df.to_excel(writer, sheet_name="Variance_Explained", index=False)

    # Block variable selection
    if not block_sel_df.empty:
        block_sel_df.to_excel(writer, sheet_name="Block_Vars_Selected", index=False)

    # IRFs (2 specs: Baseline, WithCtrls)
    if IRF_ENABLE:
        (IRF_df if not IRF_df.empty else pd.DataFrame(
            columns=['Sector','Shock','Horizon','Spec','Beta','SE_Clustered','P_Clustered','CI_Lower','CI_Upper','N_Obs','Beta_cum'])
         ).to_excel(writer, sheet_name="IRF_LP_CLUSTERED", index=False)

        # -------- IRF SUMMARY --------
        summ_rows = []
        if not IRF_df.empty:
            for (sec, shock, spec), g in IRF_df.groupby(["Sector","Shock","Spec"]):
                if g.empty:
                    continue
                g = g.dropna(subset=["Beta"])
                if g.empty:
                    continue
                # Peak absolute beta
                idx_peak = g["Beta"].abs().idxmax()
                peak_beta = float(g.loc[idx_peak, "Beta"])
                h_peak    = int(g.loc[idx_peak, "Horizon"])
                # p at the same horizon (NaN if missing)
                try:
                    p_at_peak = float(g.loc[idx_peak, "P_Clustered"])
                except Exception:
                    p_at_peak = np.nan
                # count significant horizons
                sig_count = int((g["P_Clustered"] < 0.05).sum()) if "P_Clustered" in g.columns else np.nan
                # cumulative 0–5
                cum_0_5 = float(g.loc[g["Horizon"].between(0,5), "Beta"].sum())
                # minimum N obs used across horizons
                nobs_min = int(g["N_Obs"].min()) if "N_Obs" in g.columns else np.nan

                summ_rows.append({
                    "Sector": sec, "Shock": shock, "Spec": spec,
                    "Peak_beta": peak_beta, "h_peak": h_peak, "P_at_peak": p_at_peak,
                    "#Sig_horizons_p<0.05": sig_count,
                    "Cum_0_5": cum_0_5, "Nobs_min": nobs_min
                })

        IRF_summ_df = pd.DataFrame(summ_rows)
        IRF_summ_df.to_excel(writer, sheet_name="IRF_Summary_CLUSTERED", index=False)

print(f"Saved workbook with all sheets: {OUT_XLSX}")
print(f"IRF plots (clustered-only) saved under: {IRF_PLOTS_DIR}")
