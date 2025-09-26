# -*- coding: utf-8 -*-
"""
FISEI — Financial & Insurance Sector Digital Economy Index
==========================================================

Overview
--------
This script builds a **sector-specific composite index** for the Financial & Insurance
sector (FISEI) from country–year indicators. It produces a *baseline* (equal-weight)
composite and a *PCA-weighted* composite (within-pillar PC1 weights), with full
transparency on imputation, normalization, coverage rules, and diagnostics.

Pipeline
--------
1) **Load & Validate**: Read the SDEI sheet, ensure required columns exist.
2) **Imputation (ladder)**: For each indicator:
   - (a) Country-wise linear interpolation (±2-year gaps).
   - (b) Regional mean (per year) if `Region` exists and ≥10 non-missing.
   - (c) OECD mean (per year) if ≥10 non-missing.
   Flags are written per indicator: `is_imputed_<ind>` and `impute_method_<ind>`.
3) **Winsorize & Normalize (per year)**:
   - Winsorize at the tails (default: 1% / 99%) for robustness.
   - Scale each indicator to **0–100** either with min–max or ECDF rank scaling.
   - Reverse-sign *only* the Unemployment indicator (higher unemployment → lower score).
   - Normalized columns are named `<Indicator>_norm`.
4) **Pillars**: Compute 5 pillar scores as the **mean of available normalized indicators**,
   enforcing a **minimum coverage** within each pillar (default: 60%).
5) **Composite**:
   - **Baseline FISEI**: Equal-weight average of available pillar scores with **composite
     coverage** (default: 60% of pillars present).
   - **PCA FISEI**: Within each pillar, compute **PC1 weights** on pooled data across years
     (abs(loadings) normalized to sum to 1). Score each pillar as the weighted mean of its
     normalized indicators (row-wise handling of missing). Composite is the average of pillar
     PCA-scores, respecting the same composite coverage rule.
6) **Diagnostics & Metadata**:
   - Spearman correlation between Baseline FISEI and PCA FISEI overall and by year.
   - Per-year indicator missingness table.
   - PC1 variance shares per pillar.
   - A `Meta` sheet with runtime, versions, and reproducibility hash of the input file.

Inputs
------
• Excel file with sheet `SDEI` containing:
  - Panel keys: `Country`, `Year` (case-insensitive; auto-renamed)
  - Optional: `Region` (for regional means in the imputation ladder)
  - Indicators listed below under POSITIVE / NEGATIVE and grouped by pillars.

Outputs
-------
An Excel workbook with the following sheets:
• `FISEI_Base`      : Country–Year with pillars and **Baseline** FISEI (0–100)
• `FISEI_PCA`       : Country–Year with PCA pillar scores and **FISEI_PCA** (0–100)
• `PCA_Weights`     : Pillar-level PC1 weights (abs(loadings) normalized)
• `PCA_VarShare`    : PC1 variance share per pillar
• `Diagnostics`     : Spearman rho (overall + by year) between FISEI and FISEI_PCA
• `Missingness`     : Per-year missingness by indicator
• `Meta`            : Run timestamp, versions, input path + SHA256, and settings

Key Design Choices
------------------
• **Sign policy**: Only Unemployment is reverse-signed (as requested).
• **Scaling**: Choose `minmax` (default) or `rank` (ECDF percentiles in 0–100).
• **Coverage**: Pillar/Composite require a minimum fraction of available components.
• **Robustness**: Winsorization guards against outliers; ECDF option is rank-robust.
• **Reproducibility**: The input file SHA256 is stored in `Meta`.

How to Run
----------
1) Adjust the paths in **Settings & Constants** below.
2) (Optional) Tweak knobs (winsorization, scaling method, coverage thresholds).
3) Install dependencies:
   pip install pandas numpy scipy xlsxwriter openpyxl
4) Run:
   python fisei_build.py
5) Open the output workbook `FISEI_results.xlsx`.

Notes
-----
• If a pillar has only one indicator, the code handles it gracefully (weight=1 in PCA).
• If all values are flat/degenerate in an indicator-year slice, the scaler returns **50**.
• Regional step in the imputation ladder is used **only** if `Region` exists in data.
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

# ===============================
# 0) Settings & Constants
# ===============================
#Edit respectively
DATA_PATH = r"C:\Users\.................."
SHEET_NAME = "SDEI"
OUT_XLSX = r"C:\Users\Ggouskos\Desktop\PhD\DATA\FISEI_results.xlsx"

# --- Easy, useful knobs ---
WINSOR = 0.01                 # 1–99% winsorization (set 0.0 to disable)
SCALE_METHOD = "minmax"       # {"minmax","rank"}; rank=ECDF percentile scaling
PILLAR_MIN_COVER = 0.60       # min fraction of indicators needed per pillar
COMPOSITE_MIN_COVER = 0.60    # min fraction of pillars needed for composite
REGION_COL = "Region"         # if exists in data, used for regional means

# ===============================
# Indicators for Financial & Insurance Sector (12 total)
# Sign: only Unemployment is reverse-signed (as requested)
# ===============================
POSITIVE: List[str] = [
    "DII_HumaDeveInde",   # Human Development Index
    "CNTRL_LifeExpeBirt", # Life expectancy at birth
    "IMD_InteUserIndi",   # Individuals using the Internet
    "DETF_TeleCompInfo",  # Telecom, computer & info services
    "CNTRL_InlfAnnuGrow", # Inflation (kept positive as per user's note)
    "DETF_NarrDigiDivi",  # Narrowing the digital divide (Internet users)
    "DESI_OverFixeBroa",  # Overall fixed broadband take-up
    "CNTRL_HousDebt",     # Household debt
    "CNTRL_PurcPowePari", # PPP
    "DII_HousNpisFina",   # Households & NPISHs final consumption exp.
    "CNTRL_AverSalaAnnu", # Average salary annual
]
NEGATIVE: List[str] = [
    "CNTRL_UnemRate",     # Unemployment rate — reverse-signed
]

# ===============================
# Pillars (as discussed)
# P1–P5 purposely include 1-indicator pillars (P2, P5); the code handles that.
# ===============================
PILLARS: Dict[str, List[str]] = {
    "P1_DigitalAccessConnectivity": [
        "IMD_InteUserIndi",
        "DESI_OverFixeBroa",
        "DETF_NarrDigiDivi",
    ],
    "P2_DigitalServicesInfra": [
        "DETF_TeleCompInfo",
    ],
    "P3_HumanCapitalSocioecon": [
        "DII_HumaDeveInde",
        "CNTRL_LifeExpeBirt",
        "CNTRL_PurcPowePari",
        "CNTRL_AverSalaAnnu",
    ],
    "P4_FinancialStabilityHouseholds": [
        "CNTRL_HousDebt",
        "DII_HousNpisFina",
        "CNTRL_UnemRate",  # reverse-signed (handled in normalization)
    ],
    "P5_MacroFinancialEnvironment": [
        "CNTRL_InlfAnnuGrow",
    ],
}
ALL_INDICATORS: List[str] = POSITIVE + NEGATIVE

# (Optional list retained; not used unless you want a regional slice later)
EUROPEAN_OECD = [
    "Austria","Belgium","Czech Republic","Denmark","Estonia","Finland","France","Germany",
    "Greece","Hungary","Iceland","Ireland","Italy","Latvia","Lithuania","Luxembourg",
    "Netherlands","Norway","Poland","Portugal","Slovak Republic","Slovenia","Spain",
    "Sweden","Switzerland","Turkey","United Kingdom"
]

# ===============================
# 1) Utils
# ===============================
def file_sha256(path: str) -> str:
    """Compute SHA256 of a file for reproducibility logging."""
    try:
        with open(path, "rb") as f:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""

def check_required_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    """Return missing columns from `required`."""
    return [c for c in required if c not in df.columns]

def winsorize_series(s: pd.Series, p: float) -> pd.Series:
    """Winsorize at p / (1-p) quantiles; if p<=0, just coerce to numeric."""
    if p <= 0:
        return pd.to_numeric(s, errors="coerce")
    s = pd.to_numeric(s, errors="coerce")
    ql, qh = np.nanpercentile(s, p*100), np.nanpercentile(s, (1-p)*100)
    return s.clip(ql, qh)

def scale_0_100(s: pd.Series, method: str = "minmax") -> pd.Series:
    """Scale numeric series to [0,100] by min–max or rank (ECDF). Degenerate → 50."""
    s = pd.to_numeric(s, errors="coerce")
    if method == "rank":
        r = s.rank(method="average", na_option="keep")
        if r.dropna().nunique() <= 1:
            out = pd.Series(50.0, index=s.index)  # flat → neutral
        else:
            out = 100.0 * (r - r.min()) / (r.max() - r.min())
        out[s.isna()] = np.nan
        return out
    # min–max
    lo, hi = np.nanmin(s.values), np.nanmax(s.values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(50.0, index=s.index)  # flat/degenerate → neutral
    return 100.0 * (s - lo) / (hi - lo)

# ===============================
# 2) Imputation ladder (easy & transparent)
# ===============================
def impute_ladder(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    """
    Impute missing values in three steps:
      1) Intra-country linear interpolation (limit 2 years).
      2) Regional mean per year if REGION_COL exists and group size >= 10.
      3) OECD mean per year if count >= 10.

    Flags created:
      - is_imputed_<ind>  ∈ {0,1}
      - impute_method_<ind> ∈ {"interp","region","oecd", None}
    """
    df = df.copy()
    # Ensure flags exist
    for ind in indicators:
        df[f"is_imputed_{ind}"] = 0
        df[f"impute_method_{ind}"] = None

    # Step 1: interpolation within country
    for ind in indicators:
        if ind not in df:
            continue

        def _interp(g: pd.DataFrame) -> pd.Series:
            s = pd.to_numeric(g[ind], errors="coerce")
            s = s.reindex(g.sort_values("Year").index)
            si = s.interpolate(method="linear", limit=2, limit_direction="both")
            si = si.reindex(g.index)
            return si

        df["_tmp"] = df.groupby("Country", group_keys=False).apply(_interp)
        filled_mask = df[ind].isna() & df["_tmp"].notna()
        df.loc[filled_mask, ind] = df.loc[filled_mask, "_tmp"]
        df.loc[filled_mask, f"is_imputed_{ind}"] = 1
        df.loc[filled_mask, f"impute_method_{ind}"] = "interp"
        df = df.drop(columns=["_tmp"])

    # Step 2: regional mean (if Region exists)
    has_region = REGION_COL in df.columns
    for ind in indicators:
        if ind not in df:
            continue
        mask = df[ind].isna()
        if has_region:
            grp = df.groupby(["Year", REGION_COL])[ind].transform(
                lambda s: s.mean() if s.notna().sum() >= 10 else np.nan
            )
            fill_mask = mask & grp.notna()
            df.loc[fill_mask, ind] = grp[fill_mask]
            df.loc[fill_mask, f"is_imputed_{ind}"] = 1
            df.loc[fill_mask, f"impute_method_{ind}"] = "region"

    # Step 3: OECD/year mean
    for ind in indicators:
        if ind not in df:
            continue
        mask = df[ind].isna()
        means = df.groupby("Year")[ind].transform(
            lambda s: s.mean() if s.notna().sum() >= 10 else np.nan
        )
        fill_mask = mask & means.notna()
        df.loc[fill_mask, ind] = means[fill_mask]
        df.loc[fill_mask, f"is_imputed_{ind}"] = 1
        method_col = f"impute_method_{ind}"
        df.loc[fill_mask & df[method_col].isna(), method_col] = "oecd"

    return df

# ===============================
# 3) Normalization per year (with sign)
# ===============================
def normalize_by_year_with_sign(
    df: pd.DataFrame,
    year_col: str,
    pos_cols: List[str],
    neg_cols: List[str],
    winsor: float = WINSOR,
    method: str = SCALE_METHOD
) -> pd.DataFrame:
    """Winsorize and scale indicators to 0–100 per year, reversing NEGATIVE ones."""
    df_norm = df.copy()
    for y, g in df.groupby(year_col):
        idx = g.index
        # positive
        for col in pos_cols:
            if col not in df_norm:
                continue
            s = winsorize_series(g[col], winsor)
            df_norm.loc[idx, f"{col}_norm"] = scale_0_100(s, method=method)
        # negative (reverse)
        for col in neg_cols:
            if col not in df_norm:
                continue
            s = winsorize_series(g[col], winsor)
            df_norm.loc[idx, f"{col}_norm"] = 100.0 - scale_0_100(s, method=method)
    return df_norm

# ===============================
# 4) Pillar & Composite (coverage checks)
# ===============================
def compute_pillars(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pillar means with pillar-level coverage requirement."""
    df = df.copy()
    for p_name, inds in PILLARS.items():
        n_needed = max(1, int(np.ceil(PILLAR_MIN_COVER * len(inds))))
        norm_cols = [f"{c}_norm" for c in inds if f"{c}_norm" in df.columns]
        vals = df[norm_cols]
        counts = vals.notna().sum(axis=1)
        mask_ok = counts >= n_needed
        mean_vals = vals.mean(axis=1)
        mean_vals[~mask_ok] = np.nan
        df[p_name] = mean_vals
    return df

def compute_composite(df: pd.DataFrame, pillar_names: List[str], out_col: str) -> pd.DataFrame:
    """Average across pillars with composite-level coverage requirement."""
    df = df.copy()
    n_needed = max(1, int(np.ceil(COMPOSITE_MIN_COVER * len(pillar_names))))
    vals = df[pillar_names]
    counts = vals.notna().sum(axis=1)
    mask_ok = counts >= n_needed
    comp = vals.mean(axis=1)
    comp[~mask_ok] = np.nan
    df[out_col] = comp
    return df

# ===============================
# 5) PCA weighting (pooled years) + PC1 variance share
# ===============================
def _zscore_matrix(M: np.ndarray) -> np.ndarray:
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=0)
    Z = (M - mu) / sd
    Z[:, (sd == 0) | np.isnan(sd)] = 0.0
    return Z

def pca_weights_for_pillar(df: pd.DataFrame, pillar_cols_norm: List[str]) -> Tuple[Dict[str, float], float]:
    """
    Compute **abs(PC1 loadings)** as weights (normalized to sum=1) on pooled rows,
    and return the PC1 variance share for reporting.
    """
    X = df[pillar_cols_norm].copy()
    col_means = X.mean(axis=0, skipna=True)
    X_imp = X.fillna(col_means).to_numpy(dtype=float)
    if X_imp.shape[1] == 1:
        return {pillar_cols_norm[0]: 1.0}, 1.0
    Z = _zscore_matrix(X_imp)
    try:
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        loadings = Vt[0, :]
        eigvals = (S**2) / (Z.shape[0]-1) if Z.shape[0] > 1 else S**2
        var_share = float(eigvals[0] / eigvals.sum()) if np.isfinite(eigvals).all() and eigvals.sum() > 0 else np.nan
    except np.linalg.LinAlgError:
        loadings = np.ones(X_imp.shape[1], dtype=float)
        var_share = np.nan
    w = np.abs(loadings)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w)/len(w)
    return {c: float(wi) for c, wi in zip(pillar_cols_norm, w)}, var_share

def compute_pillars_and_index_pca(df: pd.DataFrame, out_col: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Compute PCA-weighted pillars and composite.
    Returns:
      - res: DataFrame with Country, Year, pillar_PCA columns, and `out_col`
      - weights_dict: mapping Pillar → {indicator_norm: weight}
      - varshare_dict: mapping Pillar → PC1 variance share
    """
    res = df[["Country","Year"]].copy()
    weights_dict: Dict[str, Dict[str, float]] = {}
    varshare_dict: Dict[str, float] = {}
    pcols = []
    for p_name, inds in PILLARS.items():
        norm_cols = [f"{c}_norm" for c in inds if f"{c}_norm" in df.columns]
        if not norm_cols:
            res[p_name + "_PCA"] = np.nan
            continue
        w_map, vs = pca_weights_for_pillar(df, norm_cols)
        weights_dict[p_name] = w_map
        varshare_dict[p_name] = vs

        W = np.array([w_map[c] for c in norm_cols], dtype=float)
        X = df[norm_cols].to_numpy(dtype=float)
        mask = ~np.isnan(X)
        W_row = np.where(mask, W, 0.0)
        W_sum = W_row.sum(axis=1, keepdims=True)
        W_safe = np.divide(W_row, W_sum, out=np.zeros_like(W_row), where=W_sum > 0)
        score = np.nansum(W_safe * np.nan_to_num(X), axis=1)
        res[p_name + "_PCA"] = score
        pcols.append(p_name + "_PCA")

    # Composite with coverage
    res["_count_pillars"] = res[pcols].notna().sum(axis=1)
    need = max(1, int(np.ceil(COMPOSITE_MIN_COVER * len(pcols))))
    comp = res[pcols].mean(axis=1)
    comp[res["_count_pillars"] < need] = np.nan
    res = res.drop(columns=["_count_pillars"])
    res[out_col] = comp
    return res, weights_dict, varshare_dict

# ===============================
# 6) Diagnostics
# ===============================
def spearman_by_year(df_all: pd.DataFrame, base_col: str, pca_col: str) -> pd.DataFrame:
    """Overall and by-year Spearman correlation between two index variants."""
    try:
        from scipy.stats import spearmanr
    except Exception:
        return pd.DataFrame([{"Year":"Overall","Spearman_rho":np.nan,"p_value":np.nan,"N":0}])
    rows = []
    x, y = df_all[base_col], df_all[pca_col]
    m = x.notna() & y.notna()
    if m.sum() >= 3:
        rho, p = spearmanr(x[m], y[m])
        rows.append({"Year":"Overall","Spearman_rho":rho,"p_value":p,"N":int(m.sum())})
    for yv, g in df_all.groupby("Year"):
        xv, yv2 = g[base_col], g[pca_col]
        mm = xv.notna() & yv2.notna()
        if mm.sum() >= 3:
            rho, p = spearmanr(xv[mm], yv2[mm])
            rows.append({"Year":int(yv),"Spearman_rho":rho,"p_value":p,"N":int(mm.sum())})
    return pd.DataFrame(rows)

def missingness_report(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    """Per-year missingness counts and rates for each indicator."""
    rows = []
    for y, g in df.groupby("Year"):
        for ind in indicators:
            n = g[ind].shape[0] if ind in g else 0
            miss = g[ind].isna().sum() if ind in g else n
            rows.append({"Year": y, "Indicator": ind, "N": n, "Missing": miss, "MissingRate": (miss/n) if n>0 else np.nan})
    return pd.DataFrame(rows)

# ===============================
# 7) Save
# ===============================
def save_results(out_path: str,
                 df_final: pd.DataFrame,
                 df_pca: pd.DataFrame = None,
                 weights_dict: Dict[str, Dict[str, float]] = None,
                 df_diag: pd.DataFrame = None,
                 df_varshare: pd.DataFrame = None,
                 df_missing: pd.DataFrame = None,
                 meta: Dict[str, str] = None):
    """Write all outputs to a single Excel workbook with tidy sheets."""
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df_final.to_excel(writer, sheet_name="FISEI_Base", index=False)
        if df_pca is not None:
            df_pca.to_excel(writer, sheet_name="FISEI_PCA", index=False)
        if weights_dict is not None:
            rows = []
            for p, mp in weights_dict.items():
                for col, w in mp.items():
                    rows.append({"Pillar": p, "Indicator_norm": col, "PCA_weight": w})
            pd.DataFrame(rows).to_excel(writer, sheet_name="PCA_Weights", index=False)
        if df_varshare is not None:
            df_varshare.to_excel(writer, sheet_name="PCA_VarShare", index=False)
        if df_diag is not None:
            df_diag.to_excel(writer, sheet_name="Diagnostics", index=False)
        if df_missing is not None:
            df_missing.to_excel(writer, sheet_name="Missingness", index=False)
        if meta is not None:
            pd.DataFrame([meta]).to_excel(writer, sheet_name="Meta", index=False)

# ===============================
# 8) Main
# ===============================
def main():
    # Load
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df_raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    # Rename basic cols (case-insensitive)
    rename_map = {}
    for c in df_raw.columns:
        cl = str(c).strip()
        if cl.lower() == "year":
            rename_map[c] = "Year"
        if cl.lower() == "country":
            rename_map[c] = "Country"
    df = df_raw.rename(columns=rename_map)

    # Required columns check
    required = ["Country","Year"] + ALL_INDICATORS
    missing = check_required_columns(df, required)
    if missing:
        raise ValueError("❌ Missing required columns:\n" + "\n".join("  - "+m for m in missing))

    # Impute (ladder)
    df_imp = impute_ladder(df, ALL_INDICATORS)

    # Normalize per year with sign
    df_norm = normalize_by_year_with_sign(df_imp, "Year", POSITIVE, NEGATIVE,
                                          winsor=WINSOR, method=SCALE_METHOD)

    # Pillars + Composite (baseline)
    df_pill = compute_pillars(df_norm)
    pillar_names = list(PILLARS.keys())
    df_base = compute_composite(df_pill, pillar_names, out_col="FISEI")

    # PCA pillars + composite
    df_pca, weights_dict, varshare_dict = compute_pillars_and_index_pca(df_norm, out_col="FISEI_PCA")
    df_all = df_base.merge(df_pca, on=["Country","Year"], how="left")

    # Diagnostics
    df_diag = spearman_by_year(df_all, base_col="FISEI", pca_col="FISEI_PCA")
    df_miss = missingness_report(df, ALL_INDICATORS)
    df_varshare = pd.DataFrame(
        [{"Pillar": p, "PC1_Variance_Share": vs} for p, vs in varshare_dict.items()]
    )

    # Meta (reproducibility + settings)
    meta = {
        "RunTimestamp": datetime.now().isoformat(timespec="seconds"),
        "Python": sys.version.split()[0],
        "Pandas": pd.__version__,
        "Numpy": np.__version__,
        "InputSheet": SHEET_NAME,
        "InputPath": DATA_PATH,
        "InputSHA256": file_sha256(DATA_PATH),
        "ScalingMethod": SCALE_METHOD,
        "Winsor": str(WINSOR),
        "PillarMinCover": str(PILLAR_MIN_COVER),
        "CompositeMinCover": str(COMPOSITE_MIN_COVER),
        "RegionColPresent": str(REGION_COL in df.columns),
        "IndexName": "FISEI (Financial & Insurance Sector Digital Economy Index)",
    }

    # Save
    Path(Path(OUT_XLSX).parent).mkdir(parents=True, exist_ok=True)
    save_results(
        OUT_XLSX,
        df_final=df_all,
        df_pca=df_pca,
        weights_dict=weights_dict,
        df_diag=df_diag,
        df_varshare=df_varshare,
        df_missing=df_miss,
        meta=meta
    )

    print("✅ FISEI computed (Baseline + PCA).")
    print(f"➡ Results: {OUT_XLSX}")
    print("Sheets: FISEI_Base, FISEI_PCA, PCA_Weights, PCA_VarShare, Diagnostics, Missingness, Meta.")

if __name__ == "__main__":
    main()
