import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd

# -------------------------------
# 0) Settings & Constants
# -------------------------------
# MAIN CHANGE: Specify the dataset path (sheet = "SDEI")
DATA_PATH = "data/raw/SDEI Dataset.xlsx"
SHEET_NAME = "SDEI"

# Output file name
OUT_XLSX = "data/processed/SDEI_Edu_resultsF.xlsx"

# Four pillars with EXACT indicator abbreviations
PILLARS: Dict[str, List[str]] = {
    "P1_ConnectivityAccess": [
        "DESI_OverFixeBroa",
        "IMD_MobiBroaSubs2",
        "DETF_RiseMobiBroa",
    ],
    "P2_UsageOnlineInteraction": [
        "ITU_IndiUsinInte",
        "MCI_MobiSociMedi",
        "DWB_OnliUsePubl",
        "DETF_Econ",
        "DII_PopuUsinInte",
    ],
    "P3_SkillsHumanCapital": [
        "DESI_IctGrad",
    ],
    "P4_InnovationKnowledge": [
        "DETF_RdInfoIndu",
        "DII_TaxeIncoProf",
    ],
}
ALL_INDICATORS = [ind for inds in PILLARS.values() for ind in inds]

# List of European OECD countries (for the EU-only slice)
EUROPEAN_OECD = [
    "Austria", "Belgium", "Czech Republic", "Denmark", "Estonia", "Finland",
    "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy",
    "Latvia", "Lithuania", "Luxembourg", "Netherlands", "Norway", "Poland",
    "Portugal", "Slovak Republic", "Slovenia", "Spain", "Sweden", "Switzerland",
    "Turkey", "United Kingdom"
]


# -------------------------------
# 1) Utility Functions
# -------------------------------
def check_required_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    """Return a list of missing required columns."""
    missing = [c for c in required if c not in df.columns]
    return missing

def minmax_by_year(df: pd.DataFrame, year_col: str, cols: List[str]) -> pd.DataFrame:
    """
    Apply Min–Max normalisation (0–100) by year (ensures cross-country comparability within the same year).
    If all values are NaN or max==min, return NaN.
    """
    df_norm = df.copy()
    for y, g in df.groupby(year_col):
        idx = g.index
        for col in cols:
            if col not in df_norm.columns:
                continue
            x = g[col].astype(float)
            xmin = np.nanmin(x.values)
            xmax = np.nanmax(x.values)
            if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                df_norm.loc[idx, f"{col}_norm"] = 100.0 * (x - xmin) / (xmax - xmin)
            else:
                df_norm.loc[idx, f"{col}_norm"] = np.nan
    return df_norm

def compute_pillars_and_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute:
      - Pillar scores (equal weights within each pillar)
      - Composite SDEI_Edu (equal weights across pillars)
    """
    # Pillars
    for p_name, inds in PILLARS.items():
        norm_cols = [f"{c}_norm" for c in inds if f"{c}_norm" in df.columns]
        if norm_cols:
            df[p_name] = df[norm_cols].mean(axis=1)
        else:
            df[p_name] = np.nan

    # Composite (equal weights across pillars)
    pillar_cols = list(PILLARS.keys())
    df["SDEI_Edu"] = df[pillar_cols].mean(axis=1)

    return df

def save_results(out_path: str, df_final: pd.DataFrame, df_eu_2022: pd.DataFrame = None):
    """Save results to Excel, including the EU-only 2022 slice if provided."""
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df_final.to_excel(writer, sheet_name="SDEI_Edu", index=False)
        if df_eu_2022 is not None:
            df_eu_2022.to_excel(writer, sheet_name="EU_2022", index=False)


# -------------------------------
# 2) Main Script
# -------------------------------
def main():
    # Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    # Standardise column names for Country/Year
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip()
        if cl.lower() == "year":
            rename_map[c] = "Year"
        if cl.lower() == "country":
            rename_map[c] = "Country"
        if cl.lower() == "countryshort":
            # Keep CountryShort if present, do not rename/remove
            pass
    df = df.rename(columns=rename_map)

    # Check required columns
    required_cols = ["Country", "Year"] + ALL_INDICATORS
    missing = check_required_columns(df, required_cols)
    if missing:
        msg = (
            "❌ Missing required columns in the dataset:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nPlease add them with EXACT names (abbreviations) and re-run."
        )
        raise ValueError(msg)

    # Min–Max normalisation by year
    df_norm = minmax_by_year(df, "Year", ALL_INDICATORS)

    # Compute pillars & composite index
    df_out = compute_pillars_and_index(df_norm)

    # Optional: European OECD slice for 2022 (fairer comparison)
    df_eu_2022 = (
        df_out.loc[(df_out["Year"] == 2022) & (df_out["Country"].isin(EUROPEAN_OECD))]
        .copy()
        .sort_values(by="SDEI_Edu", ascending=False)
    )

    # Save results
    out_dir = Path(OUT_XLSX).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    save_results(OUT_XLSX, df_out, df_eu_2022)

    # Console report
    print("✅ SDEI–Education computed successfully.")
    print(f"➡ Results saved to: {OUT_XLSX}")
    if not df_eu_2022.empty:
        print("➡ A European-only 2022 sheet (EU_2022) was also included for fairness due to non-EU data gaps.")
    else:
        print("ℹ No EU 2022 slice produced (no matching rows).")


if __name__ == "__main__":
    main()
