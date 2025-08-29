# --- Path normalization for GitHub repo ---
# Using relative paths so it runs anywhere:
# - dataset: data/raw/DatasetMasterCAP.xlsx
# - indicator table: references/Indicator_Table.xlsx
# - outputs: data/processed/, figures/

# === Expected dataset columns (first 7 at minimum) ===
# Year, CountryShort, Country, GVA_Total, GVA_Agric_Total, GVA_Mining_Total, GVA_Manuf_Total
# Adjust names below if your file uses different headers.


import pandas as pd
import numpy as np
import os
import re
import matplotlib
matplotlib.use('Agg')      # <--- Use non-GUI backend to avoid Tk errors
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparsePCA
from sklearn.feature_selection import VarianceThreshold
from statsmodels.tsa.api import VAR
import warnings

# --- Settings ---
DATASET_FILE_PATH = "data/raw/DatasetMasterCAP.xlsx"
INDICATOR_TABLE_PATH = "references/Indicator_Table.xlsx"
DATE_COLUMN_NAME = "Year"
PANEL_ID_COLUMN = "CountryShort"
OUTPUT_EXCEL_FILE = r'C:\Users\Ggouskos\Desktop\data\FAVAR_FINAL_RESULTS.xlsx'
MODEL_SUMMARY_TXT = 'Model_Summary_Full.txt'
N_FACTORS = 3
VAR_THRESHOLD = 0.01
CORR_THRESHOLD = 0.9
ALPHA_SPCA = 1.0
PLOT_FOLDER = "figures/irf"

warnings.simplefilter('ignore')

os.makedirs('data/processed', exist_ok=True)
os.makedirs('figures/irf', exist_ok=True)

def safe_filename(s):
    return re.sub(r'[^A-Za-z0-9_\-]', '_', str(s))

# Create plot output folder if it does not exist
os.makedirs(PLOT_FOLDER, exist_ok=True)

print("--- Step 1: Load Data ---")
df = pd.read_excel(DATASET_FILE_PATH)
indicator_map_df = pd.read_excel(INDICATOR_TABLE_PATH)
indicator_map_df = indicator_map_df[indicator_map_df['Indicator (abbr.)'].notna()]
indicator_map_df['Indicator Category'] = indicator_map_df['Indicator Category'].astype(str)

gva_columns = [col for col in df.columns if col.startswith('GVA_')]
INDICATOR_NAME_COL = 'Indicator (abbr.)'
INDICATOR_CATEGORY_COL = 'Indicator Category'
digital_economy_columns = indicator_map_df[indicator_map_df[INDICATOR_CATEGORY_COL] == 'Digital Economy'][INDICATOR_NAME_COL].tolist()
entrepreneurship_columns = indicator_map_df[indicator_map_df[INDICATOR_CATEGORY_COL] == 'Entrepreneurship'][INDICATOR_NAME_COL].tolist()
digital_economy_columns = [col for col in digital_economy_columns if col in df.columns]
entrepreneurship_columns = [col for col in entrepreneurship_columns if col in df.columns]
informational_columns = digital_economy_columns + entrepreneurship_columns

# --- PANEL index: [Year, CountryShort] ---
df = df.set_index([DATE_COLUMN_NAME, PANEL_ID_COLUMN])
df = df.sort_index()

# --- Step 2: Make GVA stationary (panel diff) ---
slow_moving_stationary = df[gva_columns].groupby(PANEL_ID_COLUMN, group_keys=False).diff().dropna(how='all')

# --- Prepare SPCA input (already stationary) ---
de_cols = [col for col in digital_economy_columns if col in df.columns]
ent_cols = [col for col in entrepreneurship_columns if col in df.columns]
X_de = df[de_cols]
X_ent = df[ent_cols]

# --- SPCA block ---
def block_spca(X, n_factors=3, var_thr=0.01, corr_thr=0.9, prefix='X', alpha=1.0):
    X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns, index=X.index)
    selector = VarianceThreshold(threshold=var_thr)
    scaled = StandardScaler().fit_transform(X)
    X_var_selected = selector.fit_transform(scaled)
    selected_vars = [X.columns[i] for i in range(len(X.columns)) if selector.get_support()[i]]
    if len(selected_vars) == 0:
        return None, None, None, []
    X_var_df = pd.DataFrame(X_var_selected, columns=selected_vars, index=X.index)
    corr_matrix = X_var_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_thr)]
    final_vars = [col for col in selected_vars if col not in to_drop]
    if len(final_vars) == 0:
        return None, None, None, []
    X_final = X_var_df[final_vars]
    scaled_final = StandardScaler().fit_transform(X_final)
    n_components = min(n_factors, scaled_final.shape[1])
    if n_components == 0:
        return None, None, None, []
    spca = SparsePCA(n_components=n_components, alpha=alpha, random_state=0)
    factors_arr = spca.fit_transform(scaled_final)
    factors_df = pd.DataFrame(factors_arr, index=X.index, columns=[f'{prefix}_Factor_{i+1}' for i in range(n_components)])
    loadings_df = pd.DataFrame(spca.components_.T, index=X_final.columns, columns=factors_df.columns)
    # Variance explained (reconstruction)
    total_var = np.sum(np.var(scaled_final, axis=0))
    var_explained = []
    for i in range(spca.n_components):
        Z = np.zeros_like(factors_arr)
        Z[:, i] = factors_arr[:, i]
        X_reconst = np.dot(Z, spca.components_)
        single_var = np.sum(np.var(X_reconst, axis=0))
        var_explained.append(single_var)
    var_ratio = np.array(var_explained) / total_var
    cum_ratio = np.cumsum(var_ratio)
    variance_df = pd.DataFrame({
        'Factor': [f'{prefix}_Factor_{i+1}' for i in range(n_components)],
        'Variance_Explained': var_ratio,
        'Cumulative': cum_ratio
    })
    return factors_df, loadings_df, variance_df, final_vars

# --- Step 3: Υπολογισμός Factors ---
factors_de, loadings_de, variance_de, _ = block_spca(X_de, n_factors=N_FACTORS, var_thr=VAR_THRESHOLD, corr_thr=CORR_THRESHOLD, prefix='DE', alpha=ALPHA_SPCA)
factors_ent, loadings_ent, variance_ent, _ = block_spca(X_ent, n_factors=N_FACTORS, var_thr=VAR_THRESHOLD, corr_thr=CORR_THRESHOLD, prefix='ENT', alpha=ALPHA_SPCA)

# --- Plot Variance Explained by Factors and save ---
if variance_de is not None:
    plt.figure(figsize=(6,4))
    plt.bar(variance_de['Factor'], variance_de['Variance_Explained'])
    plt.title("Variance Explained by Digital Economy Factors")
    plt.ylabel('Variance Explained')
    plt.xlabel('Factor')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "variance_de.png"))
    plt.close()

if variance_ent is not None:
    plt.figure(figsize=(6,4))
    plt.bar(variance_ent['Factor'], variance_ent['Variance_Explained'])
    plt.title("Variance Explained by Entrepreneurship Factors")
    plt.ylabel('Variance Explained')
    plt.xlabel('Factor')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "variance_ent.png"))
    plt.close()

# --- Step 4: Create Interactions ---
factors_inter = pd.DataFrame(index=factors_de.index)
for i in range(1, factors_de.shape[1]+1):
    for j in range(1, factors_ent.shape[1]+1):
        col_name = f'Interaction_DE{i}_ENT{j}'
        factors_inter[col_name] = factors_de[f'DE_Factor_{i}'] * factors_ent[f'ENT_Factor_{j}']

# --- Step 5: Join all factors ---
factors_final = pd.concat([factors_de, factors_ent, factors_inter], axis=1)

# --- Step 6: Prepare final model data ---
common_idx = slow_moving_stationary.index.intersection(factors_final.index)
Y = slow_moving_stationary.loc[common_idx]
X = factors_final.loc[common_idx]
model_data = pd.concat([Y, X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
print('Final model_data shape:', model_data.shape)

# --- Step 7: Fit VAR (FAVAR) με lags=1 ---
model = VAR(model_data)
model_fitted = model.fit(1)
print(model_fitted.summary())

# === STABILITY CHECK ===
print('\n--- VAR Stability Check ---')
if hasattr(model_fitted, "is_stable"):
    is_stable = model_fitted.is_stable(verbose=True)
    if not is_stable:
        print("WARNING: VAR model is NOT stable! Τα IRFs είναι αναξιόπιστα.")
    else:
        print("VAR model is stable. IRFs are valid.")
else:
    print("Stability check not available in this statsmodels version.")

# --- Υπολογισμός IRFs και confidence intervals, και SAVE ALL PLOTS ---
all_vars = model_data.columns
gva_vars = [col for col in all_vars if col.startswith('GVA_')]
factor_vars = [col for col in all_vars if col.startswith('DE_Factor_') or col.startswith('ENT_Factor_') or col.startswith('Interaction_')]
irf_res = model_fitted.irf(10)
periods = np.arange(irf_res.irfs.shape[0])
irf_dict = {}

for impulse_var in factor_vars:
    for response_var in gva_vars:
        impulse_idx = model_data.columns.get_loc(impulse_var)
        response_idx = model_data.columns.get_loc(response_var)
        irf_values = irf_res.irfs[:, response_idx, impulse_idx]
        stderr = np.sqrt(np.abs(irf_res.cov(orth=True)[:, response_idx, impulse_idx]))
        lci = irf_values - 1.96 * stderr
        uci = irf_values + 1.96 * stderr
        irf_dict[f'IRF_{safe_filename(response_var)}_to_{safe_filename(impulse_var)}'] = irf_values
        irf_dict[f'LCI_{safe_filename(response_var)}_to_{safe_filename(impulse_var)}'] = lci
        irf_dict[f'UCI_{safe_filename(response_var)}_to_{safe_filename(impulse_var)}'] = uci

        # Save IRF plot for each response-impulse pair
        plt.figure(figsize=(6,4))
        plt.plot(periods, irf_values, label='IRF')
        plt.fill_between(periods, lci, uci, color='blue', alpha=0.2, label='95% CI')
        plt.axhline(0, color='k', linestyle='--', linewidth=1)
        plt.title(f'IRF: {response_var} to {impulse_var}')
        plt.xlabel('Periods')
        plt.ylabel('Response')
        plt.legend()
        plt.tight_layout()
        fname = f"IRF_{safe_filename(response_var)}_to_{safe_filename(impulse_var)}.png"
        plt.savefig(os.path.join(PLOT_FOLDER, fname))
        plt.close()

print("Finished calculating all IRF results and saved plots to folder.")

# --- Step 10: Export results to Excel ---
print(f"--- Step 10: Export results to {OUTPUT_EXCEL_FILE} ---")
with pd.ExcelWriter(OUTPUT_EXCEL_FILE) as writer:
    if loadings_de is not None:
        loadings_de.to_excel(writer, sheet_name='SPCA_Loadings_DE')
    if loadings_ent is not None:
        loadings_ent.to_excel(writer, sheet_name='SPCA_Loadings_ENT')
    if variance_de is not None:
        variance_de.to_excel(writer, sheet_name='Variance_Explained_DE', index=False)
    if variance_ent is not None:
        variance_ent.to_excel(writer, sheet_name='Variance_Explained_ENT', index=False)
    if factors_de is not None:
        factors_de.to_excel(writer, sheet_name='Factors_DE')
    if factors_ent is not None:
        factors_ent.to_excel(writer, sheet_name='Factors_ENT')
    factors_inter.to_excel(writer, sheet_name='Factors_Interactions')
    model_data.to_excel(writer, sheet_name='ModelData_VAR')
    # VAR coefficients etc.
    for i, variable in enumerate(model_fitted.names):
        params = model_fitted.params.iloc[:, i]
        stderr = model_fitted.stderr.iloc[:, i]
        tvalues = model_fitted.tvalues.iloc[:, i]
        pvalues = model_fitted.pvalues.iloc[:, i]
        var_df = pd.DataFrame({
            'Coef.': params,
            'Std.Err.': stderr,
            't': tvalues,
            'P>|t|': pvalues
        })
        var_df.to_excel(writer, sheet_name=f'VAR_{variable[:20]}')
    # R^2
    rsquared_list = []
    for i, col in enumerate(model_fitted.names):
        y_pred = model_fitted.fittedvalues[col]
        idx = y_pred.index
        col_idx = list(model_fitted.names).index(col)
        y_true = pd.Series(model_fitted.model.endog[-len(idx):, col_idx], index=idx)
        if len(y_true) != len(y_pred):
            r2 = np.nan
        else:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        rsquared_list.append({'Variable': col, 'R_squared': r2})
    rsquared_df = pd.DataFrame(rsquared_list)
    rsquared_df.to_excel(writer, sheet_name='R_squared', index=False)
    # Residuals
    residuals_df = pd.DataFrame(model_fitted.resid, columns=model_fitted.names, index=model_fitted.resid.index)
    residuals_df.to_excel(writer, sheet_name='Residuals')
    # IRF σε ένα sheet (χωρίς plots, μόνο values!)
    irf_all_df = pd.DataFrame(irf_dict, index=periods)
    irf_all_df.index.name = 'Period'
    irf_all_df.to_excel(writer, sheet_name='IRF_ALL')
    # VIF Report placeholder
    vif_report_placeholder = pd.DataFrame({'Status': ['VIF filtering was not performed in this run.']})
    vif_report_placeholder.to_excel(writer, sheet_name='VIF_Report', index=False)

# --- Export model summary as txt
summary_text = str(model_fitted.summary())
with open(MODEL_SUMMARY_TXT, 'w', encoding='utf-8') as txtfile:
    txtfile.write("Model Summary\n")
    txtfile.write("="*60 + "\n")
    txtfile.write(summary_text)

print("\n--- Script Finished ---")
