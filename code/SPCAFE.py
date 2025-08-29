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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparsePCA
from sklearn.feature_selection import VarianceThreshold
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import f
import os

# === Plot settings ===
PLOTS_FOLDER = "figures/spca"
os.makedirs(PLOTS_FOLDER, exist_ok=True)
sns.set(style="whitegrid")

def safe_sheet_name(name):
    return name[:31]

def wooldridge_test(data, depvar, entity='CountryShort', time='Year'):
    try:
        df = data.copy()
        df = df[[depvar]].dropna()
        df = df.sort_index()
        df['depvar_lag'] = df.groupby(entity)[depvar].shift(1)
        df = df.dropna()
        y = df[depvar] - df['depvar_lag']
        x = df['depvar_lag'] - df['depvar_lag'].groupby(df.index.get_level_values(entity)).shift(1)
        x = x.dropna()
        y = y.loc[x.index]
        if len(y) < 3:
            return np.nan, np.nan
        res = sm.OLS(y, x, hasconst=False).fit()
        fval = res.rsquared / (1 - res.rsquared)
        df1 = 1
        df2 = len(y) - 1
        pval = 1 - f.cdf(fval, df1, df2)
        return fval, pval
    except Exception as e:
        print("[Wooldridge] ERROR:", e)
        return np.nan, np.nan

# --- Settings ---
dataset_path = "data/raw/DatasetMasterCAP.xlsx"
indicator_path = "references/Indicator_Table.xlsx"
N_FACTORS = 3
LAG_ORDER = 1
VAR_THRESHOLD = 0.01
CORR_THRESHOLD = 0.9

# --- Load Data ---
indicator_map_df = pd.read_excel(indicator_path)
indicator_map_df = indicator_map_df[indicator_map_df['Indicator (abbr.)'].notna()]
indicator_map_df['Indicator Category'] = indicator_map_df['Indicator Category'].astype(str)
df = pd.read_excel(dataset_path)

INDICATOR_NAME_COL = 'Indicator (abbr.)'
INDICATOR_CATEGORY_COL = 'Indicator Category'

digital_economy_columns = indicator_map_df[indicator_map_df[INDICATOR_CATEGORY_COL] == 'Digital Economy'][INDICATOR_NAME_COL].tolist()
entrepreneurship_columns = indicator_map_df[indicator_map_df[INDICATOR_CATEGORY_COL] == 'Entrepreneurship'][INDICATOR_NAME_COL].tolist()
digital_economy_columns = [col for col in digital_economy_columns if col in df.columns]
entrepreneurship_columns = [col for col in entrepreneurship_columns if col in df.columns]

digital_blocks = {'DigitalEconomy': digital_economy_columns}
entre_blocks = {'Entrepreneurship': entrepreneurship_columns}

def spca_variance_explained(X_scaled, spca, factors_arr, prefix):
    X_hat = np.dot(factors_arr, spca.components_)
    total_var = np.sum(np.var(X_scaled, axis=0))
    var_explained = []
    for i in range(spca.n_components):
        Z = np.zeros_like(factors_arr)
        Z[:, i] = factors_arr[:, i]
        X_reconst = np.dot(Z, spca.components_)
        single_var = np.sum(np.var(X_reconst, axis=0))
        var_explained.append(single_var)
    var_explained = np.array(var_explained)
    var_ratio = var_explained / total_var
    cum_ratio = np.cumsum(var_ratio)
    return pd.DataFrame({
        'Factor': [f'{prefix}_Factor_{i+1}' for i in range(spca.n_components)],
        'Variance_Explained': var_ratio,
        'Cumulative': cum_ratio
    })

def process_block_spca(block_vars, df, N_FACTORS=2, var_thr=0.01, corr_thr=0.9, block_prefix='X'):
    factors_all = pd.DataFrame(index=df.index)
    loadings_all = {}
    var_exp_all = []
    for block_name, vars_list in block_vars.items():
        if not vars_list:
            continue
        block = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(df[vars_list]), columns=vars_list)
        selector = VarianceThreshold(threshold=var_thr)
        block_scaled = StandardScaler().fit_transform(block)
        block_var_selected = selector.fit_transform(block_scaled)
        selected_vars = [vars_list[i] for i in range(len(vars_list)) if selector.get_support()[i]]
        if not selected_vars:
            continue
        block_var_df = pd.DataFrame(block_var_selected, columns=selected_vars, index=block.index)
        corr_matrix = block_var_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_thr)]
        final_vars = [col for col in selected_vars if col not in to_drop]
        if not final_vars:
            continue
        block_final = block_var_df[final_vars]
        block_scaled_final = StandardScaler().fit_transform(block_final)
        n_components = min(N_FACTORS, block_final.shape[1])
        if n_components == 0:
            continue
        spca = SparsePCA(n_components=n_components, alpha=1.0, random_state=0)
        factors_arr = spca.fit_transform(block_scaled_final)
        factor_names = [f"{block_prefix}_Factor_{i+1}" for i in range(n_components)]
        factors_df = pd.DataFrame(factors_arr, columns=factor_names, index=block.index)
        spca.name_prefix = block_prefix
        var_exp_df = spca_variance_explained(block_scaled_final, spca, factors_arr, block_prefix)
        loadings_df = pd.DataFrame(spca.components_.T, index=block_final.columns, columns=factor_names)
        factors_all = pd.concat([factors_all, factors_df], axis=1)
        loadings_all[block_name] = loadings_df
        var_exp_all.append(var_exp_df)
        print(f"Block: {block_name} | Original: {len(vars_list)} cols, After reduction: {len(final_vars)} cols")
        print(f"Columns kept: {final_vars}")
    return factors_all, loadings_all, pd.concat(var_exp_all) if var_exp_all else pd.DataFrame()

gva_cols = [col for col in df.columns if col.startswith('GVA_')]

excel_path = "data/processed/SPCA_Results.xlsx"
all_results_list = []
all_diag_list = []
r2_results = []

# --- SPCA Extraction ---
digital_factors, digital_loadings, digital_var_exp = process_block_spca(
    digital_blocks, df, N_FACTORS, VAR_THRESHOLD, CORR_THRESHOLD, block_prefix='DE'
)
entre_factors, entre_loadings, entre_var_exp = process_block_spca(
    entre_blocks, df, N_FACTORS, VAR_THRESHOLD, CORR_THRESHOLD, block_prefix='ENT'
)
var_exp_df = pd.concat([digital_var_exp, entre_var_exp]).reset_index(drop=True)

# === PLOTS: Variance Explained ===
plt.figure(figsize=(10,6))
sns.barplot(data=var_exp_df, x='Factor', y='Variance_Explained')
plt.title('Variance Explained by SPCA Factors')
plt.xticks(rotation=45)
plt.ylabel('Proportion of Variance Explained')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_FOLDER, "variance_explained.png"))
plt.close()

# === PLOTS: SPCA Loadings Heatmaps ===
for name, loadings_dict in zip(['DigitalEconomy', 'Entrepreneurship'], [digital_loadings, entre_loadings]):
    for block, loadings in loadings_dict.items():
        plt.figure(figsize=(10, min(1+0.4*len(loadings), 12)))
        ax = sns.heatmap(
            loadings, annot=True, cmap="coolwarm", fmt=".3f",
            annot_kws={"size": 7}
        )
        plt.title(f'SPCA Loadings: {name} - {block}')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_FOLDER, f"{name}_{block}_loadings.png"), dpi=300)
        plt.close()

with pd.ExcelWriter(excel_path) as writer:
    for SECTOR in gva_cols:
        print(f"\nProcessing sector: {SECTOR}")

        panel_vars = ['CountryShort', 'Year', SECTOR]
        df_panel = df[panel_vars].dropna()
        df_panel = df_panel.set_index(['CountryShort', 'Year'])

        # Merge the factors by country-year
        for f_name, factors in zip(['digital_factors', 'entre_factors'], [digital_factors, entre_factors]):
            temp_factors = factors.copy()
            temp_factors['CountryShort'] = df['CountryShort']
            temp_factors['Year'] = df['Year']
            temp_factors.set_index(['CountryShort', 'Year'], inplace=True)
            if f_name == 'digital_factors':
                digital_factors_aligned = temp_factors.loc[df_panel.index]
            else:
                entre_factors_aligned = temp_factors.loc[df_panel.index]

        # Interaction Terms
        interaction_dict = {}
        interactions = []
        for i in range(digital_factors_aligned.shape[1]):
            for j in range(entre_factors_aligned.shape[1]):
                name = f'DE_Factor_{i+1}_X_ENT_Factor_{j+1}'
                interaction_dict[name] = digital_factors_aligned.iloc[:, i] * entre_factors_aligned.iloc[:, j]
                interactions.append(name)
        interaction_df = pd.DataFrame(interaction_dict, index=df_panel.index)
        
        df_factors = pd.concat([df_panel, digital_factors_aligned, entre_factors_aligned, interaction_df], axis=1)

        regressors = list(digital_factors_aligned.columns) + list(entre_factors_aligned.columns) + interactions
        for reg in regressors:
            if reg in df_factors.columns and df_factors[reg].dtype == 'O':
                df_factors[reg] = pd.to_numeric(df_factors[reg], errors='coerce')
        
        y_reg = np.log(df_factors[SECTOR] + 1e-9)
        final_data = pd.concat([y_reg, df_factors[regressors]], axis=1).dropna()

        for lag in range(1, LAG_ORDER+1):
            lag_name = f'{SECTOR}_lag{lag}'
            final_data[lag_name] = final_data.groupby('CountryShort')[y_reg.name].shift(lag)
        final_data = final_data.dropna()

        y_reg = final_data[y_reg.name]
        X_reg = final_data[regressors + [f'{SECTOR}_lag{l}' for l in range(1, LAG_ORDER+1)]]

        nunq = X_reg.apply(pd.Series.nunique)
        to_drop = nunq[nunq == 1].index
        if len(to_drop) > 0:
            print(f"Dropping constant columns: {to_drop.tolist()}")
            X_reg = X_reg.drop(columns=to_drop)
        
        clusters = final_data.index.get_level_values('CountryShort')
        clusters = pd.Series(clusters, index=final_data.index, name='CountryShort')
        print("\nRunning PanelOLS (two-way FE, clustered SE by country, block SPCA)...")
        fe_model = PanelOLS(y_reg, X_reg, entity_effects=True, time_effects=True, drop_absorbed=True)
        results = fe_model.fit(cov_type='clustered', clusters=clusters)
        print(results.summary)

        # Diagnostics
        dw_stat = durbin_watson(results.resids)
        X_reg_for_bp = sm.add_constant(X_reg, has_constant='add')
        if X_reg_for_bp.shape[1] < 2:
            bp_pval = np.nan
        else:
            try:
                bp_test = het_breuschpagan(results.resids, X_reg_for_bp)
                bp_pval = bp_test[1]
            except Exception as e:
                print("Breusch-Pagan test failed:", e)
                bp_pval = np.nan

        try:
            wool_f, wool_p = wooldridge_test(final_data.reset_index().set_index(['CountryShort','Year']), y_reg.name)
        except Exception as e:
            print("[Wooldridge] test failed:", e)
            wool_f, wool_p = np.nan, np.nan

        try:
            bg_lags = 1
            ols_model = sm.OLS(y_reg, sm.add_constant(X_reg)).fit()
            bg_test = acorr_breusch_godfrey(ols_model, nlags=bg_lags)
            bg_stat = bg_test[0]
            bg_pval = bg_test[1]
        except Exception as e:
            print("Breusch-Godfrey test failed:", e)
            bg_stat, bg_pval = np.nan, np.nan

        # Results DataFrame
        results_df = results.params.to_frame('Coefficient')
        results_df['StdErr'] = results.std_errors
        results_df['PValue'] = results.pvalues
        results_df.reset_index(inplace=True)
        results_df.rename(columns={'index': 'Indicator'}, inplace=True)

        results_df['GVA'] = SECTOR.replace('GVA_', '').replace('_Total','').replace('_', '')
        results_df['R2_within'] = results.rsquared_within
        results_df['R2_between'] = results.rsquared_between
        results_df['R2_overall'] = results.rsquared_overall

        col_order = ['GVA','Indicator','Coefficient','StdErr','PValue','R2_within','R2_between','R2_overall']
        results_df = results_df[col_order]

        all_results_list.append(results_df)

        # Save for R² plot
        r2_results.append({
            'GVA': results_df['GVA'].iloc[0],
            'R2_within': results.rsquared_within,
            'R2_between': results.rsquared_between,
            'R2_overall': results.rsquared_overall
        })

        # Diagnostics
        diagnostics = {
            'GVA': results_df['GVA'].iloc[0],
            'Durbin-Watson': dw_stat,
            'Breusch-Pagan p-value': bp_pval,
            'Wooldridge F': wool_f,
            'Wooldridge p-value': wool_p,
            'Breusch-Godfrey stat': bg_stat,
            'Breusch-Godfrey p-value': bg_pval
        }
        all_diag_list.append(diagnostics)

        diagnostics_df = pd.DataFrame({
            'Statistic': [
                'Durbin-Watson', 'Breusch-Pagan p-value', 
                'Wooldridge F', 'Wooldridge p-value', 
                'Breusch-Godfrey stat', 'Breusch-Godfrey p-value'
            ],
            'Value': [
                dw_stat, bp_pval, 
                wool_f, wool_p, 
                bg_stat, bg_pval
            ]
        })
        diagnostics_df.to_excel(writer, sheet_name=safe_sheet_name(f"{SECTOR}_Diag"), index=False)

    all_results_df = pd.concat(all_results_list, axis=0).reset_index(drop=True)
    all_results_df.to_excel(writer, sheet_name=safe_sheet_name("All_Regressions"), index=False)

    for block, loadings in digital_loadings.items():
        loadings.to_excel(writer, sheet_name=safe_sheet_name(f"DE_{block}_Loadings"))
    for block, loadings in entre_loadings.items():
        loadings.to_excel(writer, sheet_name=safe_sheet_name(f"EN_{block}_Loadings"))

    if not var_exp_df.empty:
        var_exp_df.to_excel(writer, sheet_name=safe_sheet_name("SPCA_VarExplained"), index=False)

# === PLOT: Regression Coefficients Per GVA (significant only, with stars) ===
def pval_stars(p):
    if p <= 0.01:
        return '***'
    elif p <= 0.05:
        return '**'
    elif p <= 0.1:
        return '*'
    else:
        return ''

all_results_df = pd.concat(all_results_list, axis=0).reset_index(drop=True)
gvas = all_results_df['GVA'].unique()
for gva in gvas:
    gva_df = all_results_df[all_results_df['GVA'] == gva].copy()
    signif = gva_df[gva_df['PValue'] < 0.1].copy()
    if len(signif) == 0:
        continue
    signif['stars'] = signif['PValue'].apply(pval_stars)
    signif['label'] = signif.apply(lambda row: f"{row['Coefficient']:.2f}{row['stars']}", axis=1)
    plt.figure(figsize=(10, max(4, 0.5*len(signif))))
    bars = plt.barh(signif['Indicator'], signif['Coefficient'], color='tab:blue')
    for i, (bar, lbl) in enumerate(zip(bars, signif['label'])):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, lbl, va='center', fontsize=10)
    plt.title(f"Significant Regression Coefficients ({gva})")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FOLDER, f"regression_coefficients_{gva}.png"), dpi=300)
    plt.close()

# === PLOT: Diagnostics per GVA ===
diag_df = pd.DataFrame(all_diag_list)
for stat in ['Durbin-Watson','Breusch-Pagan p-value','Wooldridge F','Wooldridge p-value','Breusch-Godfrey stat','Breusch-Godfrey p-value']:
    plt.figure(figsize=(10,4))
    sns.barplot(data=diag_df, x='GVA', y=stat)
    plt.title(f"{stat} per Sector")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FOLDER, f"diagnostic_{stat.replace(' ','_')}.png"), dpi=300)
    plt.close()

# === PLOT: R2 per GVA ===
r2_df = pd.DataFrame(r2_results)
r2_melt = r2_df.melt(id_vars='GVA', value_vars=['R2_within','R2_between','R2_overall'],
                     var_name='R2_Type', value_name='R2')
plt.figure(figsize=(10,6))
sns.barplot(data=r2_melt, x='GVA', y='R2', hue='R2_Type')
plt.title("R² (within/between/overall) per GVA Sector")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_FOLDER, "r2_per_sector.png"), dpi=300)
plt.close()

print("\n--- Όλα τα αποτελέσματα συγκεντρωτικά σε 1 tab + diagnostics per GVA + loadings/var_explained + όλα τα plots + R2/diagnostics per sector! ---")
