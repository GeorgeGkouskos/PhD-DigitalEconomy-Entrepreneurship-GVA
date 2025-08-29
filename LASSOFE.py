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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.impute import SimpleImputer
import warnings
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import chi2, norm
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
import os
import textwrap

# --- Constants ---
VIF_THRESHOLD = 5
MIN_YEARS_PER_COUNTRY = 7

# --- Helper Functions ---

def calculate_vif(X):
    """Calculates the Variance Inflation Factor (VIF) for each feature."""
    vif_data = []
    # Adding a constant for VIF calculation is crucial
    X_with_const = sm.add_constant(X, has_constant='add')
    for i in range(1, X_with_const.shape[1]): # Start from 1 to skip the constant
        vif = variance_inflation_factor(X_with_const.values, i)
        vif_data.append({'Variable': X.columns[i-1], 'VIF': vif})
    return pd.DataFrame(vif_data)


def remove_high_vif(X, vif_threshold=VIF_THRESHOLD, gva_target=None, drop_log=None):
    """Iteratively removes variables with VIF above a threshold."""
    dropped = []
    X_iter = X.copy()
    while True:
        vif_df = calculate_vif(X_iter)
        if vif_df.empty:
            break
        high_vif = vif_df[vif_df['VIF'] > vif_threshold]
        if high_vif.empty or len(X_iter.columns) <= 2:
            break
        
        worst_var = high_vif.sort_values('VIF', ascending=False).iloc[0]
        worst_feature = worst_var['Variable']
        
        if drop_log is not None and gva_target is not None:
            drop_log.append({
                'Sector': gva_target,
                'Dropped_Variable': worst_feature,
                'VIF': worst_var['VIF']
            })
        X_iter = X_iter.drop(columns=[worst_feature])
        dropped.append(worst_feature)
    return X_iter, dropped


def filter_countries_by_data_availability(df, country_col='CountryShort', year_col='Year', min_years=MIN_YEARS_PER_COUNTRY):
    """
    Filters out countries that have data for fewer than a minimum number of unique years.
    """
    country_year_counts = df.groupby(country_col)[year_col].nunique()
    countries_to_drop = country_year_counts[country_year_counts < min_years].index.tolist()
    
    exclusion_log = pd.DataFrame({
        'CountryShort': countries_to_drop,
        'Reason': [f'Fewer than {min_years} unique years of data'] * len(countries_to_drop)
    })
    
    if not countries_to_drop:
        print(f"All countries have at least {min_years} years of data. No countries dropped.")
    else:
        print(f"Identified {len(countries_to_drop)} countries to drop for having < {min_years} years: {countries_to_drop}")

    df_filtered = df[~df[country_col].isin(countries_to_drop)]
    return df_filtered, exclusion_log

def run_lasso_variable_selection(df):
    """Performs variable selection using LASSO regression for each GVA target."""
    warnings.filterwarnings('ignore', category=UserWarning)
    gva_cols = [col for col in df.columns if 'GVA_' in col]
    desi_cols = [col for col in df.columns if 'DESI_' in col]
    control_cols = [col for col in df.columns if 'CNTRL_' in col]
    selected_features_by_gva = {}
    
    print("Running LASSO for feature selection...")
    for gva_target in gva_cols:
        print(f"   ... for target: {gva_target}")
        df_iter = df.copy()
        
        # Log-transform and handle NaNs in target
        if pd.api.types.is_numeric_dtype(df_iter[gva_target]):
            df_iter[gva_target] = np.log(df_iter[gva_target].astype(float) + 1e-9)
        df_iter.dropna(subset=[gva_target], inplace=True)
        
        X = df_iter[desi_cols + control_cols]
        y = df_iter[gva_target]
        
        # Pre-imputation cleaning
        X = X.dropna(axis=1, how='all')
        
        # Imputation and Scaling
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
        
        # LASSO with Cross-Validation
        lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1, tol=1e-3)
        lasso_cv.fit(X_scaled, y)
        
        # Extract selected features
        selected_feature_names = X.columns[np.abs(lasso_cv.coef_) > 1e-6].tolist()
        selected_features_by_gva[gva_target] = selected_feature_names
        
    return selected_features_by_gva

def hausman_test(fixed, random):
    """Performs the Hausman test to choose between fixed and random effects."""
    b = fixed.params
    B = random.params
    # Ensure parameter alignment for comparison
    common_params = fixed.params.index.intersection(random.params.index)
    if len(common_params) == 0:
        return np.nan, np.nan, "No common parameters."
    
    b = b[common_params]
    B = B[common_params]
    
    v_b = fixed.cov.loc[common_params, common_params]
    v_B = random.cov.loc[common_params, common_params]
    
    diff = b - B
    try:
        # Use pseudo-inverse for stability if the matrix is singular
        inv_diff_cov = np.linalg.pinv(v_b - v_B)
        chi2_stat = diff.T @ inv_diff_cov @ diff
        pval = chi2.sf(chi2_stat, diff.shape[0])
        return chi2_stat, pval
    except np.linalg.LinAlgError as e:
        return np.nan, np.nan, f"Hausman test failed due to matrix inversion error: {e}"


def generate_and_save_diagnostic_plots(results, ols_results, X_final, gva_target, plots_dir):
    """
    Generates and saves a set of diagnostic plots for a regression model.
    """
    print(f"   ... generating plots for {gva_target}")
    os.makedirs(plots_dir, exist_ok=True)
    influence = ols_results.get_influence()

    # 1. Residuals vs. Fitted Plot
    plt.figure(figsize=(10, 6))
    sns.residplot(x=results.fitted_values.values.squeeze(), y=results.resids.values.squeeze(), lowess=True,
                  scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plt.title(f'Residuals vs Fitted for {gva_target}', fontsize=16)
    plt.xlabel('Fitted values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(plots_dir, f'{gva_target}_1_residuals_vs_fitted.png'))
    plt.close()

    # 2. Q-Q Plot
    plt.figure(figsize=(10, 6))
    sm.qqplot(results.resids, line='s', fit=True)
    plt.title(f'Q-Q Plot of Residuals for {gva_target}', fontsize=16)
    plt.savefig(os.path.join(plots_dir, f'{gva_target}_2_qq_plot.png'))
    plt.close()
    
    # 3. Scale-Location Plot
    model_norm_residuals = ols_results.get_influence().resid_studentized_internal
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    plt.figure(figsize=(10, 6))
    plt.scatter(ols_results.fittedvalues, model_norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(x=ols_results.fittedvalues, y=model_norm_residuals_abs_sqrt,
                scatter=False, ci=False, lowess=True, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plt.title(f'Scale-Location Plot for {gva_target}', fontsize=16)
    plt.xlabel('Fitted values', fontsize=12)
    plt.ylabel('Sqrt(|Standardized Residuals|)', fontsize=12)
    plt.savefig(os.path.join(plots_dir, f'{gva_target}_3_scale_location.png'))
    plt.close()

    # 4. Coefficients Plot (No Constant)
    params = results.params.drop('const', errors='ignore')
    conf = results.conf_int().drop('const', errors='ignore')
    
    if not params.empty:
        conf['Coefficient'] = params
        conf.columns = ['Lower CI', 'Upper CI', 'Coefficient']
        conf = conf.sort_values('Coefficient')
        errors = conf['Upper CI'] - conf['Coefficient']
        plt.figure(figsize=(10, max(6, len(conf) * 0.5)))
        plt.barh(conf.index, conf['Coefficient'], xerr=errors, capsize=5, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--', linewidth=1)
        plt.title(f'Coefficients and 95% CIs for {gva_target}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{gva_target}_4_coefficients.png'))
        plt.close()

    # 5. Correlation Heatmap of Predictors
    if not X_final.empty:
        plt.figure(figsize=(12, 10))
        corr_matrix = X_final.corr()
        # Annotate heatmap only if there are few enough variables to be readable
        annotate = len(X_final.columns) <= 15
        sns.heatmap(corr_matrix, annot=annotate, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
        plt.title(f'Correlation Heatmap of Predictors for {gva_target}', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{gva_target}_5_correlation_heatmap.png'))
        plt.close()

    # 6. Influence Plot (Leverage and Cook's Distance)
    fig, ax = plt.subplots(figsize=(12, 8))
    sm.graphics.influence_plot(ols_results, ax=ax, criterion="cooks")
    plt.title(f'Influence Plot for {gva_target}', fontsize=16)
    fig.tight_layout(pad=1.0)
    plt.savefig(os.path.join(plots_dir, f'{gva_target}_6_influence_plot.png'))
    plt.close()

    # 7. Cook's Distance Plot
    (cooks_d, pvals) = influence.cooks_distance
    plt.figure(figsize=(12, 6))
    plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
    plt.title(f"Cook's Distance Plot for {gva_target}", fontsize=16)
    plt.xlabel("Observation Index", fontsize=12)
    plt.ylabel("Cook's Distance", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{gva_target}_7_cooks_distance.png'))
    plt.close()


def run_fixed_effects_models(df_csv_path, selected_features_dict, vif_threshold, plots_dir, results_txt_path):
    """Runs FE models, performs diagnostics, and saves comprehensive text and CSV results."""
    warnings.filterwarnings('ignore', category=FutureWarning)
    df_panel = pd.read_csv(df_csv_path)

    # Initialize the text file for saving summaries
    with open(results_txt_path, 'w') as f:
        f.write(f"Panel Regression and Diagnostics Summary\nGenerated on: {pd.Timestamp.now()}\n")

    print("\nRunning Fixed-Effects Panel Data Models...")
    all_results, drop_log = [], []

    for gva_target, features in selected_features_dict.items():
        print(f"\n--- Processing sector: {gva_target} ---")
        if not features:
            print(f"   Skipping {gva_target} as LASSO selected no features.")
            continue
            
        df_iter = df_panel.copy()
        
        # Prepare data for the model
        df_iter[gva_target] = np.log(df_iter[gva_target].astype(float) + 1e-9)
        model_vars = [gva_target] + features
        df_subset = df_iter[['CountryShort', 'Year'] + model_vars].copy()
        df_subset.dropna(subset=[gva_target], inplace=True)

        # ==============================================================================
        # === MODIFICATION: Add Lagged Dependent Variable to Model Autocorrelation ===
        # ==============================================================================
        # This directly addresses the serial correlation indicated by a low Durbin-Watson value.
        # By including the previous period's value, we explicitly model the time dependency.
        # Note: This can introduce "Nickell bias" in panels with a small time dimension (T).
        print(f"   ... adding lagged dependent variable to address serial correlation.")
        lagged_dv_col = f'L1_{gva_target}'
        df_subset[lagged_dv_col] = df_subset.groupby('CountryShort')[gva_target].shift(1)
        
        # Add the new lagged variable to our list of predictors
        features_with_lag = features + [lagged_dv_col]
        
        # Drop rows with NaNs created by the lagging operation (the first year for each country)
        df_subset.dropna(inplace=True)
        
        # Set index after all data manipulations are complete
        df_subset = df_subset.set_index(['CountryShort', 'Year'])
        # ==============================================================================

        if df_subset.empty:
            print(f"   Skipping {gva_target} due to no data after handling NaNs and lags.")
            continue

        y = df_subset[gva_target]
        # Use the feature list that now includes the lag
        X = df_subset[features_with_lag]
        
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), index=X.index, columns=X.columns)
        
        # VIF check on LASSO-selected variables (including the new lag)
        X_final, dropped = remove_high_vif(X_imputed, vif_threshold, gva_target, drop_log)
        if dropped:
            print(f"   ... removed for high VIF: {', '.join(dropped)}")
        
        y, X_final = y.align(X_final, join='inner', axis=0)
        
        if len(y) < len(X_final.columns) + 2: # Not enough data to fit model
            print(f"   Skipping {gva_target} due to insufficient data after VIF removal and alignment ({len(y)} obs).")
            continue

        print(f"   Final sample size: {len(y)} obs across {y.index.get_level_values('CountryShort').nunique()} countries.")

        try:
            # Add constant for OLS, RE, and FE models
            X_with_const = sm.add_constant(X_final)

            # Fit all required models
            fe_model = PanelOLS(y, X_with_const, entity_effects=True, time_effects=True)
            # Use robust standard errors to account for heteroskedasticity and any remaining autocorrelation
            robust_results = fe_model.fit(cov_type='robust')
            standard_results = fe_model.fit() # For Hausman and plots
            
            # Need to use original y and X_with_const that are aligned
            ols_model = sm.OLS(y, X_with_const).fit()
            re_model = RandomEffects(y, X_with_const).fit()

        except Exception as e:
            print(f"   Could not fit model for {gva_target}: {e}")
            continue

        # --- Generate and Save Text Summary ---
        with open(results_txt_path, 'a') as f:
            f.write(f"\n\n{'='*80}\n")
            f.write(f"Regression Results for: {gva_target}\n")
            f.write(f"(Note: Includes a 1-period lag of the dependent variable to control for autocorrelation)\n")
            f.write(f"{'='*80}\n")
            f.write(str(robust_results))
            
            # --- Diagnostic Section ---
            dw_value = durbin_watson(robust_results.resids.values)
            bp_test = het_breuschpagan(ols_model.resid, ols_model.model.exog)
            chi2_stat, haus_pvalue = hausman_test(standard_results, re_model)
            final_vif_df = calculate_vif(X_final)
            
            f.write(f"\n\n{'-'*30} Model Diagnostics {'-'*30}\n")
            f.write(f"\n1. Durbin-Watson Statistic: {dw_value:.4f}")
            f.write("\n   - Interpretation: Values around 2.0 suggest no first-order autocorrelation. "
                    "Values < 1.5 suggest positive, > 2.5 suggest negative autocorrelation.\n"
                    # === MODIFICATION HERE ===
                    "   - Note: Model includes a 1-period lag of the dependent variable to control for autocorrelation.\n")
            
            f.write(f"\n2. Breusch-Pagan Test (p-value): {bp_test[1]:.4f}")
            f.write("\n   - Interpretation: Tests for heteroskedasticity. A p-value < 0.05 suggests its presence, "
                    "justifying the use of robust standard errors.\n")
            
            f.write(f"\n3. Hausman Test (p-value): {haus_pvalue:.4f}")
            f.write("\n   - Interpretation: Compares Fixed vs. Random Effects. A p-value < 0.05 suggests "
                    "that fixed effects are the preferred model.\n")

            # --- MODIFIED VIF REPORTING ---
            min_vif = final_vif_df['VIF'].min() if not final_vif_df.empty else 'N/A'
            max_vif = final_vif_df['VIF'].max() if not final_vif_df.empty else 'N/A'
            f.write("\n4. Final Variable VIFs:")
            f.write(f"\n   - Min VIF: {min_vif:.4f}" if isinstance(min_vif, (int, float)) else f"\n   - Min VIF: {min_vif}")
            f.write(f"\n   - Max VIF: {max_vif:.4f}" if isinstance(max_vif, (int, float)) else f"\n   - Max VIF: {max_vif}")
            f.write(f"\n{'='*80}\n")

        print(f"   ... full results and diagnostics for {gva_target} saved to text file.")
        
        # Pass X_final to the plotting function
        generate_and_save_diagnostic_plots(standard_results, ols_model, X_final, gva_target, plots_dir)

        # --- MODIFIED: Gather Numerical Results for Excel ---
        params = robust_results.params.drop('const', errors='ignore')
        robust_std_errors = robust_results.std_errors.drop('const', errors='ignore')
        pvalues = robust_results.pvalues.drop('const', errors='ignore')
        conf_int = robust_results.conf_int().drop('const', errors='ignore')
        
        for var in params.index:
            all_results.append({
                'Sector': gva_target, 'Variable': var, 'Coefficient': params[var],
                'Robust_Std_Error': robust_std_errors[var], 'P_Value': pvalues[var],
                'CI_Lower': conf_int.loc[var, 'lower'], 'CI_Upper': conf_int.loc[var, 'upper'],
                'R-squared': robust_results.rsquared,
                'R-squared (Between)': robust_results.rsquared_between,
                'R-squared (Within)': robust_results.rsquared_within,
                'R-squared (Overall)': robust_results.rsquared_overall,
                'VIF': final_vif_df.set_index('Variable').loc[var, 'VIF'],
                'DurbinWatson': dw_value, 'BreuschPagan_p': bp_test[1], 'Hausman_p': haus_pvalue
            })

    if not all_results:
        return None, None
        
    full_results_df = pd.DataFrame(all_results)
    drop_log_df = pd.DataFrame(drop_log) if drop_log else pd.DataFrame(columns=['Sector','Dropped_Variable','VIF'])
    return full_results_df, drop_log_df


# --- Main Execution Block ---
if __name__ == '__main__':
    # Define file paths - PLEASE UPDATE THIS PATH
    base_path = "data/processed"
    file_path = "data/raw/DatasetMasterCAP.xlsx"
    csv_path = os.path.join(base_path, "DatasetMasterCAP_filtered.csv")
    results_excel_path = os.path.join(base_path, "DESILASSOPERCAP.xlsx")
    results_txt_path = os.path.join(base_path, "DESILASSOPERCAP_summary.txt") # For detailed text output
    scats_dir = "figures/coefficients"

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs(scats_dir, exist_ok=True)
    print(f"Diagnostic plots will be saved in '{os.path.abspath(scats_dir)}' directory.")
    
    # Step 1: Load and Filter Data
    print("\n--- Step 1: Loading and Filtering Data ---")
    df = pd.read_excel(file_path)
    df_filtered, exclusion_log = filter_countries_by_data_availability(df)
    
    # Step 2: LASSO Variable Selection
    print("\n--- Step 2: Performing LASSO Variable Selection ---")
    selected_features = run_lasso_variable_selection(df_filtered)
    
    if any(selected_features.values()):
        print("\n--- Summary of Selected Features (from LASSO) ---")
        for gva_var, features in selected_features.items():
            print(f"\nFor GVA Target: {gva_var}")
            print(f"   Selected {len(features)} features: {', '.join(features) if features else 'None'}")
        
        df_filtered.to_csv(csv_path, index=False)
        print(f"\nSaved filtered data for panel regression to {csv_path}")
        
        # Step 3: Run Panel Models & Generate Outputs
        print("\n--- Step 3: Running Panel Models and Generating Outputs ---")
        full_results_df, drop_log_df = run_fixed_effects_models(
            csv_path, selected_features, 
            vif_threshold=VIF_THRESHOLD, 
            plots_dir=scats_dir,
            results_txt_path=results_txt_path
        )
        
        # Step 4: Save Numerical Results
        print("\n--- Step 4: Saving Results ---")
        if full_results_df is not None:
            with pd.ExcelWriter(results_excel_path) as writer:
                full_results_df.to_excel(writer, sheet_name="Regression_Results", index=False)
                drop_log_df.to_excel(writer, sheet_name="Dropped_Variables_VIF", index=False)
                if not exclusion_log.empty:
                    exclusion_log.to_excel(writer, sheet_name="Excluded_Countries_Initial", index=False)
            print(f"Excel results summary saved to '{results_excel_path}'")
            print(f"Detailed text summary with diagnostics saved to '{results_txt_path}'")
            
            # --- MODIFIED: Generate and save summary R-squared plots ---
            
            # 1. Within R-squared Plot
            r2_within_data = full_results_df[['Sector', 'R-squared (Within)']].drop_duplicates().set_index('Sector')
            if not r2_within_data.empty:
                r2_within_data.plot(kind='bar', figsize=(14, 8), grid=True, legend=False)
                plt.title('Comparison of Within R-squared Values by Sector', fontsize=16)
                plt.ylabel('R-squared (Within)', fontsize=12)
                plt.xlabel('Sector', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                r2_plot_path = os.path.join(scats_dir, '_summary_r_squared_within_comparison.png')
                plt.savefig(r2_plot_path)
                plt.close()
                print(f"Summary Within R-squared plot saved to '{r2_plot_path}'")

            # 2. Between R-squared Plot
            r2_between_data = full_results_df[['Sector', 'R-squared (Between)']].drop_duplicates().set_index('Sector')
            if not r2_between_data.empty:
                r2_between_data.plot(kind='bar', figsize=(14, 8), grid=True, legend=False, color='coral')
                plt.title('Comparison of Between R-squared Values by Sector', fontsize=16)
                plt.ylabel('R-squared (Between)', fontsize=12)
                plt.xlabel('Sector', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                r2_plot_path = os.path.join(scats_dir, '_summary_r_squared_between_comparison.png')
                plt.savefig(r2_plot_path)
                plt.close()
                print(f"Summary Between R-squared plot saved to '{r2_plot_path}'")
            
            # 3. Overall R-squared Plot
            r2_overall_data = full_results_df[['Sector', 'R-squared (Overall)']].drop_duplicates().set_index('Sector')
            if not r2_overall_data.empty:
                r2_overall_data.plot(kind='bar', figsize=(14, 8), grid=True, legend=False, color='lightgreen')
                plt.title('Comparison of Overall R-squared Values by Sector', fontsize=16)
                plt.ylabel('R-squared (Overall)', fontsize=12)
                plt.xlabel('Sector', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                r2_plot_path = os.path.join(scats_dir, '_summary_r_squared_overall_comparison.png')
                plt.savefig(r2_plot_path)
                plt.close()
                print(f"Summary Overall R-squared plot saved to '{r2_plot_path}'")

        else:
            print("Process finished, but no numerical results were generated.")
            
    else:
        print("LASSO selection returned no features. Halting process.")
        
    print("\nProcess completed.")