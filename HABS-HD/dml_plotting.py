"""
DoubleML Analysis and Visualization for HABS-HD Alzheimer's Study

This module implements causal inference analysis using Double Machine Learning (DML) 
to estimate the effect of plasma pTau-217 biomarker levels on Alzheimer's diagnosis,
controlling for confounding variables including age, sex, education, and APOE4 genotype.

The analysis uses data from the Harvard Aging Brain Study - Huntington's Disease (HABS-HD)
cohort and employs Causal Forest models to estimate heterogeneous treatment effects.

Author: [Author Name]
Date: July 2025
Dependencies: pandas, numpy, sklearn, econml, doubleml, pygam, matplotlib, shap, flaml
"""

# Standard data manipulation and analysis libraries
import pandas as pd
import numpy as np
from tqdm import tqdm

# Machine learning and causal inference libraries
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestRegressor, 
                            RandomForestClassifier, GradientBoostingRegressor, 
                            GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Causal inference specific libraries
import doubleml as dml
from doubleml import DoubleMLData, DoubleMLPLR
from econml.dr import DRLearner
from econml.dml import CausalForestDML

# Visualization and interpretation libraries
import matplotlib.pyplot as plt
from pygam import LinearGAM, s
import shap
from flaml import AutoML
import pickle

# Numerical computing libraries
import scipy.sparse

# Compatibility patches for deprecated NumPy types
np.int = int  # Patch for deprecated np.int to maintain backward compatibility

# Sparse matrix compatibility patches to ensure CSR/CSC matrices behave like dense arrays
scipy.sparse.csr.csr_matrix.A = property(lambda self: self.toarray())
scipy.sparse.csc.csc_matrix.A = property(lambda self: self.toarray())


# Load HABS-HD clinical, genomics, and biomarker datasets
# These Excel files contain the core data for the Alzheimer's study
clinical_df = pd.read_excel('../../raw_data/HABS/HABS-HD/HABS-HD_Clinical_Data/RP_HD_7_Clinical.xlsx')
genomics_df = pd.read_excel('../../raw_data/HABS/HABS-HD/HABS-HD_Clinical_Data/RP_HD_7_Genomics.xlsx')
biomarker_df = pd.read_excel('../../raw_data/HABS/HABS-HD/HABS-HD_Clinical_Data/RP_HD_7_Biomarkers.xlsx')

def get_data():
    """
    Prepare and clean HABS-HD data for Double Machine Learning analysis.
    
    This function performs comprehensive data preprocessing including:
    - Filtering valid biomarker measurements (removing missing/invalid codes)
    - Matching clinical data to biomarker timepoints
    - Excluding patients with confounding neurological conditions from controls
    - Creating dummy variables for categorical features
    - Handling missing data and low-variance features
    
    Returns:
        pd.DataFrame: Cleaned dataset ready for causal inference analysis containing:
            - Demographics: Age, Gender, Education, Ethnicity
            - Genetics: APOE4 genotype (dummy encoded)
            - Biomarker: Plasma pTau-217 levels
            - Outcome: Alzheimer's diagnosis (binary)
            - Clinical covariates: GDS category and other relevant measures
    
    Notes:
        - Invalid biomarker codes (-999999, -888888, -777777) are excluded
        - Patients with other neurodegenerative diseases are removed from controls
        - Clinical data is matched to the closest visit prior to biomarker collection
        - Features with >20% missingness or single unique values are dropped
    """
    # Define the biomarker of interest: plasma pTau-217
    # This is our treatment variable in the causal analysis
    bm_col = 'r7_QTX_Plasma_pTau217_PLUS'
    
    # Filter biomarker data to exclude invalid/missing measurements
    # Standard HABS-HD codes for missing data: -999999, -888888, -777777
    bm = biomarker_df[biomarker_df[bm_col].isin([-999999, -888888, -777777]) == False]

    # Get clinical data for participants with valid biomarker measurements
    bm_clinical = clinical_df[clinical_df.Med_ID.isin(bm.Med_ID.unique())]

    # Define exclusion criteria for control subjects
    # Remove participants with other neurodegenerative diseases to create clean controls
    control_remove_cols = ['IMH_Alzheimers', 'IMH_Parkinsons', 'IMH_VaD',  
                          'IMH_FTD', 'IMH_Dementia_due_Parkinsons', 'IMH_Dementia_Other_1']
    pts_acd = bm_clinical.loc[(bm_clinical[control_remove_cols] == 1).any(axis=1), 'Med_ID'].unique()
    print(f"Patients with diseases for control exclusion: {pts_acd.shape}")

    # Match clinical data to biomarker timepoints
    # For each biomarker measurement, find the closest preceding clinical assessment
    new_clinical_rows = []
    for index, row in bm.iterrows():
        med_id = row['Med_ID']
        max_age = row['Age']
        # Find the most recent clinical visit at or before the biomarker collection
        clinical_row = bm_clinical[(bm_clinical.Med_ID == med_id) & 
                                  (bm_clinical.Age <= max_age)].sort_values(by='Age', ascending=False).head(1)
        clinical_row.Age = max_age  # Align age to biomarker collection timepoint
        if not clinical_row.empty:
            new_clinical_rows.append(clinical_row.iloc[0])

    # Create matched clinical dataset
    new_clinical_df = pd.DataFrame(new_clinical_rows)
    print(f"new_clinical_df.shape, bm.shape: {new_clinical_df.shape}, {bm.shape}")

    # Select core biomarker and demographic variables
    # Keep essential variables for the causal analysis
    bm = bm.loc[:, ['Med_ID', 'Ethnicity', 'Visit_ID', 'Age', 'ID_Education', 'ID_Gender', bm_col]]
    
    # Merge with genetic data (APOE4 genotype - major Alzheimer's risk factor)
    bm = bm.merge(genomics_df[['Med_ID', 'APOE4_Genotype']], on='Med_ID', how='left')

    # Create dummy variables for APOE4 genotype
    # Note: Could group E2 carriers, but keeping all genotypes separate for now
    # bm.loc[bm['APOE4_Genotype'].isin(['E2E2', 'E2E3', 'E2E4']), 'APOE4_Genotype'] = 'E2_carrier'
    # bm = pd.get_dummies(bm, columns=['APOE4_Genotype'], prefix='APOE4', drop_first=True)
    bm = pd.concat([bm, pd.get_dummies(bm.APOE4_Genotype, prefix='APOE4')], axis=1)

    print(bm.shape)

    # Merge with matched clinical data
    # Exclude duplicate columns that exist in both datasets
    merge_cols = [col for col in new_clinical_df.columns 
                 if col not in ['Visit_ID', 'ID_Gender', 'ID_Education', 'Ethnicity']]
    bm = pd.merge(bm, new_clinical_df[merge_cols], on=['Med_ID', 'Age'], how='left')

    # Remove problematic columns with inconsistent data
    bm = bm.drop(columns=['IMH_Dementia_Other_1Type', 'IMH_AlzheimersAge'])
    
    # Create dummy variables for Global Deterioration Scale (GDS) categories
    # GDS is a measure of cognitive decline severity
    bm = pd.get_dummies(bm, columns=['GDS_Category'], prefix='GDS')

    # Handle missing data: replace HABS-HD missing code with standard NaN
    bm[bm == -9999] = np.nan

    # Data quality checks and feature selection
    # Remove columns with insufficient variation (constant or near-constant)
    drops = []
    for col in bm.columns:
        if col in ['Med_ID', 'Visit_ID']:
            continue
        if bm[col].nunique() <= 1:
            # Skip features with only one unique value
            drops.append(col)
            
    # Quality check: ensure outcome variable has variation
    # if 'IMH_Alzheimers' in drops:
    #     print(f'Alzheimers diagnosis column only has one unique value, skipping biomarker.')
    #     continue 

    bm.drop(columns=drops, inplace=True)

    # Remove features with excessive missing data
    # High missingness can bias causal effect estimates
    missingness_threshold = 0.2  # 20% threshold
    missingness = bm.isnull().mean()
    high_missing_cols = missingness[missingness > missingness_threshold].index.tolist()
    if high_missing_cols:
        print(f'Removing columns with high missingness: {high_missing_cols}')
        bm.drop(columns=high_missing_cols, inplace=True)

    # Create final case-control dataset
    # Filter to participants with clear Alzheimer's diagnosis status (0 or 1)
    bm = bm[bm.IMH_Alzheimers.isin([0, 1])]
    case_ids = bm[bm.IMH_Alzheimers == 1].Med_ID.unique()
    control_ids = bm[bm.IMH_Alzheimers == 0].Med_ID.unique()
    print(f"Case IDs: {case_ids.shape}, Control IDs: {control_ids.shape}")

    # Apply exclusion criteria to create clean control group
    # Remove participants with other neurodegenerative diseases from controls
    case_df = bm[bm.Med_ID.isin(case_ids)]
    control_df = bm[bm.Med_ID.isin(control_ids) & ~bm.Med_ID.isin(pts_acd)]
    print(f"Case shape: {case_df.shape}, Control shape: {control_df.shape}")

    # Quality check for sufficient sample size
    # if case_df.shape[0] <= 10:
    #     print(f"Skipping biomarker {bm_col} due to insufficient cases.")
    #     continue

    # Combine cases and controls for final analysis dataset
    bm = pd.concat([case_df, control_df], ignore_index=True)

    return bm


def plot_cate_histogram(X, cf):
    """
    Plot histogram of estimated Conditional Average Treatment Effects (CATE).
    
    This function visualizes the distribution of individualized treatment effects
    estimated by the Causal Forest model. The CATE represents how much the treatment
    (pTau-217 levels) affects the outcome (Alzheimer's diagnosis) for each individual.
    
    Parameters:
        X (pd.DataFrame): Feature matrix containing covariates for all individuals.
        cf (CausalForestDML): Fitted Causal Forest model from econml.
    
    Returns:
        None: Displays matplotlib figure and prints summary statistics.
    
    Notes:
        - Positive CATE values indicate higher pTau-217 increases Alzheimer's risk
        - Negative CATE values indicate protective or null effects
        - Heterogeneity in CATE distribution suggests subgroup differences
        
    Example:
        >>> plot_cate_histogram(X_test, fitted_causal_forest)
    """

    # Estimate individualized treatment effects for all participants
    # T0=0, T1=1 estimates the effect of a 1-unit increase in pTau-217
    tau_hat = cf.effect(X, T0=0, T1=1)

    # Create publication-ready histogram with professional styling
    plt.figure(figsize=(8, 6))
    plt.hist(tau_hat, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    plt.title("Distribution of Estimated Conditional Average Treatment Effects (CATE)\nfor pTau-217 on Alzheimer's Diagnosis", 
            fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Estimated Treatment Effect (∂Y/∂D)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add reference line for mean effect
    plt.axvline(x=np.mean(tau_hat), color='red', linestyle='--', linewidth=2, 
            label=f'Mean CATE = {np.mean(tau_hat):.4f}')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    # Display comprehensive summary statistics
    print(f"CATE Summary Statistics:")
    print(f"Mean: {np.mean(tau_hat):.4f}")
    print(f"Median: {np.median(tau_hat):.4f}")
    print(f"Std Dev: {np.std(tau_hat):.4f}")
    print(f"Min: {np.min(tau_hat):.4f}")
    print(f"Max: {np.max(tau_hat):.4f}")

def plot_cate_vs_X(X, cf, variable_name, unit_increase=1, stratify_by_sex=False, sex_column=None):
    """
    Plot smoothed marginal treatment effect vs a feature variable.
    
    This function creates a scatter plot with GAM smoothing to visualize how
    treatment effects vary across levels of a specific covariate. This helps
    identify subgroups with differential treatment responses.
    
    Parameters:
        X (pd.DataFrame): Feature matrix containing all covariates.
        cf (CausalForestDML): Fitted Causal Forest model.
        variable_name (str): Name of the feature variable to plot on x-axis.
        unit_increase (float, optional): Treatment dose for effect estimation. Default is 1.
        stratify_by_sex (bool, optional): Whether to create separate curves by sex. Default False.
        sex_column (str, optional): Column name for sex variable if stratifying. Required if stratify_by_sex=True.
    
    Returns:
        None: Displays matplotlib figure.
    
    Raises:
        ValueError: If variable_name not found in DataFrame columns.
    
    Notes:
        - GAM (Generalized Additive Model) provides smooth non-linear fits
        - Confidence intervals show uncertainty in the smoothed estimates
        - Sex stratification can reveal important gender differences in treatment response
        
    Example:
        >>> plot_cate_vs_X(X, cf, 'Age', stratify_by_sex=True, sex_column='ID_Gender')
    """

    # Validate that the specified variable exists in the dataset
    try:
        column_index = X.columns.get_loc(variable_name)
    except KeyError:
        raise ValueError(f"Variable '{variable_name}' not found in DataFrame columns.")
    
    # Estimate individualized treatment effects
    tau_hat = cf.effect(X, T0=0, T1=unit_increase)
    tau_arr = tau_hat

    # Prepare data for GAM fitting
    # Convert to numpy arrays with explicit float type for numerical stability
    arr = X.iloc[:, column_index].to_numpy(dtype=float).reshape(-1, 1)
    
    # Initialize the plot with professional styling
    plt.figure(figsize=(8, 6))
    plt.grid(True, alpha=0.3, linestyle='--')
    
    if stratify_by_sex:
        # Create separate smoothed curves for each sex
        if sex_column is None:
            raise ValueError("sex_column must be specified when stratify_by_sex=True")
        
        for label, group_df in X.groupby(sex_column):
            # Map numeric codes to descriptive labels
            group_name = "Female" if label == 1 else "Male"

            # Extract data for this sex group
            plot_var_arr = group_df[variable_name].to_numpy(dtype=float).reshape(-1, 1)
            sex_tau_arr = tau_arr[group_df.index].reshape(-1, 1)
            
            # Fit GAM with smoothing spline for this group
            gam = LinearGAM(s(0)).fit(plot_var_arr, sex_tau_arr)

            # Create prediction grid for smooth curves
            age_grid = np.linspace(X[variable_name].min(), X[variable_name].max(), 100).reshape(-1, 1)
            pred = gam.predict(age_grid)
            conf = gam.prediction_intervals(age_grid, width=0.95)

            # Plot data points and smooth fit for this group
            color = 'orange' if label == 1 else 'blue'
            plt.scatter(plot_var_arr, sex_tau_arr, alpha=0.1, s=10, color=color)
            plt.plot(age_grid, pred, label=f"{group_name}", lw=2, color=color)
            plt.fill_between(age_grid.ravel(), conf[:, 0], conf[:, 1], alpha=0.12, color=color)
    else:
        # Single smoothed curve for all participants
        # Fit GAM with automatic smoothing parameter selection
        gam = LinearGAM(s(0)).fit(arr, tau_arr)

        # Create fine-grained prediction grid
        arr_grid = np.linspace(arr.min(), arr.max(), 100).reshape(-1, 1)
        pred = gam.predict(arr_grid)
        conf = gam.prediction_intervals(arr_grid, width=0.95)

        # Plot raw data points and smoothed fit
        plt.scatter(arr, tau_arr, alpha=0.2, s=10, color='steelblue')
        plt.plot(arr_grid, pred, color='red', lw=2, label='Smoothed GAM fit')
        plt.fill_between(arr_grid.ravel(), conf[:, 0], conf[:, 1], color='red', alpha=0.1)

    # Finalize plot formatting
    plt.xlabel(f"{variable_name}", fontsize=12)
    plt.ylabel("Estimated dY/dD", fontsize=12)
    plt.title(f"Smoothed Marginal Treatment Effect vs {variable_name}", fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    
def print_top10_cate(X, cf, age_index=None):
    """
    Display the top 10 individuals with highest estimated treatment effects.
    
    This function identifies and displays participants who are predicted to have
    the strongest response to the treatment (pTau-217), along with their confidence
    intervals. This is useful for identifying high-risk subgroups.
    
    Parameters:
        X (pd.DataFrame): Feature matrix containing covariates.
        cf (CausalForestDML): Fitted Causal Forest model.
        age_index (int): Column index of age variable in X for display purposes.
                        Required to show age alongside CATE estimates.
    
    Returns:
        None: Prints formatted results to console.
    
    Raises:
        ValueError: If age_index is not provided.
        
    Notes:
        - Higher CATE values indicate stronger treatment effects
        - Confidence intervals that exclude zero suggest significant effects
        - Age is displayed as a key covariate for clinical interpretation
        
    Example:
        >>> print_top10_cate(X, cf, age_index=0)  # Age is first column
    """
    if age_index is None:
        raise ValueError("age_index must be provided to identify the age column in X.")
    
    # Get point estimates and confidence intervals for treatment effects
    tau_hat = cf.effect(X).ravel()
    tau_lower, tau_upper = cf.effect_interval(X, alpha=0.05)  # 95% confidence intervals
    tau_lower = tau_lower.ravel()
    tau_upper = tau_upper.ravel()

    # Identify individuals with highest estimated treatment effects
    top_k = 10
    top_idx = np.argsort(-tau_hat)[:top_k]  # Sort in descending order

    # Display results in formatted table
    print("Top 10 individuals with highest estimated treatment effect:")
    print("-" * 80)
    for i in top_idx:
        print(f"Index {i} | Age: {X.values[i, age_index]:.1f} | "
              f"CATE: {tau_hat[i]:.3f} | "
              f"95% CI: ({tau_lower[i]:.3f}, {tau_upper[i]:.3f})")
        
def meta_model(X, tau_hat):
    """
    Fit a meta-model to predict individualized treatment effects from covariates.
    
    This secondary analysis fits a machine learning model to understand which
    participant characteristics predict differential treatment responses. This
    can reveal important moderating factors and guide clinical decision-making.
    
    Parameters:
        X (pd.DataFrame): Feature matrix containing all covariates.
        tau_hat (np.ndarray): Estimated treatment effects from Causal Forest.
                             Must be 1-dimensional array.
    
    Returns:
        sklearn.ensemble.GradientBoostingRegressor: Fitted meta-model that can
                                                   predict CATE from covariates.
    
    Raises:
        ValueError: If tau_hat is not a 1D array.
        
    Notes:
        - Higher R² indicates covariates strongly predict treatment heterogeneity
        - Feature importance scores reveal which factors moderate treatment effects
        - Meta-model can be used for treatment allocation decisions
        
    Example:
        >>> tau_estimates = cf.effect(X)
        >>> meta_model_fitted = meta_model(X, tau_estimates)
    """
    # Validate input format
    if len(tau_hat.shape) > 1 and tau_hat.shape[1] > 1:
        raise ValueError("tau_hat should be a 1D array of estimated CATEs.")

    # Import required modules for meta-modeling
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score
    
    # Fit gradient boosting model to predict CATE from covariates
    # This reveals which participant characteristics predict treatment response
    meta_model_fitted = GradientBoostingRegressor(n_estimators=100, random_state=42)
    meta_model_fitted.fit(X, tau_hat)
    
    # Evaluate meta-model performance
    predictions = meta_model_fitted.predict(X)
    r2 = r2_score(tau_hat, predictions)
    print(f"Meta-model R²: {r2:.3f}")
    print("(Higher R² indicates covariates strongly predict treatment heterogeneity)")
    print()

    # Display feature importance for clinical interpretation
    print("Feature Importance (predictors of treatment effect heterogeneity):")
    print("-" * 60)
    importances = meta_model_fitted.feature_importances_
    
    # Sort features by importance for easier interpretation
    feature_importance_pairs = [(X.columns[i], imp) for i, imp in enumerate(importances)]
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for feature_name, importance in feature_importance_pairs:
        print(f"{feature_name}: {importance:.3f}")

    return meta_model_fitted


def plot_dose_response(cf, X, D):
    """
    Plot the estimated dose-response curve for the treatment effect.
    
    This function visualizes how the treatment effect changes across different
    levels of the treatment variable (pTau-217). This helps understand whether
    the relationship between biomarker levels and Alzheimer's risk is linear
    or exhibits threshold effects.
    
    Parameters:
        cf (CausalForestDML): Fitted Causal Forest model.
        X (pd.DataFrame): Feature matrix containing covariates.
        D (pd.Series): Treatment values (pTau-217 levels).
                      Must be 1-dimensional.
    
    Returns:
        None: Displays matplotlib figure.
    
    Raises:
        ValueError: If D is not a 1D array.
        
    Notes:
        - Dose-response curves show average treatment effects at each dose level
        - Non-linear curves suggest threshold effects or saturation
        - Effects are relative to D=0 (baseline pTau-217 level)
        - Useful for determining optimal biomarker cutpoints
        
    Example:
        >>> plot_dose_response(cf, X, ptau_levels)
    """
    
    # Validate treatment variable format
    if len(D.shape) > 1 and D.shape[1] > 1:
        raise ValueError("D should be a 1D array of treatment values.")
    
    # Create grid of treatment values covering the observed range
    # Focus on 5th-99th percentiles to avoid extreme outliers
    d_vals = np.linspace(np.percentile(D, 5), np.percentile(D, 99), 50)

    # Calculate average treatment effect at each dose level
    # This shows the population-average effect at different biomarker levels
    print("Computing dose-response curve...")
    mu_d = []
    for d in d_vals:
        # Effect relative to zero treatment level
        effects = cf.effect(X, T0=0, T1=d)
        mu_d.append(effects.mean())

    # Create publication-ready dose-response plot
    plt.figure(figsize=(8, 6))
    plt.plot(d_vals, mu_d, lw=3, color='steelblue', marker='o', markersize=4)
    plt.xlabel("Treatment level D (pTau-217)", fontsize=12)
    plt.ylabel("Estimated E[Y(d)] - E[Y(0)]", fontsize=12)
    plt.title("Estimated Dose-Response Curve\n(Average Treatment Effect Relative to D=0)", 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add reference line at zero effect
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No effect')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print interpretation guidance
    print("\nDose-Response Interpretation:")
    print(f"- Treatment range: {d_vals.min():.2f} to {d_vals.max():.2f}")
    print(f"- Maximum effect: {max(mu_d):.4f} at dose {d_vals[np.argmax(mu_d)]:.2f}")
    print(f"- Minimum effect: {min(mu_d):.4f} at dose {d_vals[np.argmin(mu_d)]:.2f}")

def main():
    """
    Main analysis pipeline for DoubleML causal inference.
    
    This function orchestrates the complete causal analysis workflow:
    1. Load and preprocess HABS-HD data
    2. Set up variables for causal analysis (X, D, Y)
    3. Fit Causal Forest model
    4. Generate treatment effect estimates
    
    The analysis estimates the causal effect of plasma pTau-217 levels
    on Alzheimer's diagnosis, controlling for demographic and genetic confounders.
    
    Returns:
        None: Saves results to the dataframe and demonstrates analysis pipeline.
        
    Notes:
        - Uses gradient boosting for both outcome and treatment models
        - Cross-validation (CV=5) provides robust effect estimates
        - Treatment effects are stored in 'cate' column for further analysis
        
    Example:
        >>> main()  # Runs complete analysis pipeline
    """
    # Define analysis variables
    bm_col = 'r7_QTX_Plasma_pTau217_PLUS'  # Treatment: pTau-217 biomarker
    oc = 'IMH_Alzheimers'                   # Outcome: Alzheimer's diagnosis
    
    # Prepare analysis dataset with all required variables
    dml_df = bm.loc[:, ['Age', 'ID_Education', 'ID_Gender', bm_col, oc] + 
                    [col for col in bm if 'APOE4' in col] + 
                    [col for col in bm if 'Ethnicity' in col]]
    
    # Store ethnicity for potential future use, then create dummy variables
    ethnicity = dml_df.Ethnicity
    dml_df = pd.get_dummies(dml_df, columns=['Ethnicity'], prefix='Ethnicity')

    # Define covariate matrix (X): confounders to control for
    # Include demographics, genetics, and ethnicity as confounders
    X = dml_df.loc[:, ['Age', 'ID_Gender', 'ID_Education'] + 
                   [col for col in dml_df if 'APOE4_E' in col] + 
                   [col for col in dml_df if 'Ethnicity' in col]]
    
    # Define treatment (D) and outcome (Y) variables
    D = dml_df[bm_col]  # Treatment: pTau-217 levels (continuous)
    Y = dml_df[oc]      # Outcome: Alzheimer's diagnosis (binary)

    # Initialize Causal Forest model with appropriate configurations
    print("Fitting Causal Forest model...")
    cf = CausalForestDML(
        model_y=GradientBoostingClassifier(random_state=42),  # Outcome model (classification)
        model_t=GradientBoostingRegressor(random_state=42),   # Treatment model (regression)
        discrete_treatment=False,  # pTau-217 is continuous
        discrete_outcome=True,     # Alzheimer's diagnosis is binary
        cv=5,                      # 5-fold cross-validation for robust estimates
        n_estimators=100,          # Number of trees in the forest
        min_samples_leaf=10,       # Minimum samples per leaf (prevents overfitting)
        max_depth=10,              # Maximum tree depth
        random_state=123           # For reproducible results
    )

    # Fit the model and estimate individualized treatment effects
    cf.fit(Y, D, X=X)
    print("Model fitting complete.")
    
    # Store treatment effect estimates in the dataset
    dml_df['cate'] = cf.effect(X)
    print(f"Added CATE estimates to dataset. Mean CATE: {dml_df['cate'].mean():.4f}")
    
    return cf, X, D, Y, dml_df


# Example usage and analysis pipeline
if __name__ == "__main__":
    """
    Example workflow for running the complete DoubleML analysis.
    
    This demonstrates how to use the functions in this module to:
    1. Load and preprocess data
    2. Fit causal models
    3. Generate visualizations
    4. Interpret results
    """
    print("Starting DoubleML Analysis for HABS-HD Alzheimer's Study")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    print("1. Loading and preprocessing data...")
    bm = get_data()
    print(f"Final dataset shape: {bm.shape}")
    
    # Step 2: Fit causal model
    print("\n2. Fitting Causal Forest model...")
    cf, X, D, Y, dml_df = main()
    
    # Step 3: Generate visualizations and analyses
    print("\n3. Generating treatment effect visualizations...")
    
    # Plot distribution of treatment effects
    print("   a. CATE histogram...")
    plot_cate_histogram(X, cf)
    
    # Plot treatment effects vs age (overall and by sex)
    print("   b. CATE vs Age (stratified by sex)...")
    plot_cate_vs_X(X, cf, 'Age', stratify_by_sex=True, sex_column='ID_Gender')
    
    # Show top responders
    print("   c. Top 10 treatment responders...")
    age_index = X.columns.get_loc('Age')
    print_top10_cate(X, cf, age_index=age_index)
    
    # Fit meta-model to understand treatment effect predictors
    print("\n   d. Meta-model analysis...")
    tau_estimates = cf.effect(X)
    meta_model_fitted = meta_model(X, tau_estimates)
    
    # Plot dose-response relationship
    print("   e. Dose-response curve...")
    plot_dose_response(cf, X, D)
    
    print("\n" + "=" * 60)
    print("Analysis complete! Results saved in dml_df['cate'] column.")
    print("\nKey Findings:")
    print(f"- Average treatment effect: {tau_estimates.mean():.4f}")
    print(f"- Treatment effect heterogeneity (std): {tau_estimates.std():.4f}")
    print(f"- Sample size: {len(X)} participants")
    print(f"- Cases: {Y.sum()} | Controls: {len(Y) - Y.sum()}")
    
    # Clinical interpretation guidance
    print("\nClinical Interpretation:")
    print("- Positive CATE values indicate increased Alzheimer's risk with higher pTau-217")
    print("- Negative CATE values suggest protective or null effects")
    print("- Large heterogeneity suggests important subgroup differences")
    print("- Meta-model features reveal key moderators of treatment response")


"""
Additional Notes for Users:

Data Requirements:
- Excel files must be in the specified directory structure
- Required columns: Med_ID, Age, Visit_ID, pTau-217 measurements, clinical diagnoses
- APOE4 genotype data should be available in genomics file

Model Assumptions:
- Unconfoundedness: All confounders are observed and included in X
- Overlap: Treatment levels are observed across all covariate strata  
- Stable Unit Treatment Value Assumption (SUTVA): No interference between units

Interpretation Guidelines:
- CATE estimates represent individualized treatment effects
- Confidence intervals provide uncertainty quantification
- Meta-model reveals which characteristics predict treatment response
- Dose-response curves show optimal biomarker thresholds

For questions or issues, consult the econml documentation:
https://econml.azurewebsites.net/
"""
