import os
import pandas as pd
import numpy as np
import sys

# Add custom utility modules to the path
sys.path.append("../ukb_func")
import ml_utils

sys.path.append("../pet")
from build_datasets import (
    map_time_from_baseline,
    find_first_or_last_visit,
    replace_apoe,
    add_one_day_to_zero_time,
    create_stratified_folds,
)
from sklearn.preprocessing import StandardScaler


def process_nacc(nacc_file_path, nacc_csf_file_path, ad_outcome=False):
    """
    Process NACC (National Alzheimer's Coordinating Center) clinical and CSF data.

    This function loads, cleans, and processes clinical and CSF data from NACC,
    identifies cases and controls, and prepares the data for analysis.

    Args:
        nacc_file_path: Path to the NACC clinical data file
        nacc_csf_file_path: Path to the NACC CSF data file
        ad_outcome: Boolean flag to determine if Alzheimer's disease is the outcome (True)
                   or if all dementias are included (False)

    Returns:
        DataFrame containing processed CSF data with case/control labels
    """
    # Load raw NACC data
    # check if nacc_file_path ends in csv or parquet
    if nacc_file_path.endswith(".csv"):
        df = pd.read_csv(nacc_file_path)
    elif nacc_file_path.endswith(".parquet"):
        df = pd.read_parquet(nacc_file_path)
    else:
        raise ValueError(f"Invalid file extension: {nacc_file_path}")

    # Select relevant columns for analysis
    df = df.loc[
        :,
        [
            "NACCID",
            "NACCADC",
            "VISITDAY",
            "VISITMO",
            "VISITYR",
            "NACCACTV",
            "NACCNORM",
            "DEMENTED",
            "NACCADMD",
            "NACCALZD",
            "NACCALZP",
            "PROBAD",
            "PROBADIF",
            "POSSAD",
            "POSSADIF",
            "NACCETPR",
            "NACCLBDE",
            "NACCLBDP",
            "FTLDMO",
            "FTLDMOIF",
            "FTLDNOS",
            "FTLDNOIF",
            "FTD",
            "FTDIF",
            "PPAPH",
            "PPAPHIF",
            "CVDIF",
            "VASC",
            "VASCIF",
            "VASCPS",
            "VASCPSIF",
            "BIRTHMO",
            "BIRTHYR",
            "SEX",
            "RACE",
            "EDUC",
            "NACCAGE",
            "NACCAGEB",
            "NACCAPOE",
            "NACCNE4S",
            "NACCUDSD",
            "NACCTBI",
            "TBI",
            "TBIBRIEF",
            "TBIWOLOS",
            "TOBAC30",
            "TOBAC100",
            "SMOKYRS",
            "PACKSPER",
            "QUITSMOK",
            "ALCFREQ",
            "ALCOHOL",
            "ALCABUSE",
            "HYPERT",
            "HYPERTEN",
            "HXHYPER",
            "NACCAHTN",
            "NACCHTNC",
            "NACCBMI",
            "NACCDBMD",
            "DIABET",
            "DIABETES",
            "HEARING",
            "HEARAID",
            "HEARWAID",
            "DEPD",
            "DEPDSEV",
            "NACCGDS",
            "NACCADEP",
            "DEPTREAT",
            "DEP2YRS",
            "DEPOTHR",
        ],
    ]

    # rename NACCAPOE column to apoe and NACCID to id
    df.rename(columns={"NACCAPOE": "apoe", "NACCID": "id"}, inplace=True)

    # Create datetime columns for visits and sort chronologically
    df = df.sort_values(by=["id", "VISITYR", "VISITMO", "VISITDAY"])
    df["VISITDATE"] = pd.to_datetime(
        pd.DataFrame(
            {"year": df["VISITYR"], "month": df["VISITMO"], "day": df["VISITDAY"]}
        )
    )
    df["BIRTHDAY"] = pd.to_datetime(
        pd.DataFrame({"year": df["BIRTHYR"], "month": df["BIRTHMO"], "day": 15})
    )

    # Clean TBI (Traumatic Brain Injury) related columns
    # Replace unknown (9) and not available (-4) values with NaN
    tbi_cols = [
        "NACCTBI",
        "TBI",
        "TBIBRIEF",
        "TBIWOLOS",
        "TOBAC30",
        "TOBAC100",
        "ALCOHOL",
        "ALCABUSE",
        "HYPERTEN",
        "DIABET",
        "DIABETES",
        "HEARING",
        "HEARAID",
        "DEPD",
        "DEP2YRS",
        "DEPOTHR",
    ]
    for col in tbi_cols:
        df[col] = df[col].replace({9: np.nan, -4: np.nan})

    # Overwrite 8, 9, and -4 as NaN in the PACKSPER column
    for col in ["PACKSPER", "ALCFREQ", "HEARWAID", "DEPDSEV"]:
        df[col] = df[col].replace({8: np.nan, 9: np.nan, -4: np.nan})

    # Overwrite -4 as NaN in the HXHYPER column
    for col in ["HXHYPER", "NACCAHTN", "NACCHTNC", "NACCDBMD", "NACCADEP"]:
        df[col] = df[col].replace({-4: np.nan})

    # Overwrite 8 and -4 as NaN in the HYPERT column
    df["HYPERT"] = df["HYPERT"].replace({8: np.nan, -4: np.nan})
    df["DEPTREAT"] = df["DEPTREAT"].replace({8: np.nan, -4: np.nan})

    # Overwrite 88, 99, and -4 as NaN in the SMOKYRS column
    df["SMOKYRS"] = df["SMOKYRS"].replace({88: np.nan, 99: np.nan, -4: np.nan})

    # Clean quit smoking data
    df["QUITSMOK"] = df["QUITSMOK"].replace({888: np.nan, 999: np.nan, -4: np.nan})

    # Clean BMI data
    df["NACCBMI"] = df["NACCBMI"].replace({888.8: np.nan, -4: np.nan})
    df["NACCBMI"] = np.where(df["NACCBMI"] > 800, np.nan, df["NACCBMI"])

    # Clean depression scale data
    df["NACCGDS"] = df["NACCGDS"].replace({88: np.nan, -4: np.nan})

    # Clean education data
    df["EDUC"] = df["EDUC"].replace({99: np.nan})

    # Load and process CSF data
    csf = pd.read_csv(nacc_csf_file_path)
    csf.rename(
        columns={
            "NACCID": "id",
            "CSFLPYR": "year",
            "CSFLPMO": "month",
            "CSFLPDY": "day",
        },
        inplace=True,
    )
    csf["VISITDATE"] = pd.to_datetime(csf[["year", "month", "day"]])
    csf = csf.drop(columns=["year", "month", "day"])
    csf = csf.sort_values(by=["id", "VISITDATE"])

    # Find common subjects between clinical and CSF data
    common_ids = set(csf.id).intersection(set(df.id))
    print(f"Number of subjects with both clinical and CSF data: {len(common_ids)}")

    # Filter data to include only subjects with both clinical and PET data
    df = df[df.id.isin(common_ids)]
    csf = csf[csf.id.isin(common_ids)]

    # Calculate baseline visits and time from baseline
    earliest_baseline = find_first_or_last_visit(
        [csf, df],
        id_col="id",
        examdate_col="VISITDATE",
        first_or_last="first",
    )
    latest_date = find_first_or_last_visit(
        [csf, df],
        id_col="id",
        examdate_col="VISITDATE",
        first_or_last="last",
    )
    latest_date = map_time_from_baseline(
        latest_date, earliest_baseline, id_col="id", examdate_col="VISITDATE"
    )

    # Map time from baseline for both datasets
    csf = map_time_from_baseline(
        csf, earliest_baseline, id_col="id", examdate_col="VISITDATE"
    )
    df = map_time_from_baseline(
        df, earliest_baseline, id_col="id", examdate_col="VISITDATE"
    )

    # Exclude anyone with dementia at their first visit

    # ### outcomes
    # DEMENTED - dementia or not
    # NACCADMD - Reported current use of a FDAapproved medication for Alzheimer’s disease symptoms
    # NACCALZD - Presumptive etiologic diagnosis of the cognitive disorder — Alzheimer’s disease
    # NACCALZP - Primary, contributing, or noncontributing cause of observed cognitive impairment — Alzheimer’s disease (AD)
    # PROBAD - Presumptive etiologic diagnosis of the cognitive disorder — Probable Alzheimer’s disease
    # PROBADIF - Primary, contributing, or noncontributing cause of cognitive impairment — Probable Alzheimer’s disease
    # POSSAD - Presumptive etiologic diagnosis of the cognitive disorder — Possible Alzheimer’s disease
    # POSSADIF - Primary, contributing, or noncontributing cause of cognitive impairment — Possible Alzheimer’s disease
    # NACCETPR - Primary etiologic diagnosis (MCI); impaired, not MCI; or dementia

    # Define case subjects based on diagnosis criteria
    if ad_outcome:
        # For AD-specific outcome, only include subjects with AD diagnosis
        case_dx = df[
            (df.NACCADMD == 1)
            | (df.NACCALZD == 1)
            | (df.PROBAD == 1)
            | (df.PROBADIF == 1)
            | (df.POSSAD == 1)
            | (df.POSSADIF == 1)
            | (df.NACCETPR == 1)
        ]
    else:
        # For all dementias outcome, include subjects with any dementia diagnosis
        case_dx = df[
            (df.DEMENTED == 1)
            | (df.NACCADMD == 1)
            | (df.NACCALZD == 1)
            | (df.PROBAD == 1)
            | (df.PROBADIF == 1)
            | (df.POSSAD == 1)
            | (df.POSSADIF == 1)
            | (df.NACCETPR.isin([1, 7, 8, 27, 30]))
            | (df.NACCLBDE == 1)
            | (df.NACCLBDP == 1)
            # | (df.MSA == 1)
            # | (df.CORT == 1)
            | (df.FTLDMO == 1)
            | (df.FTLDMOIF == 1)
            | (df.FTLDNOS == 1)
            | (df.FTLDNOIF == 1)
            | (df.FTD == 1)
            | (df.FTDIF == 1)
            | (df.PPAPH == 1)
            | (df.PPAPHIF == 1)
            | (df.CVDIF == 1)
            | (df.VASC == 1)
            | (df.VASCIF == 1)
            | (df.VASCPS == 1)
            | (df.VASCPSIF == 1)
        ]

    # Get first diagnosis date for each case
    case_dx_first = case_dx.sort_values(by=["id", "VISITDATE"]).drop_duplicates(
        subset=["id"], keep="first"
    )

    # Process CSF scans for cases
    csf_case_df = []

    # For each case subject
    for c in case_dx_first.id.unique():
        c_csf = csf[csf.id == c]
        c_first_dx = case_dx_first[case_dx_first.id == c]
        # Only include CSF scans before diagnosis
        c_csf_before_dx = c_csf[c_csf["VISITDATE"] < c_first_dx["VISITDATE"].values[0]]
        if c_csf_before_dx.shape[0] == 0:
            continue
        else:
            # Add time-to-event information
            c_csf = c_csf.merge(
                c_first_dx[["id", "visit_to_days"]], on="id", how="left"
            )
            c_csf.rename({"visit_to_days_x": "time"}, axis=1, inplace=True)
            c_csf.rename({"visit_to_days_y": "time_to_event"}, axis=1, inplace=True)
            csf_case_df.append(c_csf)

    # Combine all case CSF data
    csf_case_df = pd.concat(csf_case_df, axis=0).reset_index(drop=True)

    # Process control subjects (those who never developed dementia)
    csf_control_df = csf[~csf.id.isin(case_dx_first.id.unique())]
    # Only include controls who have normal cognition at all visits (72 people get removed; N goes from 555 to 483)
    csf_control_df = csf_control_df[
        csf_control_df.id.isin(df[df.NACCNORM == 1].id.unique())
    ]

    # Add time-to-event information for controls (time to last visit)
    csf_control_df = csf_control_df.merge(
        latest_date[["id", "visit_to_days"]], on="id", how="left"
    )
    csf_control_df.rename({"visit_to_days_x": "time"}, axis=1, inplace=True)
    csf_control_df.rename({"visit_to_days_y": "time_to_event"}, axis=1, inplace=True)

    # Combine case and control data
    csf_df = pd.concat([csf_case_df, csf_control_df], axis=0).reset_index(drop=True)
    csf_df["event"] = [1] * csf_case_df.shape[0] + [0] * csf_control_df.shape[0]

    # Add demographic information
    print(f"Initial dataset size: {csf_df.shape}")
    csf_df = csf_df.merge(
        df.loc[
            :, ["id", "BIRTHDAY", "SEX", "apoe", "NACCNE4S", "EDUC"]
        ].drop_duplicates(),
        on="id",
        how="inner",
    )

    # rename NACCAPOE column to apoe
    print(f"Dataset size after adding demographics: {csf_df.shape}")

    # Calculate age at examination
    csf_df["age"] = (csf_df["VISITDATE"] - csf_df["BIRTHDAY"]).dt.days / 365.25
    return csf_df, df


def process_fold(data, fold_assignments, fold):
    """
    Process a single cross-validation fold by splitting data into training and validation sets
    and applying necessary preprocessing steps.

    Args:
        data: DataFrame containing the full dataset
        fold_assignments: DataFrame containing fold assignments for each subject
        fold: Integer indicating which fold to process (0-based index)

    Returns:
        Tuple of (train_set, val_set) DataFrames
    """
    # Merge fold assignments with original data
    data2 = data.merge(fold_assignments, on="id")

    # Define validation and training sets
    val_set = data2[data2["fold"] == fold].copy()
    train_set = data2[data2["fold"] != fold].copy()

    # Print fold information
    print(f"Fold {fold + 1}:")
    print(f"  Training IDs: {train_set['id'].nunique()}")
    print(f"  Validation IDs: {val_set['id'].nunique()}")
    print(f"  Positive class in Training: {train_set['event'].mean() * 100:.2f}%")
    print(f"  Positive class in Validation: {val_set['event'].mean() * 100:.2f}%")

    # Print time-to-event distribution for cases
    train_cases = train_set[train_set["event"] == 1]
    val_cases = val_set[val_set["event"] == 1]
    print("\nTime-to-event statistics for cases (years):")
    print("  Training:")
    print(f"    Mean: {train_cases['time_to_event'].mean():.2f}")
    print(f"    Median: {train_cases['time_to_event'].median():.2f}")
    print("  Validation:")
    print(f"    Mean: {val_cases['time_to_event'].mean():.2f}")
    print(f"    Median: {val_cases['time_to_event'].median():.2f}\n")

    # Preprocess data
    # center age
    centering_cols = ["age"]

    # Initialize the scaler
    scaler = StandardScaler(with_std=False)

    # Fit on only the continuous columns from training data
    scaler.fit(train_set[centering_cols])
    train_set[centering_cols] = scaler.transform(train_set[centering_cols])
    val_set[centering_cols] = scaler.transform(val_set[centering_cols])

    # center age squared
    train_set["age_squared"] = train_set.age**2
    val_set["age_squared"] = val_set.age**2

    # zscore other continuous variables
    train_set["ratio_ptau_abeta"] = train_set["CSFPTAU"] / train_set["CSFABETA"]
    val_set["ratio_ptau_abeta"] = val_set["CSFPTAU"] / val_set["CSFABETA"]

    zscore_cols = [
        "EDUC",
        "CSFABETA",
        "CSFPTAU",
        "CSFTTAU",
        # "CSFABMD",
        # "CSFPTMD",
        # "CSFTTMD",
        "ratio_ptau_abeta",
    ]

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit on only the continuous columns from training data
    scaler.fit(train_set[zscore_cols])

    train_set[zscore_cols] = scaler.transform(train_set[zscore_cols])
    val_set[zscore_cols] = scaler.transform(val_set[zscore_cols])

    # set id column to string
    train_set["id"] = train_set["id"].astype(str)
    val_set["id"] = val_set["id"].astype(str)

    return train_set, val_set


def make_cv_folds(output_path, ad_outcome=False):
    """
    Create cross-validation folds for the dataset and save them to disk.

    This function:
    1. Loads the processed dataset
    2. Encodes categorical and ordinal variables
    3. Creates stratified cross-validation folds
    4. Processes each fold
    5. Saves the training and validation sets for each fold

    Args:
        output_path: Path to save the output files
        ad_outcome: Boolean flag to determine if Alzheimer's disease is the outcome (True)
                   or if all dementias are included (False)
    """
    if ad_outcome:
        ad_outcome_path = "ad_outcome/"
    else:
        ad_outcome_path = ""

    nacc = pd.read_parquet(output_path + ad_outcome_path + "t2e_csf.parquet")

    # # encode sex, ethnicity, APOEe4 alleles, education qualifications
    # catcols = [
    #     "SEX",
    #     "RACE",
    #     "NACCNE4S",
    #     "NACCTBI",
    #     "TBI",
    #     "TOBAC30",
    #     "TOBAC100",
    #     "ALCOHOL",
    #     "ALCABUSE",
    #     "HYPERT",
    #     "HYPERTEN",
    #     "HXHYPER",
    #     "NACCAHTN",
    #     "NACCHTNC",
    #     "NACCDBMD",
    #     "DIABET",
    #     "DIABETES",
    #     "HEARING",
    #     "HEARAID",
    #     "HEARWAID",
    #     "DEPD",
    #     "NACCADEP",
    #     "DEPTREAT",
    #     "DEP2YRS",
    #     "DEPOTHR",
    # ]
    # categ_enc = ml_utils.encode_categorical_vars(nacc, catcols)

    # ordinal_cols = ["TBIBRIEF", "TBIWOLOS", "PACKSPER", "ALCFREQ", "DEPDSEV"]

    # ordinal_enc = ml_utils.encode_ordinal_vars(nacc, ordinal_cols)

    # other_cols = nacc.columns[~nacc.columns.isin(catcols + ordinal_cols)]

    # # concatenate encoded categorical, ordinal, and continuous variables
    # encoded_df = pd.concat(
    #     [
    #         nacc[other_cols].reset_index(drop=True),
    #         categ_enc.reset_index(drop=True),
    #         ordinal_enc.reset_index(drop=True),
    #     ],
    #     axis=1,
    # )
    # encoded_df.head()

    # # create an age column for the csf data by subtracting 'BIRTHMO' and 'BIRTHYR' from the 'csf_date'
    # encoded_df["csf_age"] = (
    #     encoded_df["csf_date"].dt.year
    #     - encoded_df["BIRTHYR"]
    #     + (encoded_df["csf_date"].dt.month - encoded_df["BIRTHMO"]) / 12
    # )

    # # controls are the ones with label 0
    # control_encoded = encoded_df[encoded_df.label == 0]
    # control_visit_age = (
    #     control_encoded.visit_date.dt.year
    #     - control_encoded["BIRTHYR"]
    #     + (control_encoded["csf_date"].dt.month - control_encoded["BIRTHMO"]) / 12
    # )
    # case_encoded = encoded_df[encoded_df.label == 1]
    # case_visit_age = (
    #     case_encoded.visit_date.dt.year
    #     - case_encoded["BIRTHYR"]
    #     + (case_encoded["csf_date"].dt.month - case_encoded["BIRTHMO"]) / 12
    # )
    # visit_age = pd.concat([control_visit_age, case_visit_age])
    # encoded_df["visit_age"] = visit_age

    # https://files.alz.washington.edu/documentation/dervarprev.pdf
    # 1 = 33
    # 2 = 34
    # 3 = 23
    # 4 = 44
    # 5 = 24
    # 6 = 22
    # 9 = NA
    nacc.loc[nacc.apoe == 1, "apoe"] = "33"
    nacc.loc[nacc.apoe == 2, "apoe"] = "34"
    nacc.loc[nacc.apoe == 3, "apoe"] = "23"
    nacc.loc[nacc.apoe == 4, "apoe"] = "44"
    nacc.loc[nacc.apoe == 5, "apoe"] = "24"
    nacc.loc[nacc.apoe == 6, "apoe"] = "22"
    nacc = replace_apoe(nacc)

    # divide time and time_to_event by 365.25
    nacc = add_one_day_to_zero_time(nacc)
    nacc["time"] = nacc["time"] / 365.25
    nacc["time_to_event"] = nacc["time_to_event"] / 365.25

    # # Define final columns to include in the dataset
    # final_cols = [
    #     "NACCID",
    #     "label",
    #     "year_csf",
    #     "first_dementia_date_encoded",
    #     "EDUC",
    #     "visit_age",
    #     "SMOKYRS",
    #     "QUITSMOK",
    #     "NACCBMI",
    #     "NACCGDS",
    #     "CSFABETA",
    #     "CSFPTAU",
    #     "CSFTTAU",
    #     "CSFABMD",
    #     "CSFABMDX",
    #     "CSFPTMD",
    #     "CSFPTMDX",
    #     "CSFTTMD",
    #     "CSFTTMDX",
    #     "csf_date",
    #     "SEX_2",
    #     "RACE_1",
    #     "RACE_2",
    #     "RACE_3",
    #     "RACE_4",
    #     "RACE_5",
    #     "RACE_50",
    #     "NACCNE4S_0",
    #     "NACCNE4S_1",
    #     "NACCNE4S_2",
    #     "NACCNE4S_9",
    #     "NACCTBI_0.0",
    #     "NACCTBI_1.0",
    #     "NACCTBI_nan",
    #     "TBI_0.0",
    #     "TBI_1.0",
    #     "TBI_2.0",
    #     "TBI_nan",
    #     "TOBAC30_0.0",
    #     "TOBAC30_1.0",
    #     "TOBAC30_nan",
    #     "TOBAC100_0.0",
    #     "TOBAC100_1.0",
    #     "TOBAC100_nan",
    #     "ALCOHOL_0.0",
    #     "ALCOHOL_1.0",
    #     "ALCOHOL_2.0",
    #     "ALCOHOL_nan",
    #     "ALCABUSE_0.0",
    #     "ALCABUSE_1.0",
    #     "ALCABUSE_8.0",
    #     "ALCABUSE_nan",
    #     "HYPERT_0.0",
    #     "HYPERT_1.0",
    #     "HYPERT_nan",
    #     "HYPERTEN_0.0",
    #     "HYPERTEN_1.0",
    #     "HYPERTEN_2.0",
    #     "HYPERTEN_nan",
    #     "HXHYPER_0.0",
    #     "HXHYPER_1.0",
    #     "HXHYPER_nan",
    #     "NACCAHTN_1.0",
    #     "NACCHTNC_1.0",
    #     "NACCDBMD_1.0",
    #     "DIABET_0.0",
    #     "DIABET_1.0",
    #     "DIABET_2.0",
    #     "DIABET_nan",
    #     "DIABETES_0.0",
    #     "DIABETES_1.0",
    #     "DIABETES_2.0",
    #     "DIABETES_nan",
    #     "HEARING_0.0",
    #     "HEARING_1.0",
    #     "HEARING_nan",
    #     "HEARAID_0.0",
    #     "HEARAID_1.0",
    #     "HEARAID_nan",
    #     "HEARWAID_0.0",
    #     "HEARWAID_1.0",
    #     "HEARWAID_nan",
    #     "DEPD_0.0",
    #     "DEPD_1.0",
    #     "DEPD_nan",
    #     "NACCADEP_1.0",
    #     "DEPTREAT_0.0",
    #     "DEPTREAT_1.0",
    #     "DEPTREAT_nan",
    #     "DEP2YRS_0.0",
    #     "DEP2YRS_1.0",
    #     "DEP2YRS_nan",
    #     "DEPOTHR_0.0",
    #     "DEPOTHR_1.0",
    #     "DEPOTHR_nan",
    #     "TBIBRIEF",
    #     "TBIWOLOS",
    #     "PACKSPER",
    #     "ALCFREQ",
    #     "DEPDSEV",
    #     "csf_age",
    # ]

    # data = nacc[final_cols]
    # data = data.rename(columns={"NACCID": "id"})

    data = nacc
    ### Create stratified folds combining all cohorts
    fold_assignments = create_stratified_folds(data)

    for fold in range(5):
        # Process the fold
        train_set, val_set = process_fold(data, fold_assignments, fold)

        # print overlapping BIDs between training and validation sets
        train_bids = set(train_set["id"])
        val_bids = set(val_set["id"])
        overlap = train_bids.intersection(val_bids)
        print(f"  Overlapping IDs: {len(overlap)}\n")

        # Save datasets
        if ad_outcome:
            train_set.to_parquet(
                output_path + ad_outcome_path + f"train_{fold}.parquet"
            )
            val_set.to_parquet(output_path + ad_outcome_path + f"val_{fold}.parquet")
        else:
            train_set.to_parquet(output_path + f"train_{fold}.parquet")
            val_set.to_parquet(output_path + f"val_{fold}.parquet")


def main():
    """
    Main function that orchestrates the entire data processing pipeline:
    1. Process NACC data for both all dementias and AD-specific outcomes
    2. Create cross-validation folds for both outcomes
    """
    main_path = "../../../datasets/"
    nacc_file_path = "../../nacc/tidy_data/investigator_ftldlbd_nacc65.parquet"
    nacc_csf_file_path = main_path + "NACC/csv/raw/investigator_fcsf_nacc65.csv"
    output_path = "../../nacc/tidy_data/"

    # Process data for all dementias outcome
    csf_df, df = process_nacc(nacc_file_path, nacc_csf_file_path, ad_outcome=False)
    os.makedirs(output_path, exist_ok=True)
    csf_df.to_parquet(output_path + "t2e_csf.parquet")
    df.to_parquet(output_path + "nacc_clinical_demographics.parquet")

    # Process data for AD-specific outcome
    csf_df, df = process_nacc(nacc_file_path, nacc_csf_file_path, ad_outcome=True)
    os.makedirs(output_path + "ad_outcome/", exist_ok=True)
    csf_df.to_parquet(output_path + "ad_outcome/t2e_csf.parquet")
    df.to_parquet(output_path + "ad_outcome/nacc_clinical_demographics.parquet")

    #################################
    # Create cross-validation folds #
    #################################
    make_cv_folds(output_path)
    make_cv_folds(output_path, ad_outcome=True)


if __name__ == "__main__":
    main()
