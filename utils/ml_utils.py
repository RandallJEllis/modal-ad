import pandas as pd
import numpy as np
import sys
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_curve,
    auc,
    matthews_corrcoef,
)
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import os
from utils import save_pickle
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pyarrow as pa


def concat_labels_and_probas(dirpath):
    true_labels = []
    probas = []

    if "mci" not in dirpath:
        if "nacc" not in dirpath:
            for i in range(10):
                tl = pickle.load(
                    open(f"{dirpath}/test_true_labels_region_{i}.pkl", "rb")
                )
                true_labels.append(tl[0])

                p = pickle.load(open(f"{dirpath}/test_probas_region_{i}.pkl", "rb"))

                if "feature_selection" in dirpath:
                    df = pd.read_csv(f"{dirpath}/training_results_region_{i}.csv")
                    df = df.iloc[:20]
                    best_idx = df["auroc"].idxmax()
                    probas.append(p[best_idx])
                else:
                    probas.append(p[0])
        else:
            # Handle NACC data
            tl = pickle.load(open(f"{dirpath}/test_true_labels_region_9.pkl", "rb"))
            p = pickle.load(open(f"{dirpath}/test_probas_region_9.pkl", "rb"))

            for i in range(10):
                true_labels.append(tl[i])
                probas.append(p[i])
    else:
        # Handle MCI data
        tl = pickle.load(open(f"{dirpath}/test_true_labels.pkl", "rb"))
        true_labels.append(tl[0])

        p = pickle.load(open(f"{dirpath}/test_probas.pkl", "rb"))
        probas.append(p[0])

    return true_labels, probas


def mcc_from_conf_mtx(tp, fp, tn, fn):
    return (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def encode_categorical_vars(df, catcols):
    enc = OneHotEncoder(drop="if_binary")
    enc.fit(df.loc[:, catcols])
    categ_enc = pd.DataFrame(
        enc.transform(df.loc[:, catcols]).toarray(),
        columns=enc.get_feature_names_out(catcols),
    )
    return categ_enc


def encode_ordinal_vars(df, ordvars):
    enc = OrdinalEncoder()
    enc.fit(df.loc[:, ordvars])
    ord_enc = pd.DataFrame(
        enc.transform(df.loc[:, ordvars]), columns=enc.get_feature_names_out(ordvars)
    )
    return ord_enc


def pick_threshold(y_true, y_probas, youden=True, beta=1):
    scores = []

    if youden is True:
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(y_true, y_probas)

        for i, t in enumerate(thresholds):
            # youden index = sensitivity + specificity - 1
            # AKA sensitivity + (1 - FPR) - 1 (NOTE: (1-FPR) = TNR)
            # AKA recall_1 + recall_0 - 1
            youdens_j = tpr[i] + (1 - fpr[i]) - 1
            scores.append(youdens_j)

    else:
        # calculate pr-curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_probas)

        # convert to f score
        for i, t in enumerate(thresholds):
            fscore = ((1 + beta**2) * precision[i] * recall[i]) / (
                (beta**2 * precision[i]) + recall[i]
            )
            scores.append(fscore)

    ix = np.nanargmax(scores)
    best_threshold = thresholds[ix]

    return best_threshold


def pseudo_r2(y_true, y_pred):
    from sklearn.metrics import log_loss
    
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1-eps)
    
    # Log-likelihoods
    LL_full = -log_loss(y_true, y_pred, normalize=False)
    
    # Null model uses mean probability
    p_null = np.mean(y_true)
    LL_null = -log_loss(y_true, np.full_like(y_true, p_null), normalize=False)
    
    # McFadden
    r2_mcfadden = 1 - (LL_full / LL_null)

    # Cox-Snell
    n = len(y_true)
    r2_cs = 1 - np.exp((LL_null - LL_full) * 2 / n)
    
    # Nagelkerke
    r2_nagelkerke = r2_cs / (1 - np.exp(-2 * LL_null / n))
    
    # Tjur's
    p1 = y_pred[y_true == 1].mean()
    p0 = y_pred[y_true == 0].mean()
    r2_tjur = p1 - p0
    
    # Efron
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - p_null)**2)
    r2_efron = 1 - ss_res / ss_tot
    
    return {
        "McFadden_r2": r2_mcfadden,
        "Cox-Snell_r2": r2_cs,
        "Nagelkerke_r2": r2_nagelkerke,
        "Tjur_r2": r2_tjur,
        "Efron_r2": r2_efron
    }
    
    
def brier_decomp(y_true, y_probas):
    # brier decomposition
    df = pd.DataFrame({'obs': y_true, 'preds': y_probas})

    # Use qcut to create equal-frequency bins (deciles)
    # duplicates='drop' handles cases where many preds are identical
    df['bin'] = pd.qcut(df['preds'], 10, labels=False, duplicates='drop')
    
    overall_mean_obs = df['obs'].mean()
    total_n = len(df)
    
    rel = 0
    res = 0
    
    # Group by bin to calculate components
    bin_stats = df.groupby('bin').agg(
        bin_n=('obs', 'count'),
        bin_mean_preds=('preds', 'mean'),
        bin_mean_obs=('obs', 'mean')
    )
    
    for _, row in bin_stats.iterrows():
        rel += row['bin_n'] * (row['bin_mean_preds'] - row['bin_mean_obs'])**2
        res += row['bin_n'] * (row['bin_mean_obs'] - overall_mean_obs)**2
    
    reliability = rel / total_n
    resolution = res / total_n
    uncertainty = overall_mean_obs * (1 - overall_mean_obs)

    return reliability, resolution, uncertainty


def calc_results(
    y_true, y_probas, youden=True, beta=1, threshold=None, suppress_output=True
):
    auroc = roc_auc_score(y_true, y_probas)
    ap = average_precision_score(y_true, y_probas)
    brier = brier_score_loss(y_true, y_probas, pos_label=1)

    reliability, resolution, uncertainty = brier_decomp(y_true, y_probas)
    
    # if metric == 'roc_auc':
    #     youden = True

    # return_threshold = False
    # if threshold is None:
    #     threshold = pick_threshold(y_true, y_probas, youden, beta)
    #     return_threshold = True

    return_threshold = False
    if threshold is not None:
        pass
    else:
        threshold = pick_threshold(y_true, y_probas, youden, beta)
        return_threshold = True

    test_pred = (y_probas >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, test_pred).ravel()
    acc = accuracy_score(y_true, test_pred)
    bal_acc = balanced_accuracy_score(y_true, test_pred)
    prfs = precision_recall_fscore_support(y_true, test_pred, beta=beta)
    mcc = matthews_corrcoef(y_true, test_pred)
    pseudor2 = pseudo_r2(y_true, y_probas)

    # print(f'AUROC: {auroc}, AP: {ap}, Fscore: {best_fscore}, Accuracy: {acc}, Bal. Acc.: {bal_acc}, Best threshold: {best_threshold}')
    if suppress_output:
        pass
    else:
        print(
            f"AUROC: {np.round(auroc, 4)}, AP: {np.round(ap, 4)}, \nAccuracy: {np.round(acc, 4)}, Bal. Acc.: {np.round(bal_acc, 4)}, \nBest threshold: {np.round(threshold, 4)}"
        )
        print(f"Precision/Recall/Fscore: {prfs}")
        print("\n")
    res = pd.Series(
        data=[
            auroc,
            ap,
            threshold,
            brier,
            reliability,
            resolution,
            uncertainty,
            tp,
            tn,
            fp,
            fn,
            acc,
            bal_acc,
            prfs[0][0], # precision negative (NPV)
            prfs[0][1], # precision positive (PPV)
            prfs[1][0], # recall negative (Sensitivity)
            prfs[1][1], # recall positive (Specificity)
            prfs[2][0], # fbeta negative
            prfs[2][1], # fbeta positive
            mcc,
            pseudor2["McFadden_r2"],
            pseudor2["Cox-Snell_r2"],
            pseudor2["Nagelkerke_r2"],
            pseudor2["Tjur_r2"],
            pseudor2["Efron_r2"],
        ],
        index=[
            "auroc",
            "avg_prec",
            "threshold",
            'brier',
            'brier_reliability',
            'brier_resolution',
            'brier_uncertainty',
            "TP",
            "TN",
            "FP",
            "FN",
            "accuracy",
            "bal_acc",
            "prec_n",
            "prec_p",
            "recall_n",
            "recall_p",
            f"f{beta}_n",
            f"f{beta}_p",
            "mcc",
            "McFadden_r2",
            "Cox-Snell_r2",
            "Nagelkerke_r2",
            "Tjur_r2",
            "Efron_r2",
        ],
    )
    if return_threshold == True:
        return res, threshold
    else:
        return res
    # return res


def save_labels_probas(
    filepath,
    train_labels,
    train_probas,
    test_labels,
    test_probas,
    other_file_info="",
    survival=False,
    surv_model=None,
    train_surv_fn=None,
    test_surv_fn=None,
):
    save_pickle(f"{filepath}/train_true_labels{other_file_info}.pkl", train_labels)
    save_pickle(f"{filepath}/train_probas{other_file_info}.pkl", train_probas)
    save_pickle(f"{filepath}/test_true_labels{other_file_info}.pkl", test_labels)
    save_pickle(f"{filepath}/test_probas{other_file_info}.pkl", test_probas)

    if survival is True:
        save_pickle(f"{filepath}/surv_model{other_file_info}.pkl", surv_model)

        train_surv_fn = pd.DataFrame(train_surv_fn)
        test_surv_fn = pd.DataFrame(test_surv_fn)

        print("Saving training survival functions")
        start_time = datetime.now()
        train_surv_fn.to_parquet(
            f"{filepath}/train_survival_fns{other_file_info}.parquet", engine="pyarrow"
        )
        end_time = datetime.now()
        print(f"pyarrow, Time to save: {end_time - start_time}")

        print("Saving test survival functions")
        start_time = datetime.now()
        test_surv_fn.to_parquet(
            f"{filepath}/test_survival_fns{other_file_info}.parquet", engine="pyarrow"
        )
        end_time = datetime.now()
        print(f"pyarrow, Time to save: {end_time - start_time}")

        # pa_table = pa.table({"train_survival_functions": train_surv_fn})
        # pa.parquet.write_table(pa_table, f"{filepath}/train_survival_fns{other_file_info}.parquet")

        # pa_table = pa.table({"test_survival_functions": test_surv_fn})
        # pa.parquet.write_table(pa_table, f"{filepath}/test_survival_fns{other_file_info}.parquet")
        # np.save(f'{filepath}/train_survival_fns{other_file_info}.npy', train_surv_fn, allow_pickle=False)
        # np.save(f'{filepath}/test_survival_fns{other_file_info}.npy', test_surv_fn, allow_pickle=False)


# def get_fold_number(fname):
#     last_underscore = fname.rfind('_')
#     last_period = fname.rfind('.')
#     fold = fname[last_underscore+1:last_period]
#     return fold

# def sort_fold_results(fold_numbers, fold_results):
#     # Pair strings with their corresponding numbers
#     paired_list = list(zip(fold_numbers, fold_results))

#     # Sort the paired list based on the numbers
#     sorted_paired_list = sorted(paired_list)

#     # Extract the sorted strings
#     sorted_results = [fold_result for fold_number, fold_result in sorted_paired_list]
#     sorted_results = pd.concat(sorted_results)

#     return sorted_results


def concat_results(filepath):
    train_results = []
    test_results = []

    for i in range(10):
        train_results.append(
            pd.read_csv(f"{filepath}/training_results_region_{i}.csv", index_col=0)
        )
        test_results.append(
            pd.read_csv(f"{filepath}/test_results_region_{i}.csv", index_col=0)
        )

    train_results = pd.concat(train_results)
    test_results = pd.concat(test_results)
    return train_results, test_results
    # for fname in file_list:
    #     if fname[:2] == '._':
    #         continue
    #     if '.csv' in fname:
    #         if 'training_results' in fname and 'region' in fname:
    #             train_results.append(pd.read_csv(f'{filepath}/{fname}', index_col=0))

    #             fold = get_fold_number(fname)
    #             train_fold.append(fold)

    #         elif 'test_results' in fname and 'region' in fname:
    #             test_results.append(pd.read_csv(f'{filepath}/{fname}', index_col=0))

    #             fold = get_fold_number(fname)
    #             test_fold.append(fold)

    # train_results = sort_fold_results(train_fold, train_results)
    # test_results = sort_fold_results(test_fold, test_results)


def concat_and_save_results(filepath):

    train_results, test_results = concat_results(filepath)

    train_results.to_csv(f"{filepath}/train_results.csv")
    test_results.to_csv(f"{filepath}/test_results.csv")


def probas_to_results(filepath, youden=True, beta=1, threshold=None):
    train_res_l = []
    test_res_l = []

    if "mci" not in filepath:

        if "nacc" not in filepath:
            for i in range(10):
                train_labels = pickle.load(
                    open(f"{filepath}/train_true_labels_region_{i}.pkl", "rb")
                )
                test_labels = pickle.load(
                    open(f"{filepath}/test_true_labels_region_{i}.pkl", "rb")
                )
                train_probas = pickle.load(
                    open(f"{filepath}/train_probas_region_{i}.pkl", "rb")
                )
                test_probas = pickle.load(
                    open(f"{filepath}/test_probas_region_{i}.pkl", "rb")
                )

                if "feature_selection" in filepath:
                    df = pd.read_csv(f"{filepath}/training_results_region_{i}.csv")
                    df = df.iloc[:20]
                    best_idx = df["auroc"].idxmax()

                    # res = calc_results(test_labels[0], test_probas[best_idx], youden=youden, beta=beta, threshold=threshold)
                    # res_l.append(res)

                    train_res, thresh = calc_results(
                        train_labels[0],
                        train_probas[best_idx],
                        youden=youden,
                        beta=beta,
                        threshold=threshold,
                    )
                    res = calc_results(
                        test_labels[0],
                        test_probas[best_idx],
                        youden=youden,
                        beta=beta,
                        threshold=thresh,
                    )
                    train_res_l.append(train_res)
                    test_res_l.append(res)

                else:
                    train_res, thresh = calc_results(
                        train_labels[0],
                        train_probas[0],
                        youden=youden,
                        beta=beta,
                        threshold=threshold,
                    )
                    res = calc_results(
                        test_labels[0],
                        test_probas[0],
                        youden=youden,
                        beta=beta,
                        threshold=thresh,
                    )
                    train_res_l.append(train_res)
                    test_res_l.append(res)
        else:
            train_labels = pickle.load(
                open(f"{filepath}/train_true_labels_region_9.pkl", "rb")
            )
            test_labels = pickle.load(
                open(f"{filepath}/test_true_labels_region_9.pkl", "rb")
            )
            train_probas = pickle.load(
                open(f"{filepath}/train_probas_region_9.pkl", "rb")
            )
            test_probas = pickle.load(
                open(f"{filepath}/test_probas_region_9.pkl", "rb")
            )

            for i in range(10):
                train_res, thresh = calc_results(
                    train_labels[i],
                    train_probas[i],
                    youden=youden,
                    beta=beta,
                    threshold=threshold,
                )
                res = calc_results(
                    test_labels[i],
                    test_probas[i],
                    youden=youden,
                    beta=beta,
                    threshold=thresh,
                )
                train_res_l.append(train_res)
                test_res_l.append(res)

    else:
        test_labels = pickle.load(open(f"{filepath}/test_true_labels.pkl", "rb"))
        test_probas = pickle.load(open(f"{filepath}/test_probas.pkl", "rb"))
        res = calc_results(
            test_labels[0],
            test_probas[0],
            youden=youden,
            beta=beta,
            threshold=threshold,
        )
        test_res_l.append(res)

    train_results = pd.concat(train_res_l, axis=1).T
    test_results = pd.concat(test_res_l, axis=1).T

    return train_results, test_results


# if __name__ == "__main__":
#    if len(sys.argv) > 1:
#        function_name = sys.argv[1]
#        args = sys.argv[2:]
#        if function_name in globals():
#            globals()[function_name](*args)
#        else:
#            print(f"No function named '{function_name}' found.")
#    else:
#        print("No function name provided.")


from sklearn.metrics import (
    RocCurveDisplay,
    roc_curve,
    auc,
    roc_auc_score,
    d2_absolute_error_score,
    d2_pinball_score,
    d2_tweedie_score,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_tweedie_deviance,
    mean_pinball_loss,
    root_mean_squared_error,
    root_mean_squared_log_error,
)


def calculate_regression_metrics(y_true, y_pred, tweedie_power=0):
    """
    Calculates various regression metrics for predictions and true values.

    Parameters:
    - y_true: array-like of shape (n_samples,) True target values.
    - y_pred: array-like of shape (n_samples,) Predicted target values.
    - tweedie_power: Power parameter for Tweedie distribution deviance and D2 Tweedie score.
        Default is 0 (Gaussian).

    Returns:
    - metrics_dict: Dictionary containing all calculated metrics.
    """
    metrics_dict = {
        "r2_score": r2_score(y_true, y_pred),
        "median_absolute_error": median_absolute_error(y_true, y_pred),
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "mean_squared_error": mean_squared_error(y_true, y_pred),
        "d2_absolute_error_score": d2_absolute_error_score(y_true, y_pred),
        "d2_pinball_score": d2_pinball_score(y_true, y_pred, alpha=0.5),
        "d2_tweedie_score": d2_tweedie_score(y_true, y_pred, power=tweedie_power),
        "explained_variance_score": explained_variance_score(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(
            y_true, y_pred
        ),
        "mean_pinball_loss": mean_pinball_loss(y_true, y_pred, alpha=0.5),
        "root_mean_squared_error": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

    # check if all y_true and y_pred values are positive for mean_gamma_deviance
    # if all(y_true >= 0):
    # metrics_dict["root_mean_squared_log_error"] = np.sqrt(mean_squared_log_error(y_true, y_pred))

    # check if all y_true and y_pred values are positive for mean_gamma_deviance
    if all(y_true > 0) and all(y_pred > 0):
        metrics_dict["root_mean_squared_log_error"] = np.sqrt(
            mean_squared_log_error(y_true, y_pred)
        )
        metrics_dict["mean_squared_log_error"] = mean_squared_log_error(y_true, y_pred)
        metrics_dict["mean_gamma_deviance"] = mean_gamma_deviance(y_true, y_pred)
        metrics_dict["mean_tweedie_deviance"] = mean_tweedie_deviance(
            y_true, y_pred, power=tweedie_power
        )
        metrics_dict["mean_poisson_deviance"] = mean_poisson_deviance(y_true, y_pred)
    return metrics_dict


def _compute_categorical_nri_single(y_true, y_prob_old, y_prob_new, cuts=None):
    """
    Compute categorical NRI for a single fold/sample.
    
    Internal helper function used by calculate_nri_from_paths.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_prob_old : array-like
        Predicted probabilities from the reference model
    y_prob_new : array-like
        Predicted probabilities from the new model
    cuts : array-like, optional
        Risk category boundaries. Default: [0, 0.1, 0.2, ..., 1.0]
        
    Returns:
    --------
    dict : {'nri': float, 'nri_event': float, 'nri_nonevent': float, 'n_events': int, 'n_nonevents': int}
    """
    # Handle nested arrays from pickle files
    if hasattr(y_true, '__len__') and len(y_true) > 0:
        if isinstance(y_true[0], (list, np.ndarray)) and len(y_true) == 1:
            y_true = y_true[0]
    if hasattr(y_prob_old, '__len__') and len(y_prob_old) > 0:
        if isinstance(y_prob_old[0], (list, np.ndarray)) and len(y_prob_old) == 1:
            y_prob_old = y_prob_old[0]
    if hasattr(y_prob_new, '__len__') and len(y_prob_new) > 0:
        if isinstance(y_prob_new[0], (list, np.ndarray)) and len(y_prob_new) == 1:
            y_prob_new = y_prob_new[0]
    
    y_true = np.asarray(y_true)
    y_prob_old = np.asarray(y_prob_old)
    y_prob_new = np.asarray(y_prob_new)
    
    if cuts is None:
        cuts = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cuts = np.asarray(cuts)
    
    def _assign_risk_category(probs, cuts):
        return np.digitize(probs, cuts[1:])
    
    events = y_true == 1
    nonevents = y_true == 0
    n_events = int(np.sum(events))
    n_nonevents = int(np.sum(nonevents))
    
    if n_events == 0 or n_nonevents == 0:
        return {'nri': np.nan, 'nri_event': np.nan, 'nri_nonevent': np.nan, 
                'n_events': n_events, 'n_nonevents': n_nonevents}
    
    cat_old = _assign_risk_category(y_prob_old, cuts)
    cat_new = _assign_risk_category(y_prob_new, cuts)
    
    events_up = np.sum(cat_new[events] > cat_old[events])
    events_down = np.sum(cat_new[events] < cat_old[events])
    nri_event = (events_up - events_down) / n_events
    
    nonevents_up = np.sum(cat_new[nonevents] > cat_old[nonevents])
    nonevents_down = np.sum(cat_new[nonevents] < cat_old[nonevents])
    nri_nonevent = (nonevents_down - nonevents_up) / n_nonevents
    
    nri = nri_event + nri_nonevent
    
    return {
        'nri': nri,
        'nri_event': nri_event,
        'nri_nonevent': nri_nonevent,
        'n_events': n_events,
        'n_nonevents': n_nonevents
    }


def calculate_nri_from_paths(filepath_old, filepath_new, cuts=None):
    """
    Calculate NRI across all folds from two model directories.
    
    Loads test_probas_* and test_true_labels_* files from each directory,
    matches them by fold identifier, and calculates NRI for each fold.
    
    Parameters:
    -----------
    filepath_old : str
        Path to directory containing reference model results.
        Must contain test_probas_*.pkl and test_true_labels_*.pkl files.
    filepath_new : str
        Path to directory containing new model results.
        Must contain test_probas_*.pkl files (uses labels from filepath_old).
    cuts : array-like, optional
        Risk category boundaries for categorical NRI.
        Default is [0, 0.1, 0.2, ..., 1.0]
        
    Returns:
    --------
    pd.DataFrame : DataFrame with columns ['fold', 'nri', 'nri_event', 'nri_nonevent', 
                                           'n_events', 'n_nonevents']
    """
    import glob
    import re
    
    # Find all test_probas files in both directories
    old_probas_files = sorted(glob.glob(f"{filepath_old}/test_probas_*.pkl"))
    new_probas_files = sorted(glob.glob(f"{filepath_new}/test_probas_*.pkl"))
    
    if not old_probas_files:
        raise FileNotFoundError(f"No test_probas_*.pkl files found in {filepath_old}")
    if not new_probas_files:
        raise FileNotFoundError(f"No test_probas_*.pkl files found in {filepath_new}")
    
    # Extract fold identifiers from filenames
    def get_fold_id(filepath):
        basename = os.path.basename(filepath)
        # Match patterns like test_probas_region_0.pkl or test_probas_fold_1.pkl
        match = re.search(r'test_probas_(.+)\.pkl', basename)
        return match.group(1) if match else basename
    
    old_folds = {get_fold_id(f): f for f in old_probas_files}
    new_folds = {get_fold_id(f): f for f in new_probas_files}
    
    # Find matching folds
    common_folds = sorted(set(old_folds.keys()) & set(new_folds.keys()))
    
    if not common_folds:
        raise ValueError(f"No matching folds found between directories. "
                        f"Old: {list(old_folds.keys())}, New: {list(new_folds.keys())}")
    
    results = []
    
    for fold_id in common_folds:
        # Load probabilities
        old_probas = pickle.load(open(old_folds[fold_id], 'rb'))
        new_probas = pickle.load(open(new_folds[fold_id], 'rb'))
        
        # Load labels from the old model directory
        labels_file = old_folds[fold_id].replace('test_probas_', 'test_true_labels_')
        if not os.path.exists(labels_file):
            print(f"Warning: Labels file not found: {labels_file}, skipping fold {fold_id}")
            continue
        
        labels = pickle.load(open(labels_file, 'rb'))
        
        # Calculate NRI for this fold
        nri_result = _compute_categorical_nri_single(labels, old_probas, new_probas, cuts)
        nri_result['fold'] = fold_id
        results.append(nri_result)
    
    df = pd.DataFrame(results)
    # Reorder columns
    cols = ['fold', 'nri', 'nri_event', 'nri_nonevent', 'n_events', 'n_nonevents']
    return df[cols]

def plot_nri_boxplot(
    nri_df=None,
    filepath_old=None,
    filepath_new=None,
    model_old_name="Reference Model",
    model_new_name="New Model",
    figsize=(10, 6),
    save_path=None,
    cuts=None,
):
    """
    Plot NRI as box plots showing distribution across folds.
    
    Can take either a pre-computed DataFrame from calculate_nri_from_paths
    or two directory paths to compute NRI across all folds.
    
    Args:
        nri_df (pd.DataFrame, optional): Pre-computed NRI results from calculate_nri_from_paths().
            Must contain columns: 'nri', 'nri_event', 'nri_nonevent'.
        filepath_old (str, optional): Path to directory with reference model results.
        filepath_new (str, optional): Path to directory with new model results.
        model_old_name (str): Display name for the reference model.
        model_new_name (str): Display name for the new model.
        figsize (tuple): Figure size (width, height).
        save_path (str): Path to save the figure. If None, displays the plot.
        cuts (array-like, optional): Risk category boundaries (only used if computing from paths).
    
    Returns:
        pd.DataFrame: DataFrame with NRI values for each fold plus summary statistics.
    """
    # Get or compute NRI DataFrame
    if nri_df is not None:
        df = nri_df.copy()
    elif filepath_old is not None and filepath_new is not None:
        df = calculate_nri_from_paths(filepath_old, filepath_new, cuts=cuts)
    else:
        raise ValueError("Must provide either nri_df or (filepath_old, filepath_new)")
    
    # Prepare data for box plot
    plot_data = []
    for _, row in df.iterrows():
        plot_data.append({'Metric': 'NRI+\n(Events)', 'Value': row['nri_event']})
        plot_data.append({'Metric': 'NRI-\n(Non-events)', 'Value': row['nri_nonevent']})
        plot_data.append({'Metric': 'Overall NRI', 'Value': row['nri']})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for each metric
    colors = {'NRI+\n(Events)': '#3498db', 'NRI-\n(Non-events)': '#9b59b6', 'Overall NRI': '#2ecc71'}
    metric_order = ['NRI+\n(Events)', 'NRI-\n(Non-events)', 'Overall NRI']
    
    # Create box plot
    positions = [0, 1, 2]
    bp_data = [plot_df[plot_df['Metric'] == m]['Value'].values for m in metric_order]
    
    bp = ax.boxplot(bp_data, positions=positions, widths=0.6, patch_artist=True)
    
    # Color the boxes
    for patch, metric in zip(bp['boxes'], metric_order):
        patch.set_facecolor(colors[metric])
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Style median lines
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    # Add individual data points
    for i, metric in enumerate(metric_order):
        values = plot_df[plot_df['Metric'] == metric]['Value'].values
        x = np.random.normal(i, 0.04, size=len(values))
        ax.scatter(x, values, alpha=0.6, color='darkgray', s=50, zorder=3, edgecolors='black')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Add mean annotations
    for i, metric in enumerate(metric_order):
        values = plot_df[plot_df['Metric'] == metric]['Value'].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.annotate(
            f'Mean: {mean_val:.3f}\n(SD: {std_val:.3f})',
            xy=(i, max(values) + 0.02),
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )
    
    # Customize plot
    ax.set_xticks(positions)
    ax.set_xticklabels(metric_order)
    ax.set_ylabel('Net Reclassification Improvement', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Net Reclassification Improvement\n{model_new_name} vs {model_old_name}\n(n={len(df)} folds)',
        fontsize=16, fontweight='bold'
    )
    
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Set y-axis limits with some padding
    all_vals = plot_df['Value'].values
    y_min = min(min(all_vals), 0) - 0.15
    y_max = max(max(all_vals), 0) + 0.2
    ax.set_ylim(y_min, y_max)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add sample size info
    mean_n_events = int(df['n_events'].mean())
    mean_n_nonevents = int(df['n_nonevents'].mean())
    ax.text(
        0.02, 0.02, 
        f"Avg per fold: Events={mean_n_events}, Non-events={mean_n_nonevents}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        fig.savefig(save_path, facecolor='white', transparent=False, dpi=300)
        plt.close()
    else:
        plt.show()
    
    # Add summary statistics to the DataFrame
    summary = {
        'nri_mean': df['nri'].mean(),
        'nri_std': df['nri'].std(),
        'nri_event_mean': df['nri_event'].mean(),
        'nri_event_std': df['nri_event'].std(),
        'nri_nonevent_mean': df['nri_nonevent'].mean(),
        'nri_nonevent_std': df['nri_nonevent'].std(),
    }
    
    return df, summary


def decision_curve_analysis(y_true, y_prob, thresholds=None, model_name='Model'):
    """
    Perform Decision Curve Analysis (DCA) for binary classification.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_prob : array-like
        Predicted probabilities from the model
    thresholds : array-like, optional
        Threshold probabilities to evaluate. Default is np.arange(0.01, 1.0, 0.01)
    model_name : str, default='Model'
        Name identifier for the model
        
    Returns:
    --------
    pd.DataFrame : DataFrame with net benefit calculations
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    
    n = len(y_true)
    prevalence = np.mean(y_true)
    
    results = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        if thresh < 1.0:
            odds = thresh / (1 - thresh)
            net_benefit_model = (tp / n) - (fp / n) * odds
            net_benefit_treat_all = prevalence - (1 - prevalence) * odds
        else:
            net_benefit_model = 0
            net_benefit_treat_all = 0
        
        results.append({
            'threshold': thresh,
            'model_name': model_name,
            'net_benefit': net_benefit_model,
            'net_benefit_treat_all': net_benefit_treat_all,
            'net_benefit_treat_none': 0
        })
    
    return pd.DataFrame(results)


def decision_curve_analysis_multi(y_true, model_probs, thresholds=None):
    """
    Perform Decision Curve Analysis for multiple models.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    model_probs : dict
        Dictionary mapping model names to predicted probabilities
        e.g., {'Model A': probs_a, 'Model B': probs_b}
    thresholds : array-like, optional
        Threshold probabilities to evaluate
        
    Returns:
    --------
    pd.DataFrame : Combined DataFrame with net benefits for all models
    """
    all_results = []
    
    for model_name, y_prob in model_probs.items():
        dca_df = decision_curve_analysis(y_true, y_prob, thresholds, model_name)
        all_results.append(dca_df)
    
    return pd.concat(all_results, ignore_index=True)


def plot_decision_curve(dca_results, ax=None, show_treat_all=True, show_treat_none=True,
                        xlim=(0, 0.5), ylim=None, colors=None, title='Decision Curve Analysis'):
    """
    Plot Decision Curve Analysis results for one or multiple models.
    
    Parameters:
    -----------
    dca_results : pd.DataFrame
        Output from decision_curve_analysis() or decision_curve_analysis_multi()
        Must contain columns: 'threshold', 'net_benefit', 'model_name'
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    show_treat_all : bool
        Whether to show the "Treat All" reference line
    show_treat_none : bool
        Whether to show the "Treat None" reference line
    xlim : tuple
        X-axis limits (threshold probability range)
    ylim : tuple, optional
        Y-axis limits. If None, auto-determined.
    colors : dict, optional
        Dictionary mapping model names to colors
    title : str
        Plot title
        
    Returns:
    --------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique model names
    model_names = dca_results['model_name'].unique()
    
    # Default color palette
    if colors is None:
        default_colors = plt.cm.tab10.colors
        colors = {name: default_colors[i % len(default_colors)] 
                  for i, name in enumerate(model_names)}
    
    # Plot each model
    for model_name in model_names:
        model_data = dca_results[dca_results['model_name'] == model_name]
        ax.plot(model_data['threshold'], model_data['net_benefit'],
                label=model_name, linewidth=2, color=colors.get(model_name))
    
    # Plot reference strategies (use first model's data for treat all)
    first_model = dca_results[dca_results['model_name'] == model_names[0]]
    
    if show_treat_all:
        ax.plot(first_model['threshold'], first_model['net_benefit_treat_all'],
                label='Treat All', linestyle='--', color='gray', linewidth=1.5)
    
    if show_treat_none:
        ax.axhline(y=0, label='Treat None', linestyle=':', color='black', linewidth=1.5)
    
    ax.set_xlabel('Threshold Probability', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim(xlim)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.grid(True, alpha=0.3)
    
    return ax


def calculate_dca_from_paths(filepath, model_name='Model', thresholds=None):
    """
    Calculate Decision Curve Analysis across all folds from a model directory.
    
    Mirrors R's decision_curve_analysis and summarize_dca_across_folds functions.
    Loads test_probas_* and test_true_labels_* files from the directory,
    calculates DCA for each fold, and returns both individual and aggregated results.
    
    Parameters:
    -----------
    filepath : str
        Path to directory containing model results.
        Must contain test_probas_*.pkl and test_true_labels_*.pkl files.
    model_name : str
        Name identifier for the model.
    thresholds : array-like, optional
        Threshold probabilities to evaluate. Default is np.arange(0.01, 0.99, 0.01)
        matching R's seq(0.01, 0.99, by = 0.01)
        
    Returns:
    --------
    tuple : (all_folds_df, summary_df)
        all_folds_df: DataFrame with DCA for all folds, with columns:
            threshold, model_name, net_benefit, net_benefit_treat_all, net_benefit_treat_none, fold
        summary_df: DataFrame with mean, SD, and SE across folds for each threshold, 
            matching R's summarize_dca_across_folds output:
            threshold, model_name, mean_net_benefit, sd_net_benefit, se_net_benefit,
            mean_nb_all, sd_nb_all, se_nb_all
    """
    import glob
    import re
    
    if thresholds is None:
        thresholds = np.arange(0.01, 1, 0.01)  # Match R's seq(0.01, 0.99, by = 0.01)
    
    # Find all test_probas files
    probas_files = sorted(glob.glob(f"{filepath}/test_probas_*.pkl"))
    
    if not probas_files:
        raise FileNotFoundError(f"No test_probas_*.pkl files found in {filepath}")
    
    # Extract fold identifiers
    def get_fold_id(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'test_probas_(.+)\.pkl', basename)
        return match.group(1) if match else basename
    
    all_folds_results = []
    
    for probas_file in probas_files:
        fold_id = get_fold_id(probas_file)
        
        # Load probabilities
        probas = pickle.load(open(probas_file, 'rb'))
        
        # Handle nested arrays
        if hasattr(probas, '__len__') and len(probas) > 0:
            if isinstance(probas[0], (list, np.ndarray)) and len(probas) == 1:
                probas = probas[0]
        probas = np.asarray(probas)
        
        # Load labels
        labels_file = probas_file.replace('test_probas_', 'test_true_labels_')
        if not os.path.exists(labels_file):
            print(f"Warning: Labels file not found: {labels_file}, skipping fold {fold_id}")
            continue
        
        labels = pickle.load(open(labels_file, 'rb'))
        if hasattr(labels, '__len__') and len(labels) > 0:
            if isinstance(labels[0], (list, np.ndarray)) and len(labels) == 1:
                labels = labels[0]
        labels = np.asarray(labels)
        
        # Calculate DCA for this fold
        dca_df = decision_curve_analysis(labels, probas, thresholds, model_name)
        dca_df['fold'] = fold_id
        all_folds_results.append(dca_df)
    
    all_folds_df = pd.concat(all_folds_results, ignore_index=True)
    
    # Calculate summary statistics across folds - matching R's summarize_dca_across_folds
    n_folds = len(all_folds_df['fold'].unique())
    
    summary_df = all_folds_df.groupby(['threshold', 'model_name']).agg({
        'net_benefit': ['mean', 'std'],
        'net_benefit_treat_all': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names and rename to match R output
    summary_df.columns = ['threshold', 'model_name', 
                          'mean_net_benefit', 'sd_net_benefit',
                          'mean_nb_all', 'sd_nb_all']
    
    # Add standard error columns (matching R: sd / sqrt(n))
    summary_df['se_net_benefit'] = summary_df['sd_net_benefit'] / np.sqrt(n_folds)
    summary_df['se_nb_all'] = summary_df['sd_nb_all'] / np.sqrt(n_folds)
    
    # Reorder columns to match R output order
    summary_df = summary_df[['threshold', 'model_name', 
                             'mean_net_benefit', 'sd_net_benefit', 'se_net_benefit',
                             'mean_nb_all', 'sd_nb_all', 'se_nb_all']]
    
    return all_folds_df, summary_df


def plot_decision_curves_from_paths(
    model_paths,
    model_names=None,
    thresholds=None,
    xlim='auto',
    ylim='auto',
    figsize=(10, 6),
    show_treat_all=True,
    show_treat_none=True,
    show_ribbon=True,
    save_path=None,
    title='Decision Curve Analysis',
    prob_percentile=95
):
    """
    Plot Decision Curves for multiple models from file paths.
    
    Loads test_probas_* and test_true_labels_* files from each model directory,
    calculates DCA across all folds, and plots mean net benefit with SD ribbons.
    
    Parameters:
    -----------
    model_paths : list or dict
        If list: List of directory paths containing model results.
        If dict: Dictionary mapping model names to directory paths.
    model_names : list, optional
        Names for each model (required if model_paths is a list).
    thresholds : array-like, optional
        Threshold probabilities to evaluate. Default is np.arange(0.01, 1, 0.01)
    xlim : tuple or 'auto', default='auto'
        X-axis limits. If 'auto', sets upper limit to the prob_percentile of max 
        predicted probabilities across all models/folds.
    ylim : tuple or 'auto', default='auto'  
        Y-axis limits. If 'auto', sets limits based on model net benefit range
        (excluding extreme Treat All values at high thresholds).
    figsize : tuple
        Figure size (width, height).
    show_treat_all : bool
        Whether to show the "Treat All" reference line.
    show_treat_none : bool
        Whether to show the "Treat None" reference line.
    show_ribbon : bool
        Whether to show SD ribbons around the mean lines.
    save_path : str, optional
        Path to save the figure. If None, displays the plot.
    title : str
        Plot title.
    prob_percentile : int, default=95
        Percentile of predicted probabilities to use for auto xlim upper bound.
        Use 95 or 99 to focus on clinically relevant thresholds.
        
    Returns:
    --------
    tuple : (fig, ax, summary_data)
        summary_data: dict mapping model names to their summary DataFrames
    """
    import glob
    import re
    
    if thresholds is None:
        thresholds = np.arange(0.01, 1, 0.01)
    
    # Handle input formats
    if isinstance(model_paths, dict):
        paths_dict = model_paths
    elif isinstance(model_paths, list):
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(model_paths))]
        paths_dict = dict(zip(model_names, model_paths))
    else:
        raise ValueError("model_paths must be a list or dict")
    
    # Collect all probabilities for auto-limits calculation
    all_probas = []
    
    # Calculate DCA for each model
    summary_data = {}
    all_summaries = []
    all_folds_data = {}
    
    for model_name, filepath in paths_dict.items():
        # Also collect probabilities for auto xlim
        probas_files = sorted(glob.glob(f"{filepath}/test_probas_*.pkl"))
        for probas_file in probas_files:
            probas = pickle.load(open(probas_file, 'rb'))
            if hasattr(probas, '__len__') and len(probas) > 0:
                if isinstance(probas[0], (list, np.ndarray)) and len(probas) == 1:
                    probas = probas[0]
            all_probas.extend(np.asarray(probas).flatten())
        
        all_folds_df, summary_df = calculate_dca_from_paths(filepath, model_name, thresholds)
        summary_data[model_name] = summary_df
        all_folds_data[model_name] = all_folds_df
        all_summaries.append(summary_df)
    
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    
    # Calculate auto xlim based on predicted probability distribution
    if xlim == 'auto':
        all_probas = np.array(all_probas)
        max_prob = np.percentile(all_probas, prob_percentile)
        # Round up to nearest 0.05 for cleaner axis
        x_upper = np.ceil(max_prob * 20) / 20
        x_upper = min(x_upper, 1.0)  # Cap at 1.0
        x_upper = max(x_upper, 0.1)  # At least 0.1
        xlim = (0, x_upper)
    
    # Calculate auto ylim based on net benefit range within xlim
    if ylim == 'auto':
        # Filter to thresholds within xlim
        mask = combined_summary['threshold'] <= xlim[1]
        filtered_data = combined_summary[mask]
        
        # Get min/max of model net benefits (not treat all to avoid extreme values)
        y_min = filtered_data['mean_net_benefit'].min()
        y_max = filtered_data['mean_net_benefit'].max()
        
        # Include treat all line if showing it
        if show_treat_all:
            y_min = min(y_min, filtered_data['mean_nb_all'].min())
        
        # Include treat none (0) if showing
        if show_treat_none:
            y_min = min(y_min, 0)
            y_max = max(y_max, 0)
        
        # Add padding
        y_range = y_max - y_min
        padding = y_range * 0.1
        ylim = (y_min - padding, y_max + padding)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    colors = plt.cm.tab10.colors
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(paths_dict.keys())}
    
    # Plot each model
    for model_name in paths_dict.keys():
        model_data = combined_summary[combined_summary['model_name'] == model_name]
        color = color_map[model_name]
        
        # Plot mean line
        ax.plot(model_data['threshold'], model_data['mean_net_benefit'],
                label=model_name, linewidth=2, color=color)
        
        # Plot SD ribbon
        if show_ribbon:
            lower = model_data['mean_net_benefit'] - model_data['sd_net_benefit']
            upper = model_data['mean_net_benefit'] + model_data['sd_net_benefit']
            ax.fill_between(model_data['threshold'], lower, upper, 
                           alpha=0.2, color=color)
    
    # Plot reference strategies
    if show_treat_all:
        first_model = combined_summary[combined_summary['model_name'] == list(paths_dict.keys())[0]]
        ax.plot(first_model['threshold'], first_model['mean_nb_all'],
                label='Treat All', linestyle='--', color='gray', linewidth=1.5)
    
    if show_treat_none:
        ax.axhline(y=0, label='Treat None', linestyle=':', color='black', linewidth=1.5)
    
    # Customize plot
    ax.set_xlabel('Threshold Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Net Benefit', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        fig.savefig(save_path, facecolor='white', transparent=False, dpi=300)
        plt.close()
    else:
        plt.show()
    
    return fig, ax, summary_data
