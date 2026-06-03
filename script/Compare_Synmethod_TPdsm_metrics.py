
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, matthews_corrcoef, accuracy_score,
    precision_score, recall_score
)
from scipy.stats import norm, bootstrap
import os
import sys
import json
from pathlib import Path
from joblib import load
import seaborn as sns
import warnings


warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics')


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']



COMPARE_LIST = [
    "TPdsm_score",
    "SyMetrics",
    "CADD_RawScore",
    "DANN",
    "DDIG",
    "eigen",
    "EnDSM",
    "fathmm_MKL_coding",
    "fathmm_xf_coding",
    "frDSM",
    "PhD_SNPg",
    "PrDSM",
    "silva",
    "syntool",
    "usDSM"
]


MODEL_DISPLAY_NAMES = {
    "TPdsm_score": "TPdsm",
    "SyMetrics": "SyMetrics",
    "CADD_RawScore": "CADD",
    "DANN": "DANN",
    "DDIG": "DDIG",
    "eigen": "Eigen",
    "EnDSM": "EnDSM",
    "fathmm_MKL_coding": "Fathmm_MKL",
    "fathmm_xf_coding": "Fathmm_XF",
    "frDSM": "frDSM",
    "PhD_SNPg": "PhD_SNPg",
    "PrDSM": "PrDSM",
    "silva": "SilVA",
    "syntool": "Syntool",
    "usDSM": "usDSM"
}


THRESHOLDS = {
    "TPdsm_score": 0.055,
    "SyMetrics": 0.875,
    "CADD_RawScore": 0.4,
    "DANN": 0.7,
    "DDIG": 0.5,
    "eigen": 0.4,
    "EnDSM": 0.5,
    "fathmm_MKL_coding": 0.5,
    "fathmm_xf_coding": 0.97,
    "frDSM": 0.5,
    "PhD_SNPg": 0.45,
    "PrDSM": 0.5,
    "silva": 0.27,
    "syntool": 0,
    "usDSM": 0.5
}


TEST_DATASETS = {
    'test1': {
        'file': 'testset1',
        'display_name': 'Test Set 1'
    },
    'test2': {
        'file': 'testset2',
        'display_name': 'Test Set 2'
    },
    'test3': {
        'file': 'testset3',
        'display_name': 'Test Set 3'
    },
    'test4': {
        'file': 'testset4',
        'display_name': 'Test Set 4'
    }
}


model_dir = Path("/path/to/models")


print("Loading model features config...")
with open(model_dir / 'feature_380.json', 'r', encoding='utf-8') as f:
    model_features = json.load(f)
print(f"  Loaded {len(model_features)} model features")


print("Loading prediction model...")
model = load(model_dir / 'model.pkl')
print("  Model loaded successfully")


trainFeatures = [
    "gnomad41_exome_faf99", "delta_score", "silva_rankscore", "silva",
    "cadd_mapability_20bp", "delta_psi_max", "delta_score_rankscore",
    "gnomad41_exome_AF_eas", "CADD_PHRED", "gnomad41_exome_AF_asj",
    "#RSCU", "syntool_rankscore", "MES-KM?",
    "gnomad41_genome_fafmax_faf99_max", "ExAC_FIN", "gerp_gt2",
    "#MES", "CpG_exon", "ExAC_OTH", "gnomad41_genome_AF_eas",
    "gnomad41_exome_fafmax_faf99_max", "SR+"
]


population_freq_features = [
    "gnomad41_exome_faf99", "gnomad41_exome_AF_eas",
    "gnomad41_genome_fafmax_faf99_max", "ExAC_FIN",
    "gnomad41_genome_AF_eas", "gnomad41_exome_fafmax_faf99_max",
    "ExAC_OTH"
]


Condidate_features = list(set(trainFeatures + COMPARE_LIST))


def input_mean_overall(input_data, input_c, population_freq_features=None):
    
    if population_freq_features is None:
        population_freq_features = []
    df = input_data.copy()
    for col_idx in input_c:
        col_name = col_idx
        if col_name not in df.columns:
            continue  
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        if col_name in population_freq_features:
            df[col_name].fillna(0, inplace=True)
        else:
            mean_value = df[col_name].mean(skipna=True)
            df[col_name].fillna(mean_value, inplace=True)
        df[col_name] = df[col_name].astype(float)
    return df


def predict(model, data, features):
    
    model_pred = model.predict_proba(data[features])
    return model_pred[:, 1]




def get_valid_indices(y_pred):
    
    if y_pred.dtype == 'object':
        valid_indices = ~(y_pred == ".")
    elif np.issubdtype(y_pred.dtype, np.floating):
        valid_indices = ~np.isnan(y_pred)
    else:
        valid_indices = np.ones_like(y_pred, dtype=bool)
    return valid_indices


def compute_midrank_weight(x, sample_weight):
   
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T_tmp = np.empty(N, dtype=np.float64)
    T_tmp[J] = T
    return T_tmp


def fast_delong(predictions_sorted_transposed, label_1_count, sample_weight=None):
    
    if sample_weight is None:
        sample_weight = np.ones(predictions_sorted_transposed.shape[1])
    
    n_samples = predictions_sorted_transposed.shape[1]
    n_predictors = predictions_sorted_transposed.shape[0]
    
    m = label_1_count
    n = n_samples - m
    
    positive_indices = np.arange(m)
    negative_indices = np.arange(m, n_samples)
    
    m_predictions = predictions_sorted_transposed[:, positive_indices]
    n_predictions = predictions_sorted_transposed[:, negative_indices]
    
    m_weights = sample_weight[positive_indices]
    n_weights = sample_weight[negative_indices]
    
    m_sum_weights = np.sum(m_weights)
    n_sum_weights = np.sum(n_weights)
    
    V10 = np.zeros((n_predictors, m), dtype=np.float64)
    V01 = np.zeros((n_predictors, n), dtype=np.float64)
    
    for i in range(n_predictors):
        V10[i, :] = compute_midrank_weight(
            np.concatenate([m_predictions[i, :], n_predictions[i, :]]),
            np.concatenate([m_weights, n_weights])
        )[:m] - compute_midrank_weight(m_predictions[i, :], m_weights)
        
        V01[i, :] = compute_midrank_weight(
            np.concatenate([n_predictions[i, :], m_predictions[i, :]]),
            np.concatenate([n_weights, m_weights])
        )[:n] - compute_midrank_weight(n_predictions[i, :], n_weights)
    
    V10 = V10 / n_sum_weights
    V01 = V01 / m_sum_weights
    
    S10 = np.zeros((n_predictors, n_predictors), dtype=np.float64)
    S01 = np.zeros((n_predictors, n_predictors), dtype=np.float64)
    
    for i in range(n_predictors):
        for j in range(n_predictors):
            if m > 1:
                S10[i, j] = np.sum((V10[i, :] - np.mean(V10[i, :])) * 
                                  (V10[j, :] - np.mean(V10[j, :]))) / (m - 1)
            else:
                S10[i, j] = 0.0
            if n > 1:
                S01[i, j] = np.sum((V01[i, :] - np.mean(V01[i, :])) * 
                                  (V01[j, :] - np.mean(V01[j, :]))) / (n - 1)
            else:
                S01[i, j] = 0.0
    
    covariance = S10 / m + S01 / n
    
    auc_values = np.zeros(n_predictors, dtype=np.float64)
    for i in range(n_predictors):
        auc_values[i] = np.mean(V10[i, :])
    
    return auc_values, covariance




def bca_bootstrap_ci(y_true, y_pred, metric_func, n_resamples=2000, 
                     confidence_level=0.95, random_state=42):
    
    np.random.seed(random_state)
    
    valid_idx = get_valid_indices(y_pred)
    y_true = y_true[valid_idx]
    y_pred = y_pred[valid_idx]
    
    n_samples = len(y_true)
    if n_samples == 0:
        return np.nan, np.nan, np.nan
    
    
    point_estimate = metric_func(y_true, y_pred)
    
    
    try:
        print(f"    [DEBUG] Running scipy.stats.bootstrap(method='BCa', n_resamples={n_resamples})...")
        result = bootstrap(
            (y_true, y_pred),
            statistic=lambda y_true_boot, y_pred_boot: metric_func(y_true_boot, y_pred_boot),
            confidence_level=confidence_level,
            method='BCa',
            n_resamples=n_resamples,
            random_state=random_state
        )
        ci_lower = result.confidence_interval.low
        ci_upper = result.confidence_interval.high
        print(f"    [DEBUG] BCa Bootstrap SUCCESS: CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
    except TypeError as e:
        
        if 'random_state' in str(e):
            print(f"  [Warning] scipy version doesn't support random_state, using manual seeding")
            np.random.seed(random_state)
            result = bootstrap(
                (y_true, y_pred),
                statistic=lambda y_true_boot, y_pred_boot: metric_func(y_true_boot, y_pred_boot),
                confidence_level=confidence_level,
                method='BCa',
                n_resamples=n_resamples
            )
            ci_lower = result.confidence_interval.low
            ci_upper = result.confidence_interval.high
            print(f"    [DEBUG] BCa Bootstrap SUCCESS (no random_state): CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
        else:
            raise
    except Exception as e:
        print(f"  [Warning] BCa Bootstrap failed: {e}, falling back to percentile")
        
        boot_stats = []
        for _ in range(n_resamples):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            if len(np.unique(y_true[idx])) < 2:
                continue
            try:
                boot_stats.append(metric_func(y_true[idx], y_pred[idx]))
            except:
                continue
        
        if len(boot_stats) > 0:
            alpha = (1 - confidence_level) * 100
            ci_lower = np.percentile(boot_stats, alpha / 2)
            ci_upper = np.percentile(boot_stats, 100 - alpha / 2)
            print(f"    [DEBUG] Percentile Bootstrap FALLBACK: CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
        else:
            ci_lower, ci_upper = np.nan, np.nan
            print(f"    [DEBUG] Bootstrap FAILED: No valid samples")
    
    return point_estimate, ci_lower, ci_upper


def bootstrap_comparison_test(y_true, y_pred1, y_pred2, metric_func, 
                              n_resamples=2000, random_state=42):
    
    np.random.seed(random_state)
    
    valid_idx = get_valid_indices(y_pred1) & get_valid_indices(y_pred2)
    y_true = y_true[valid_idx]
    y_pred1 = y_pred1[valid_idx]
    y_pred2 = y_pred2[valid_idx]
    
    n_samples = len(y_true)
    if n_samples == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    
    metric1 = metric_func(y_true, y_pred1)
    metric2 = metric_func(y_true, y_pred2)
    
    
    diff_bootstrap = []
    for _ in range(n_resamples):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            m1 = metric_func(y_true[idx], y_pred1[idx])
            m2 = metric_func(y_true[idx], y_pred2[idx])
            diff_bootstrap.append(m1 - m2)
        except:
            continue
    
    if len(diff_bootstrap) == 0:
        return metric1, metric2, np.nan, np.nan, np.nan
    
    diff_bootstrap = np.array(diff_bootstrap)
    
   
    ci_lower = np.percentile(diff_bootstrap, 2.5)
    ci_upper = np.percentile(diff_bootstrap, 97.5)
    

    p_value = 2 * min(
        np.mean(diff_bootstrap >= 0),
        np.mean(diff_bootstrap <= 0)
    )
    p_value = min(p_value, 1.0)
    
    return metric1, metric2, ci_lower, ci_upper, p_value




def compute_auc_ci_adaptive(y_true, y_pred, reference_pred=None, reference_model_name=None):
    
    from sklearn.metrics import roc_auc_score
    
    valid_idx = get_valid_indices(y_pred)
    y_true_valid = y_true[valid_idx]
    y_pred_valid = y_pred[valid_idx]
    
    n = len(y_true_valid) 
    pos_rate = np.mean(y_true_valid == 1)
    
    auc_score = roc_auc_score(y_true_valid, y_pred_valid)
    
    if n >= 500:
        
        method_name = "DeLong"
        print(f"  [DEBUG] AUC   -> Method: DeLong | n={n}, pos_rate={pos_rate:.3f}, n_pos={np.sum(y_true_valid == 1)}, n_neg={np.sum(y_true_valid == 0)}")
        n_pos = np.sum(y_true_valid == 1)
        n_neg = np.sum(y_true_valid == 0)
        
        if n_pos > 0 and n_neg > 0:
            order = np.argsort(y_true_valid)[::-1]
            predictions_sorted = y_pred_valid[order].reshape(1, -1)
            _, covariance = fast_delong(predictions_sorted, n_pos)
            variance = covariance[0, 0]
            
            if variance > 0:
                se = np.sqrt(variance)
                ci_lower = max(0.0, auc_score - 1.96 * se)
                ci_upper = min(1.0, auc_score + 1.96 * se)
            else:
                ci_lower, ci_upper = np.nan, np.nan
        else:
            ci_lower, ci_upper = np.nan, np.nan
        
        
        p_value = np.nan
        z_stat = np.nan
        if reference_pred is not None:
            ref_valid = ~np.isnan(reference_pred)
            common_valid = valid_idx & ref_valid
            
            y_true_common = y_true[common_valid]
            y_pred_common = y_pred[common_valid]
            y_ref_common = reference_pred[common_valid]
            
            n_pos_common = np.sum(y_true_common == 1)
            if n_pos_common > 0:
                order = np.argsort(y_true_common)[::-1]
                predictions_sorted = np.vstack([y_pred_common, y_ref_common])[:, order]
                
                auc_values, covariance = fast_delong(predictions_sorted, n_pos_common)
                var1 = covariance[0, 0]
                var2 = covariance[1, 1]
                cov = covariance[0, 1]
                
                var_diff = var1 + var2 - 2 * cov
                if var_diff > 0:
                    z_stat = (auc_values[0] - auc_values[1]) / np.sqrt(var_diff)
                    p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))
        
        return {
            'auc': auc_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'z_stat': z_stat,
            'method': method_name,
            'n_samples': n,
            'positive_rate': pos_rate
        }
    
    elif n >= 50:
        
        n_resamples = 5000
        method_name = f"BCa Bootstrap ({n_resamples})"
        print(f"  [DEBUG] AUC   -> Method: BCa Bootstrap ({n_resamples}) | n={n}, pos_rate={pos_rate:.3f}")
        
        ci_lower, ci_upper = bca_bootstrap_ci(
            y_true_valid, y_pred_valid, roc_auc_score, 
            n_resamples=n_resamples
        )[1:]
        
       
        p_value = np.nan
        if reference_pred is not None:
            ref_valid = ~np.isnan(reference_pred)
            common_valid = valid_idx & ref_valid
            
            y_true_common = y_true[common_valid]
            y_pred_common = y_pred[common_valid]
            y_ref_common = reference_pred[common_valid]
            
            _, _, _, _, p_value = bootstrap_comparison_test(
                y_true_common, y_pred_common, y_ref_common,
                roc_auc_score, n_resamples=n_resamples
            )
        
        return {
            'auc': auc_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'z_stat': np.nan,
            'method': method_name,
            'n_samples': n,
            'positive_rate': pos_rate
        }
    
    else:
        
        return {
            'auc': auc_score,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'p_value': np.nan,
            'z_stat': np.nan,
            'method': "Not applicable (n<50)",
            'n_samples': n,
            'positive_rate': pos_rate
        }


def compute_auprc_ci_adaptive(y_true, y_pred, reference_pred=None, reference_model_name=None):
    
    valid_idx = get_valid_indices(y_pred)
    y_true_valid = y_true[valid_idx]
    y_pred_valid = y_pred[valid_idx]
    
    n = len(y_true_valid)  
    pos_rate = np.mean(y_true_valid == 1)
    is_imbalanced = (pos_rate < 0.3) or (pos_rate > 0.7)
    
    auprc_score = average_precision_score(y_true_valid, y_pred_valid)
    
    
    if is_imbalanced:
        if n >= 500:
            n_resamples = 2000
        elif n >= 100:
            n_resamples = 3000
        elif n >= 50:
            n_resamples = 5000
        else:
            n_resamples = 0
    else:
        if n >= 500:
            n_resamples = 1000
        elif n >= 100:
            n_resamples = 2000
        elif n >= 50:
            n_resamples = 5000
        else:
            n_resamples = 0
    
    if n_resamples == 0:
        return {
            'auprc': auprc_score,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'p_value': np.nan,
            'method': "Not applicable (n<50)",
            'n_samples': n,
            'positive_rate': pos_rate
        }
    
    
    if is_imbalanced or n < 500:
        method_name = "BCa"
        print(f"  [DEBUG] AUPRC -> Method: BCa Bootstrap | n={n}, pos_rate={pos_rate:.3f}, is_imbalanced={is_imbalanced}, n_resamples={n_resamples}")
        _, ci_lower, ci_upper = bca_bootstrap_ci(
            y_true_valid, y_pred_valid, average_precision_score,
            n_resamples=n_resamples
        )
    else:
        
        method_name = "Percentile"
        print(f"  [DEBUG] AUPRC -> Method: Percentile Bootstrap | n={n}, pos_rate={pos_rate:.3f}, is_imbalanced={is_imbalanced}, n_resamples={n_resamples}")
        boot_stats = []
        np.random.seed(42)
        for _ in range(n_resamples):
            idx = np.random.choice(n, n, replace=True)
            if len(np.unique(y_true_valid[idx])) < 2:
                continue
            try:
                boot_stats.append(average_precision_score(y_true_valid[idx], y_pred_valid[idx]))
            except:
                continue
        
        if len(boot_stats) > 0:
            ci_lower = np.percentile(boot_stats, 2.5)
            ci_upper = np.percentile(boot_stats, 97.5)
        else:
            ci_lower, ci_upper = np.nan, np.nan
    
    method_str = f"{method_name} Bootstrap ({n_resamples})"
    
    
    p_value = np.nan
    if reference_pred is not None:
        ref_valid = ~np.isnan(reference_pred)
        common_valid = valid_idx & ref_valid
        
        y_true_common = y_true[common_valid]
        y_pred_common = y_pred[common_valid]
        y_ref_common = reference_pred[common_valid]
        
        _, _, _, _, p_value = bootstrap_comparison_test(
            y_true_common, y_pred_common, y_ref_common,
            average_precision_score, n_resamples=n_resamples
        )
    
    return {
        'auprc': auprc_score,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'method': method_str,
        'n_samples': n,
        'positive_rate': pos_rate
    }



def calculate_metrics_with_ci(y_true, y_pred, threshold, model_name, n_bootstrap=1000, random_state=42):
    
    np.random.seed(random_state)
    
    valid_idx = get_valid_indices(y_pred)
    y_true = y_true[valid_idx]
    y_pred = y_pred[valid_idx]
    
    n = len(y_true)
    if n == 0:
        return {
            'f1': np.nan, 'mcc': np.nan, 'accuracy': np.nan,
            'precision': np.nan, 'sensitivity': np.nan, 'specificity': np.nan,
            'ci_lower': {}, 'ci_upper': {}
        }
    
    
    if model_name == "syntool":
        y_pred_binary = (y_pred < threshold).astype(int)
    else:
        y_pred_binary = (y_pred >= threshold).astype(int)
    
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    
    f1 = f1_score(y_true, y_pred_binary)
    mcc = matthews_corrcoef(y_true, y_pred_binary)
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    sensitivity = recall_score(y_true, y_pred_binary)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
   
    metric_names = ['f1', 'mcc', 'accuracy', 'precision', 'sensitivity', 'specificity']
    bootstrap_results = {name: [] for name in metric_names}
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]
        
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        if model_name == "syntool":
            y_pred_binary_boot = (y_pred_boot < threshold).astype(int)
        else:
            y_pred_binary_boot = (y_pred_boot >= threshold).astype(int)
        
        try:
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_boot, y_pred_binary_boot).ravel()
            
            bootstrap_results['f1'].append(f1_score(y_true_boot, y_pred_binary_boot))
            bootstrap_results['mcc'].append(matthews_corrcoef(y_true_boot, y_pred_binary_boot))
            bootstrap_results['accuracy'].append(accuracy_score(y_true_boot, y_pred_binary_boot))
            bootstrap_results['precision'].append(precision_score(y_true_boot, y_pred_binary_boot))
            bootstrap_results['sensitivity'].append(recall_score(y_true_boot, y_pred_binary_boot))
            bootstrap_results['specificity'].append(
                tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0
            )
        except:
            continue
    
    ci_lower = {}
    ci_upper = {}
    for name in metric_names:
        if len(bootstrap_results[name]) > 0:
            ci_lower[name] = np.percentile(bootstrap_results[name], 2.5)
            ci_upper[name] = np.percentile(bootstrap_results[name], 97.5)
        else:
            ci_lower[name] = np.nan
            ci_upper[name] = np.nan
    
    return {
        'f1': f1, 'mcc': mcc, 'accuracy': accuracy,
        'precision': precision, 'sensitivity': sensitivity, 'specificity': specificity,
        'ci_lower': ci_lower, 'ci_upper': ci_upper
    }



def benjamini_hochberg_fdr(p_values):
    
    p_values = np.asarray(p_values, dtype=np.float64)
    n = len(p_values)
    fdr_p_values = np.full(n, np.nan)
    
    valid_mask = ~np.isnan(p_values)
    valid_p = p_values[valid_mask]
    n_valid = len(valid_p)
    
    if n_valid == 0:
        return fdr_p_values
    
    sorted_indices = np.argsort(valid_p)
    sorted_p = valid_p[sorted_indices]
    
    adjusted = np.zeros(n_valid)
    for i in range(n_valid - 1, -1, -1):
        if i == n_valid - 1:
            adjusted[i] = sorted_p[i]
        else:
            adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n_valid / (i + 1))
    
    adjusted = np.minimum(adjusted, 1.0)
    
    original_order_adjusted = np.zeros(n_valid)
    original_order_adjusted[sorted_indices] = adjusted
    
    fdr_p_values[valid_mask] = original_order_adjusted
    
    return fdr_p_values




def pltRoc_with_ci(y, pred_y, tag, color, ci_result=None):
    
    pred_y = pd.to_numeric(pred_y, errors='coerce').astype(float)
    y = pd.to_numeric(y, errors='coerce').astype(float)
    
    valid_idx = get_valid_indices(pred_y)
    y_valid = y[valid_idx]
    pred_valid = pred_y[valid_idx]
    
    fpr, tpr, _ = roc_curve(y_valid, pred_valid, pos_label=1, drop_intermediate=False)
    auc_val = auc(fpr, tpr)
    
    if ci_result:
        ci_lower = ci_result.get('ci_lower', np.nan)
        ci_upper = ci_result.get('ci_upper', np.nan)
        p_val = ci_result.get('p_value_fdr', ci_result.get('p_value', np.nan))
        
        if not np.isnan(ci_lower) and not np.isnan(ci_upper):
            auc_str = f"{auc_val:.3f}[{ci_lower:.3f},{ci_upper:.3f}]"
        else:
            auc_str = f"{auc_val:.3f}"
        
        if not np.isnan(p_val):
            if p_val < 0.001:
                p_text = 'p<0.001'
            else:
                p_text = f'p={p_val:.3f}'
            label = f'{MODEL_DISPLAY_NAMES.get(tag, tag)} (AUC={auc_str}, {p_text})'
        else:
            label = f'{MODEL_DISPLAY_NAMES.get(tag, tag)} (AUC={auc_str})'
    else:
        label = f'{MODEL_DISPLAY_NAMES.get(tag, tag)} (AUC={auc_val:.3f})'
    
    plt.plot(fpr, tpr, color=color, lw=2, label=label)


def pltPrc_with_ci(y, pred_y, tag, color, ci_result=None):
    
    pred_y = pd.to_numeric(pred_y, errors='coerce').astype(float)
    y = pd.to_numeric(y, errors='coerce').astype(float)
    
    valid_idx = get_valid_indices(pred_y)
    y_valid = y[valid_idx]
    pred_valid = pred_y[valid_idx]
    
    precision, recall, _ = precision_recall_curve(y_valid, pred_valid)
    auprc_val = average_precision_score(y_valid, pred_valid)
    
    if ci_result:
        ci_lower = ci_result.get('ci_lower', np.nan)
        ci_upper = ci_result.get('ci_upper', np.nan)
        p_val = ci_result.get('p_value_fdr', ci_result.get('p_value', np.nan))
        
        if not np.isnan(ci_lower) and not np.isnan(ci_upper):
            auprc_str = f"{auprc_val:.3f}[{ci_lower:.3f},{ci_upper:.3f}]"
        else:
            auprc_str = f"{auprc_val:.3f}"
        
        if not np.isnan(p_val):
            if p_val < 0.001:
                p_text = 'p<0.001'
            else:
                p_text = f'p={p_val:.3f}'
            label = f'{MODEL_DISPLAY_NAMES.get(tag, tag)} (AUPRC={auprc_str}, {p_text})'
        else:
            label = f'{MODEL_DISPLAY_NAMES.get(tag, tag)} (AUPRC={auprc_str})'
    else:
        label = f'{MODEL_DISPLAY_NAMES.get(tag, tag)} (AUPRC={auprc_val:.3f})'
    
    plt.step(recall, precision, lw=2, color=color, label=label, where='post')


def plot_roc_curves(test_data, compare_list, colors, output_file, auc_results):
    
    plt.figure(figsize=(10, 10.5))
    
    for idx, model_name in enumerate(compare_list):
        if model_name in test_data.columns:
            y_true = test_data['Otherinfo1'].values
            y_pred = test_data[model_name].values
            
            ci_res = auc_results.get(model_name, None)
            pltRoc_with_ci(y_true, y_pred, model_name, colors[idx], ci_res)
    
    plt.plot([0, 1], [0, 1], color='silver', lw=2, linestyle='--')
    plt.xlim([0.0, 1.00])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate', size=8, color='black')
    plt.ylabel('True positive rate', size=8, color='black')
    plt.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.xticks(size=8, color='black')
    plt.yticks(size=8, color='black')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prc_curves(test_data, compare_list, colors, output_file, auprc_results):
    
    plt.figure(figsize=(10, 10.5))
    
    for idx, model_name in enumerate(compare_list):
        if model_name in test_data.columns:
            y_true = test_data['Otherinfo1'].values
            y_pred = test_data[model_name].values
            
            ci_res = auprc_results.get(model_name, None)
            pltPrc_with_ci(y_true, y_pred, model_name, colors[idx], ci_res)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', size=8, color='black')
    plt.ylabel('Precision', size=8, color='black')
    plt.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.xticks(size=8, color='black')
    plt.yticks(size=8, color='black')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()



def process_test_dataset(test_name, test_data, compare_list, reference_model='TPdsm_score'):
   
    print(f"\n{'='*60}")
    print(f"Processing: {test_name}")
    print(f"{'='*60}")
    
    
    y_true_raw = test_data['Otherinfo1'].values
    
    
    if y_true_raw.dtype == 'object':
        y_true = np.array([1 if x == 'P' else (0 if x == 'B' else np.nan) for x in y_true_raw])
    else:
        y_true = y_true_raw.astype(float)
    
    
    valid_label = ~np.isnan(y_true)
    y_true = y_true[valid_label]
    test_data_valid = test_data[valid_label].reset_index(drop=True)
    
    n_samples = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    pos_rate = n_pos / n_samples if n_samples > 0 else 0
    
    print(f"Samples: {n_samples} (Positive: {n_pos}, Negative: {n_neg})")
    print(f"Positive rate: {pos_rate:.4f}")
    
    
    is_imbalanced = (pos_rate < 0.3) or (pos_rate > 0.7)
    print(f"\n[DEBUG] Expected methods for this dataset:")
    print(f"  - Imbalanced: {is_imbalanced} (pos_rate={pos_rate:.4f})")
    print(f"  - AUC: {'DeLong' if n_samples >= 500 else 'BCa Bootstrap (5000)'}")
    print(f"  - AUPRC: {'BCa Bootstrap' if (is_imbalanced or n_samples < 500) else 'Percentile Bootstrap'}")
    print(f"  - AUPRC n_resamples: {2000 if (is_imbalanced and n_samples >= 500) else (3000 if (is_imbalanced and n_samples >= 100) else (5000 if n_samples >= 50 else 0))}")
    print()
    
    if n_samples == 0:
        print("Error: No valid samples after filtering!")
        return None, None, None, None
    
    
    predictions_dict = {}
    for model_name in compare_list:
        if model_name in test_data_valid.columns:
            pred = pd.to_numeric(test_data_valid[model_name], errors='coerce').values
            predictions_dict[model_name] = pred
        else:
            print(f"  Warning: {model_name} not found in dataset")
    
    if reference_model not in predictions_dict:
        print(f"Warning: Reference model {reference_model} not found!")
        return None, None, None, None
    
    ref_pred = predictions_dict[reference_model]
    
    
    print("\nComputing AUC with adaptive method...")
    auc_results = {}
    auc_p_values = []
    auc_models = []
    
    for model_name in compare_list:
        if model_name not in predictions_dict:
            continue
        
        pred = predictions_dict[model_name]
        
        if model_name == reference_model:
            result = compute_auc_ci_adaptive(y_true, pred)
        else:
            result = compute_auc_ci_adaptive(y_true, pred, ref_pred, reference_model)
        
        auc_results[model_name] = result
        print(f"  {MODEL_DISPLAY_NAMES.get(model_name, model_name)}: "
              f"AUC={result['auc']:.3f}, CI=[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}], "
              f"Method={result['method']}, n_samples={result['n_samples']}")
        
        if model_name != reference_model and not np.isnan(result.get('p_value', np.nan)):
            auc_p_values.append(result['p_value'])
            auc_models.append(model_name)
            print(f"    -> vs {reference_model}: p={result['p_value']:.4f} (n_common={result['n_samples']})")
    
   
    if auc_p_values:
        auc_fdr = benjamini_hochberg_fdr(auc_p_values)
        for idx, model_name in enumerate(auc_models):
            auc_results[model_name]['p_value_raw'] = auc_results[model_name]['p_value']
            auc_results[model_name]['p_value_fdr'] = auc_fdr[idx]
    
    
    print("\nComputing AUPRC with adaptive method...")
    auprc_results = {}
    auprc_p_values = []
    auprc_models = []
    
    for model_name in compare_list:
        if model_name not in predictions_dict:
            continue
        
        pred = predictions_dict[model_name]
        
        if model_name == reference_model:
            result = compute_auprc_ci_adaptive(y_true, pred)
        else:
            result = compute_auprc_ci_adaptive(y_true, pred, ref_pred, reference_model)
        
        auprc_results[model_name] = result
        print(f"  {MODEL_DISPLAY_NAMES.get(model_name, model_name)}: "
              f"AUPRC={result['auprc']:.3f}, CI=[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}], "
              f"Method={result['method']}, n_samples={result['n_samples']}")
        
        if model_name != reference_model and not np.isnan(result.get('p_value', np.nan)):
            auprc_p_values.append(result['p_value'])
            auprc_models.append(model_name)
            print(f"    -> vs {reference_model}: p={result['p_value']:.4f} (n_common={result['n_samples']})")
    
    
    if auprc_p_values:
        auprc_fdr = benjamini_hochberg_fdr(auprc_p_values)
        for idx, model_name in enumerate(auprc_models):
            auprc_results[model_name]['p_value_raw'] = auprc_results[model_name]['p_value']
            auprc_results[model_name]['p_value_fdr'] = auprc_fdr[idx]
    
    print("\nComputing classification metrics with Bootstrap CI...")
    metrics_results = {}
    
    for model_name in compare_list:
        if model_name not in predictions_dict:
            continue
        
        if model_name not in THRESHOLDS:
            print(f"  Warning: No threshold for {model_name}, skipping metrics")
            continue
        
        pred = predictions_dict[model_name]
        threshold = THRESHOLDS[model_name]
        
        metrics = calculate_metrics_with_ci(y_true, pred, threshold, model_name, n_bootstrap=1000)
        metrics_results[model_name] = metrics
        print(f"  {MODEL_DISPLAY_NAMES.get(model_name, model_name)}: "
              f"F1={metrics['f1']:.3f}, MCC={metrics['mcc']:.3f}, "
              f"ACC={metrics['accuracy']:.3f}")
    
    return auc_results, auprc_results, metrics_results, test_data_valid


def format_metric_with_ci(value, ci_lower, ci_upper, decimals=3):
    
    if np.isnan(value):
        return "N/A"
    if np.isnan(ci_lower) or np.isnan(ci_upper):
        return f"{value:.{decimals}f}"
    return f"{value:.{decimals}f}[{ci_lower:.{decimals}f},{ci_upper:.{decimals}f}]"


def format_p_value(p_value):
    
    if np.isnan(p_value):
        return "N/A"
    elif p_value < 0.001:
        return "<0.001"
    else:
        return f"{p_value:.3f}"


def generate_summary_tsv(all_results, output_file):
   
    rows = []
    
    for test_name, results in all_results.items():
        auc_results, auprc_results, metrics_results, _ = results  
        
        for model_name in auc_results.keys():
            auc_res = auc_results.get(model_name, {})
            auprc_res = auprc_results.get(model_name, {})
            met_res = metrics_results.get(model_name, {})
            
            display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            
           
            auc_str = format_metric_with_ci(
                auc_res.get('auc', np.nan),
                auc_res.get('ci_lower', np.nan),
                auc_res.get('ci_upper', np.nan)
            )
            auc_p_raw = format_p_value(auc_res.get('p_value_raw', np.nan))
            auc_p_fdr = format_p_value(auc_res.get('p_value_fdr', np.nan))
            auc_method = auc_res.get('method', 'N/A')
            
           
            auprc_str = format_metric_with_ci(
                auprc_res.get('auprc', np.nan),
                auprc_res.get('ci_lower', np.nan),
                auprc_res.get('ci_upper', np.nan)
            )
            auprc_p_raw = format_p_value(auprc_res.get('p_value_raw', np.nan))
            auprc_p_fdr = format_p_value(auprc_res.get('p_value_fdr', np.nan))
            auprc_method = auprc_res.get('method', 'N/A')
            
           
            f1_str = format_metric_with_ci(
                met_res.get('f1', np.nan),
                met_res.get('ci_lower', {}).get('f1', np.nan),
                met_res.get('ci_upper', {}).get('f1', np.nan)
            )
            mcc_str = format_metric_with_ci(
                met_res.get('mcc', np.nan),
                met_res.get('ci_lower', {}).get('mcc', np.nan),
                met_res.get('ci_upper', {}).get('mcc', np.nan)
            )
            acc_str = format_metric_with_ci(
                met_res.get('accuracy', np.nan),
                met_res.get('ci_lower', {}).get('accuracy', np.nan),
                met_res.get('ci_upper', {}).get('accuracy', np.nan)
            )
            prec_str = format_metric_with_ci(
                met_res.get('precision', np.nan),
                met_res.get('ci_lower', {}).get('precision', np.nan),
                met_res.get('ci_upper', {}).get('precision', np.nan)
            )
            sens_str = format_metric_with_ci(
                met_res.get('sensitivity', np.nan),
                met_res.get('ci_lower', {}).get('sensitivity', np.nan),
                met_res.get('ci_upper', {}).get('sensitivity', np.nan)
            )
            spec_str = format_metric_with_ci(
                met_res.get('specificity', np.nan),
                met_res.get('ci_lower', {}).get('specificity', np.nan),
                met_res.get('ci_upper', {}).get('specificity', np.nan)
            )
            
            row = {
                'Dataset': test_name,
                'Model': display_name,
                'AUC': auc_str,
                'AUC_p_raw': auc_p_raw,
                'AUC_p_FDR': auc_p_fdr,
                'AUC_Method': auc_method,
                'AUPRC': auprc_str,
                'AUPRC_p_raw': auprc_p_raw,
                'AUPRC_p_FDR': auprc_p_fdr,
                'AUPRC_Method': auprc_method,
                'F1': f1_str,
                'MCC': mcc_str,
                'Accuracy': acc_str,
                'Precision': prec_str,
                'Sensitivity': sens_str,
                'Specificity': spec_str
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, sep='\t', index=False, float_format='%.3f')
    print(f"\nSummary table saved: {output_file}")
    
    return df


def main():
    
    print("="*60)
    print("TPdsm Method Comparison with Confidence Intervals")
    print("="*60)
    
    
    file_dir = Path('/path/to/train/data')
    file_dir1 = Path('/path/to/test/data')
    
    
    result_dir = "TPdsm_symetrics_compare_results_usePredict"
    os.makedirs(result_dir, exist_ok=True)
    
    
    print("\nLoading training data...")
    train_file = file_dir / 'train.txt'
    if not train_file.exists():
        print(f"Error: Training file not found: {train_file}")
        print("Please update the file_dir path in main() function")
        sys.exit(1)
    
    train_data = pd.read_table(train_file, low_memory=False)
    
    
    print("Loading test datasets...")
    test_datasets = {}
    for test_key, test_info in TEST_DATASETS.items():
        test_file = file_dir1 / test_info['file']
        if test_file.exists():
            test_datasets[test_key] = pd.read_table(test_file, low_memory=False)
            print(f"  Loaded {test_key}: {len(test_datasets[test_key])} samples")
        else:
            print(f"  Warning: {test_file} not found!")
    
    
    print("\nImputing missing values...")
    print("  Imputing training data...")
    train_data = input_mean_overall(train_data, Condidate_features, population_freq_features)
    for test_key, test_data in test_datasets.items():
        print(f"  Imputing {test_key}...")
        test_datasets[test_key] = input_mean_overall(test_data, Condidate_features, population_freq_features)
    
   
    print("\nRemoving duplicates...")
    train_keys = train_data.iloc[:, :5].apply(lambda x: '|'.join(x.astype(str)), axis=1)
    
    for test_key, test_data in test_datasets.items():
        test_keys = test_data.iloc[:, :5].apply(lambda x: '|'.join(x.astype(str)), axis=1)
        unique_mask = ~test_keys.isin(train_keys)
        n_before = len(test_data)
        test_datasets[test_key] = test_data[unique_mask].reset_index(drop=True)
        n_after = len(test_datasets[test_key])
        print(f"  {test_key}: {n_before} -> {n_after} samples (removed {n_before - n_after} duplicates)")
    
   
    print("\nPredicting TPdsm scores...")
    for test_key, test_data in test_datasets.items():
        print(f"  {test_key}: predicting TPdsm_score...")
        test_datasets[test_key]["TPdsm_score"] = predict(model, test_data, trainFeatures)
        print(f"  {test_key}: predicted TPdsm_score shape = {len(test_data)}")
  
    cmaps = [plt.get_cmap('tab20'), plt.get_cmap('tab20b'), plt.get_cmap('tab20c')]
    color60 = []
    for cmap in cmaps:
        color60.extend(cmap(range(20)))
    
    all_results = {}
    
    for test_key, test_data in test_datasets.items():
        print(f"\n{'#'*60}")
        print(f"# Processing {TEST_DATASETS[test_key]['display_name']}")
        print(f"{'#'*60}")
        
        
        auc_results, auprc_results, metrics_results, test_data_valid = process_test_dataset(
            test_key, test_data, COMPARE_LIST, reference_model='TPdsm_score'
        )
        
        if auc_results is None:
            continue
        
        all_results[test_key] = (auc_results, auprc_results, metrics_results, test_data_valid)
        
       
        print(f"\nPlotting ROC curves...")
        roc_file = os.path.join(result_dir, f"{test_key}_ROC.pdf")
        plot_roc_curves(test_data_valid, COMPARE_LIST, color60, roc_file, auc_results)
        print(f"  Saved: {roc_file}")
        
       
        print(f"Plotting PR curves...")
        prc_file = os.path.join(result_dir, f"{test_key}_PRC.pdf")
        plot_prc_curves(test_data_valid, COMPARE_LIST, color60, prc_file, auprc_results)
        print(f"  Saved: {prc_file}")
    
    
    print(f"\n{'='*60}")
    print("Generating summary table...")
    summary_file = os.path.join(result_dir, "TPdsm_comparison_summary.tsv")
    generate_summary_tsv(all_results, summary_file)
    
    print(f"\n{'='*60}")
    print("All processing completed!")
    print(f"Results saved in: {result_dir}/")
    print(f"{'='*60}")
    
    return all_results


if __name__ == "__main__":
    results = main()
