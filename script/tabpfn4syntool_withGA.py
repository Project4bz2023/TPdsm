#!/usr/bin/env python3


import os
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, matthews_corrcoef,
    recall_score, precision_score, average_precision_score
)
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier
from tqdm import tqdm
import json
import random
import copy


POPULATION_FREQUENCY_FEATURES = [
    'gnomad41_genome_AF', 'gnomad41_genome_AF_raw', 'gnomad41_genome_AF_XX', 'gnomad41_genome_AF_XY',
    'gnomad41_genome_AF_grpmax', 'gnomad41_genome_faf95', 'gnomad41_genome_faf99',
    'gnomad41_genome_fafmax_faf95_max', 'gnomad41_genome_fafmax_faf99_max',
    'gnomad41_genome_AF_afr', 'gnomad41_genome_AF_ami', 'gnomad41_genome_AF_amr', 'gnomad41_genome_AF_asj',
    'gnomad41_genome_AF_eas', 'gnomad41_genome_AF_fin', 'gnomad41_genome_AF_mid', 'gnomad41_genome_AF_nfe',
    'gnomad41_genome_AF_remaining', 'gnomad41_genome_AF_sas',
    'gnomad41_exome_AF', 'gnomad41_exome_AF_raw', 'gnomad41_exome_AF_XX', 'gnomad41_exome_AF_XY',
    'gnomad41_exome_AF_grpmax', 'gnomad41_exome_faf95', 'gnomad41_exome_faf99',
    'gnomad41_exome_fafmax_faf95_max', 'gnomad41_exome_fafmax_faf99_max',
    'gnomad41_exome_AF_afr', 'gnomad41_exome_AF_amr', 'gnomad41_exome_AF_asj', 'gnomad41_exome_AF_eas',
    'gnomad41_exome_AF_fin', 'gnomad41_exome_AF_mid', 'gnomad41_exome_AF_nfe', 'gnomad41_exome_AF_remaining',
    'gnomad41_exome_AF_sas',
    'ExAC_ALL', 'ExAC_AFR', 'ExAC_AMR', 'ExAC_EAS', 'ExAC_FIN', 'ExAC_NFE', 'ExAC_OTH', 'ExAC_SAS',
    'china_map_AF',
    'ALL_sites_2015_08', 'AFR_sites_2015_08', 'AMR_sites_2015_08', 'EAS_sites_2015_08', 'EUR_sites_2015_08', 'SAS_sites_2015_08'
]


def input_mean_overall(input_data, input_c, filename):
    df = input_data.copy()
    mean_values = {}
    for col_idx in input_c:
        col_name = col_idx
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        mean_value = df[col_name].mean(skipna=True)
        mean_values[col_name] = mean_value
        df[col_name].fillna(mean_value, inplace=True)
        df[col_name] = df[col_name].astype(float)
    output_path = os.path.join(os.getcwd(), "processed." + filename)
    df.to_csv(output_path, sep='\t', index=False)
    return df, mean_values


def fill_missing_with_train_mean(input_data, input_c, train_mean_values, filename, pop_freq_features=None):
    
    df = input_data.copy()
    
    for col_idx in input_c:
        col_name = col_idx
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        if pop_freq_features and col_name in pop_freq_features:
            df[col_name].fillna(0, inplace=True)
        elif col_name in train_mean_values:
            df[col_name].fillna(train_mean_values[col_name], inplace=True)
        else:
            mean_value = df[col_name].mean(skipna=True)
            df[col_name].fillna(mean_value, inplace=True)
        
        df[col_name] = df[col_name].astype(float)
    output_path = os.path.join(os.getcwd(), "processed." + filename)
    df.to_csv(output_path, sep='\t', index=False)
    return df


def deduplicate_test_data(test_data, train_data, dedup_columns=None):
   
    if dedup_columns is None:
        dedup_columns = [col for col in test_data.columns if col != 'Otherinfo1']
    
    available_cols = [col for col in dedup_columns if col in test_data.columns and col in train_data.columns]
    
    if len(available_cols) == 0:
        print("Warning: No common columns found for deduplication. Skipping deduplication.")
        return test_data
    
    print(f"\nDeduplicating test data using {len(available_cols)} columns...")
    
    train_keys = train_data[available_cols].drop_duplicates()
    
    test_data_copy = test_data.copy()
    test_data_copy['_merge_key'] = test_data_copy[available_cols].apply(lambda x: tuple(x), axis=1)
    train_keys['_merge_key'] = train_keys.apply(lambda x: tuple(x), axis=1)
    
    duplicated_keys = set(train_keys['_merge_key'])
    test_data_dedup = test_data_copy[~test_data_copy['_merge_key'].isin(duplicated_keys)].drop('_merge_key', axis=1)
    
    original_count = len(test_data)
    dedup_count = len(test_data_dedup)
    removed_count = original_count - dedup_count
    
    print(f"  Original test samples: {original_count}")
    print(f"  After deduplication: {dedup_count}")
    print(f"  Removed duplicate samples: {removed_count}")
    
    return test_data_dedup


def evaluate_model(model, X, y):
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    auc = roc_auc_score(y, y_pred_proba)
    auprc = average_precision_score(y, y_pred_proba)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    sensitivity = recall_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    return {
        'auc': auc,
        'auprc': auprc,
        'f1': f1,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'combined_score': np.mean([auc, auprc, f1, mcc, sensitivity, accuracy, precision, recall])
    }


def remove_constant_features(data, candidate_features):
    non_constant_features = []
    for feature in candidate_features:
        if data[feature].nunique() > 1:
            non_constant_features.append(feature)
    return non_constant_features


def initialize_population(features, population_size, min_features=5):
    population = []
    n = len(features)

    if min_features > n:
        min_features = max(1, n)
        print(f"Warning: min_features adjusted to {min_features} (total features: {n})")

    for _ in range(population_size):
        r = random.randint(min_features, n)
        individual = random.sample(features, r)
        population.append(individual)
    return population


def evaluate_fitness_cv(individual, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                         best_score, fitness_weight=0.7):
    X_train_sub = X_train_fold[individual]
    clf = TabPFNClassifier()
    clf.fit(X_train_sub, y_train_fold)

    X_val_sub = X_val_fold[individual]
    val_metrics = evaluate_model(clf, X_val_sub, y_val_fold)

    print(f"Feature Combination: {len(individual)} features, Validation AUC: {val_metrics['auc']:.4f}, AUPRC: {val_metrics['auprc']:.4f}")

    if val_metrics['auc'] >= best_score:
        fitness = val_metrics['auc']
    else:
        fitness = val_metrics['auc'] * fitness_weight

    return fitness, val_metrics, clf


def crossover(parent1, parent2):
    common_features = list(set(parent1) & set(parent2))
    unique_features1 = [f for f in parent1 if f not in common_features]
    unique_features2 = [f for f in parent2 if f not in common_features]

    if len(unique_features1) == 0 and len(unique_features2) == 0:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    max_len = max(len(unique_features1), len(unique_features2))
    if max_len == 0:
        max_len = 1

    crossover_point = random.randint(0, max_len)
    child1 = common_features + unique_features1[:crossover_point] + unique_features2[crossover_point:]
    child2 = common_features + unique_features2[:crossover_point] + unique_features1[crossover_point:]

    child1 = list(set(child1))
    child2 = list(set(child2))

    return child1, child2


def mutate(individual, features, mutation_rate=0.1, min_features=5):
    individual = copy.deepcopy(individual)

    available_features = [f for f in features if f not in individual]

    if len(available_features) == 0:
        return individual

    for _ in range(len(individual)):
        if random.random() < mutation_rate:
            if len(individual) > min_features and len(available_features) > 0:
                feature_to_remove = random.choice(individual)
                individual.remove(feature_to_remove)
                new_feature = random.choice(available_features)
                individual.append(new_feature)
                available_features.remove(new_feature)
                available_features.append(feature_to_remove)

    return individual


def check_convergence(fitness_history, patience, min_improvement=0.001):
    if len(fitness_history) < patience * 2:
        return False

    recent_best = max(fitness_history[-patience:])
    older_best = max(fitness_history[-2*patience:-patience])

    if recent_best - older_best < min_improvement:
        return True
    return False


def genetic_algorithm_cv(features, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                          best_score, fold_num, population_size=100, generations=100,
                          mutation_rate=0.1, early_stop_patience=10, elitism_count=5,
                          fitness_weight=0.7, min_features=5):
   
    if elitism_count >= population_size:
        elitism_count = max(1, population_size // 2)
        print(f"  Warning: elitism_count adjusted to {elitism_count} (must be < population_size)")

    if min_features > len(features):
        min_features = max(1, len(features))
        print(f"  Warning: min_features adjusted to {min_features} (total features: {len(features)})")

    population = initialize_population(features, population_size, min_features)

    best_fitness = 0
    best_individual = None
    best_model = None
    best_metrics = None
    fitness_history = []
    no_improvement_count = 0
    generation_completed = 0
    stop_reason = "max_generations"

    saved_models = []
    model_counter = 1
    saved_keys = set()

    for generation in tqdm(range(generations), desc=f"Fold {fold_num} - Generations Progress"):
        fitness_cache = {}
        metrics_cache = {}
        model_cache = {}
        fitness_scores = []

        for individual in population:
            individual_key = tuple(sorted(individual))
            if individual_key not in fitness_cache:
                fitness, metrics, model = evaluate_fitness_cv(
                    individual, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                    best_score, fitness_weight
                )
                fitness_cache[individual_key] = fitness
                metrics_cache[individual_key] = metrics
                model_cache[individual_key] = model

            fitness_scores.append(fitness_cache[individual_key])

            if individual_key not in saved_keys:
                if metrics_cache[individual_key]['auc'] >= best_score:
                    saved_keys.add(individual_key)

                    feature_filename = f"feature_fold{fold_num}_{model_counter}.json"
                    with open(feature_filename, 'w') as f:
                        json.dump(individual, f)

                    model_filename = f"bestmodel_fold{fold_num}_{model_counter}.pkl"
                    joblib.dump(model_cache[individual_key], model_filename)

                    saved_models.append({
                        'fold': fold_num,
                        'model_n': model_counter,
                        'model_name': model_filename,
                        'feature_name': feature_filename,
                        'features': individual.copy(),
                        'val_metrics': metrics_cache[individual_key]
                    })

                    print(f"\n  Saved model #{model_counter}: AUC={metrics_cache[individual_key]['auc']:.4f}, "
                          f"Features={len(individual)}")

                    model_counter += 1

        current_best_fitness = max(fitness_scores)
        current_best_idx = fitness_scores.index(current_best_fitness)
        current_best_individual = population[current_best_idx]
        current_best_key = tuple(sorted(current_best_individual))
        fitness_history.append(current_best_fitness)

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = copy.deepcopy(current_best_individual)
            best_metrics = metrics_cache[current_best_key]
            best_model = model_cache[current_best_key]

            no_improvement_count = 0

            print(f"\n  New best at generation {generation + 1}: AUC={best_metrics['auc']:.4f}, "
                  f"Features={len(best_individual)}")
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stop_patience:
            print(f"\n  Early stopping at generation {generation + 1}: No improvement for {early_stop_patience} generations")
            generation_completed = generation + 1
            stop_reason = "early_stopping"
            break

        if check_convergence(fitness_history, early_stop_patience):
            print(f"\n  Convergence detected at generation {generation + 1}")
            generation_completed = generation + 1
            stop_reason = "convergence"
            break

        sorted_pop_with_fitness = sorted(
            zip(population, fitness_scores),
            key=lambda x: x[1],
            reverse=True
        )

        elite_individuals = [copy.deepcopy(ind) for ind, fit in sorted_pop_with_fitness[:elitism_count]]

        new_population = []
        num_offspring = population_size - elitism_count
        num_pairs = (num_offspring + 1) // 2

        for _ in range(num_pairs):
            if sum(fitness_scores) <= 0:
                parent1, parent2 = random.sample(population, k=2)
            else:
                parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, features, mutation_rate, min_features)
            child2 = mutate(child2, features, mutation_rate, min_features)
            new_population.extend([child1, child2])

        new_population = new_population[:num_offspring]
        population = elite_individuals + new_population
        generation_completed = generation + 1

    print(f"\n{'='*60}")
    print(f"Fold {fold_num} Summary:")
    print(f"  Total models saved: {len(saved_models)}")
    print(f"  Generations run: {generation_completed}")
    print(f"  Stop reason: {stop_reason}")
    if best_metrics:
        print(f"  Best AUC: {best_metrics['auc']:.4f}")
        print(f"  Best AUPRC: {best_metrics['auprc']:.4f}")
        print(f"  Best Features: {len(best_individual)}")
    print(f"{'='*60}")

    fold_summary = {
        'fold': fold_num,
        'best_AUC': best_metrics['auc'] if best_metrics else None,
        'best_AUPRC': best_metrics['auprc'] if best_metrics else None,
        'best_F1': best_metrics['f1'] if best_metrics else None,
        'best_MCC': best_metrics['mcc'] if best_metrics else None,
        'best_Sensitivity': best_metrics['sensitivity'] if best_metrics else None,
        'best_Accuracy': best_metrics['accuracy'] if best_metrics else None,
        'best_Precision': best_metrics['precision'] if best_metrics else None,
        'n_features': len(best_individual) if best_individual else 0,
        'selected_features': best_individual if best_individual else [],
        'generations_run': generation_completed,
        'stop_reason': stop_reason,
        'total_models_saved': len(saved_models)
    }

    return saved_models, best_individual, best_metrics, fold_summary


def generate_results_table(all_fold_results):
    if not all_fold_results:
        print("No satisfying models found.")
        return

    results = []

    for model_info in all_fold_results:
        if model_info is None:
            continue
        fold_num = model_info['fold']
        model_n = model_info['model_n']
        model_name = model_info['model_name']
        feature_name = model_info['feature_name']
        val_metrics = model_info['val_metrics']

        row = {
            'fold': fold_num,
            'model_n': model_n,
            'model_name': model_name,
            'feature_name': feature_name,
            'n_features': len(model_info['features']),
            'val_AUC': val_metrics['auc'],
            'val_AUPRC': val_metrics['auprc'],
            'val_F1': val_metrics['f1'],
            'val_MCC': val_metrics['mcc'],
            'val_Sensitivity': val_metrics['sensitivity'],
            'val_Accuracy': val_metrics['accuracy'],
            'val_Precision': val_metrics['precision']
        }
        results.append(row)

    df_results = pd.DataFrame(results)

    column_order = ['fold', 'model_n', 'model_name', 'feature_name', 'n_features',
                    'val_AUC', 'val_AUPRC', 'val_F1', 'val_MCC', 'val_Sensitivity',
                    'val_Accuracy', 'val_Precision']
    df_results = df_results[column_order]

    output_file = "result_metric_list_cv_final.csv"
    df_results.to_csv(output_file, index=False)

    print(f"\nResults table saved to: {output_file}")
    print(df_results)

    print("\n" + "="*60)
    print("Summary Statistics Across All Models")
    print("="*60)
    print(f"Total models saved: {len(df_results)}")
    print(f"Models per fold: {df_results.groupby('fold').size().to_dict()}")
    print(f"Mean Validation AUC: {df_results['val_AUC'].mean():.4f} (+/- {df_results['val_AUC'].std():.4f})")
    print(f"Mean Validation AUPRC: {df_results['val_AUPRC'].mean():.4f} (+/- {df_results['val_AUPRC'].std():.4f})")
    print(f"Mean Features Selected: {df_results['n_features'].mean():.1f}")

    return df_results


def generate_cv_fold_summary(fold_summaries):
   
    if not fold_summaries:
        print("No fold summaries available.")
        return None

    summary_data = []
    for fs in fold_summaries:
        row = {
            'fold': fs['fold'],
            'best_AUC': fs['best_AUC'],
            'best_AUPRC': fs['best_AUPRC'],
            'best_F1': fs['best_F1'],
            'best_MCC': fs['best_MCC'],
            'best_Sensitivity': fs['best_Sensitivity'],
            'best_Accuracy': fs['best_Accuracy'],
            'best_Precision': fs['best_Precision'],
            'n_features': fs['n_features'],
            'generations_run': fs['generations_run'],
            'stop_reason': fs['stop_reason'],
            'total_models_saved': fs['total_models_saved'],
            'selected_features': ';'.join(fs['selected_features']) if fs['selected_features'] else ''
        }
        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)

    output_file = "cv_fold_summary.csv"
    df_summary.to_csv(output_file, index=False)

    print(f"\nCV Fold Summary saved to: {output_file}")
    print(df_summary[['fold', 'best_AUC', 'n_features', 'generations_run', 'stop_reason']])

    print("\n" + "="*60)
    print("CV Fold Summary Statistics")
    print("="*60)
    print(f"Mean AUC: {df_summary['best_AUC'].mean():.4f} (+/- {df_summary['best_AUC'].std():.4f})")
    print(f"Mean Features: {df_summary['n_features'].mean():.1f} (+/- {df_summary['n_features'].std():.1f})")
    print(f"Mean Generations: {df_summary['generations_run'].mean():.1f}")
    print(f"Stop reasons: {df_summary['stop_reason'].value_counts().to_dict()}")

    for fs in fold_summaries:
        if fs['selected_features']:
            feature_filename = f"fold{fs['fold']}_selected_features.json"
            with open(feature_filename, 'w') as f:
                json.dump(fs['selected_features'], f)
            print(f"Fold {fs['fold']} features saved to: {feature_filename}")

    return df_summary


def generate_feature_frequency_table(feature_selection_records, all_features, n_splits):
  
    frequency_data = []

    for feature in all_features:
        row = {'feature_name': feature}
        frequency = 0

        for fold_num in range(1, n_splits + 1):
            selected_in_fold = feature in feature_selection_records.get(fold_num, [])
            row[f'fold{fold_num}'] = 1 if selected_in_fold else 0
            if selected_in_fold:
                frequency += 1

        row['frequency'] = frequency
        frequency_data.append(row)

    df_frequency = pd.DataFrame(frequency_data)
    df_frequency = df_frequency.sort_values('frequency', ascending=False)

    output_file = "feature_frequency_table.csv"
    df_frequency.to_csv(output_file, index=False)

    print(f"\nFeature frequency table saved to: {output_file}")
    print(f"\nFeature Frequency Summary:")
    print(f"  Features selected in all {n_splits} folds: {len(df_frequency[df_frequency['frequency'] == n_splits])}")
    print(f"  Features selected in >= 3 folds: {len(df_frequency[df_frequency['frequency'] >= 3])}")
    print(f"  Features selected in >= 1 fold: {len(df_frequency[df_frequency['frequency'] >= 1])}")

    return df_frequency


def select_stable_features(df_frequency, threshold):
 
    stable_features = df_frequency[df_frequency['frequency'] >= threshold]['feature_name'].tolist()

    print(f"\nStable Features Selection (threshold >= {threshold}):")
    print(f"  Total stable features: {len(stable_features)}")

    return stable_features


def train_final_model(X_train, y_train, stable_features):
   
    print(f"\n{'='*60}")
    print("Stage 2: Final Model Training")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Stable features: {len(stable_features)}")

    X_train_sub = X_train[stable_features]
    clf = TabPFNClassifier()
    clf.fit(X_train_sub, y_train)

    model_filename = "final_model.pkl"
    joblib.dump(clf, model_filename)
    print(f"\nFinal model saved to: {model_filename}")

    feature_filename = "final_stable_features.json"
    with open(feature_filename, 'w') as f:
        json.dump(stable_features, f)
    print(f"Stable features saved to: {feature_filename}")

    return clf


def evaluate_on_test_set(model, X_test, y_test, stable_features):
   
    print(f"\n{'='*60}")
    print("Test Set Evaluation")
    print(f"{'='*60}")
    print(f"Test samples: {len(X_test)}")

    X_test_sub = X_test[stable_features]
    test_metrics = evaluate_model(model, X_test_sub, y_test)

    print(f"\nTest Set Metrics:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  AUPRC: {test_metrics['auprc']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  MCC: {test_metrics['mcc']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")

    results_df = pd.DataFrame([{
        'test_AUC': test_metrics['auc'],
        'test_AUPRC': test_metrics['auprc'],
        'test_F1': test_metrics['f1'],
        'test_MCC': test_metrics['mcc'],
        'test_Sensitivity': test_metrics['sensitivity'],
        'test_Accuracy': test_metrics['accuracy'],
        'test_Precision': test_metrics['precision'],
        'n_features': len(stable_features)
    }])

    output_file = "test_set_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nTest results saved to: {output_file}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(
        description='TabPFN with GA Feature Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--population_size', type=int, default=100,
                        help='Number of individuals in GA population (default: 100)')
    parser.add_argument('--generations', type=int, default=100,
                        help='Maximum number of GA generations/iterations (default: 100)')
    parser.add_argument('--mutation_rate', type=float, default=0.1,
                        help='Mutation rate for GA (default: 0.1)')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Early stopping patience - generations without improvement (default: 10)')
    parser.add_argument('--elitism_count', type=int, default=5,
                        help='Number of best individuals to preserve (default: 5)')
    parser.add_argument('--fitness_weight', type=float, default=0.7,
                        help='Weight for below-threshold fitness (default: 0.7)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--best_score', type=float, default=0.80,
                        help='AUC threshold for saving models (default: 0.80)')
    parser.add_argument('--min_features', type=int, default=5,
                        help='Minimum number of features in each individual (default: 5)')
    parser.add_argument('--feature_frequency_threshold', type=int, default=2,
                        help='Minimum frequency threshold for stable features (default: 2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--train_data_path', type=str, default="/path/to/train_data.txt",
                        help='Path to training data file')
    parser.add_argument('--test_data_path', type=str, nargs='+', default=None,
                        help='Path(s) to test data file(s). Multiple paths supported. (optional)')

    args = parser.parse_args()

    if args.feature_frequency_threshold < 1 or args.feature_frequency_threshold > args.n_splits:
        print(f"Warning: feature_frequency_threshold should be between 1 and {args.n_splits}")
        args.feature_frequency_threshold = min(max(1, args.feature_frequency_threshold), args.n_splits)
        print(f"Adjusted to: {args.feature_frequency_threshold}")

    random.seed(args.random_state)
    np.random.seed(args.random_state)

    print("=" * 60)
    print("TabPFN with GA Feature Selection")
    print("=" * 60)
    print("\nGA Configuration:")
    print(f"  Population Size: {args.population_size}")
    print(f"  Max Generations: {args.generations}")
    print(f"  Mutation Rate: {args.mutation_rate}")
    print(f"  Crossover Rate: 100% (implicit)")
    print(f"  Early Stop Patience: {args.early_stop_patience}")
    print(f"  Elitism Count: {args.elitism_count}")
    print(f"  Fitness Weight: {args.fitness_weight}")
    print(f"\nCross-Validation Configuration:")
    print(f"  Number of Folds: {args.n_splits}")
    print(f"  AUC Threshold for Saving: {args.best_score}")
    print(f"  Min Features: {args.min_features}")
    print(f"  Feature Frequency Threshold: {args.feature_frequency_threshold}")
    print(f"  Random State: {args.random_state}")
    print(f"\nData Paths:")
    print(f"  Training Data: {args.train_data_path}")
    print(f"  Test Data: {args.test_data_path if args.test_data_path else 'Not provided'}")
    print("=" * 60)

    print("\nLoading training data...")
    data = pd.read_csv(args.train_data_path, sep="\t", low_memory=False)

    candidate_features = [
        "CADD_PHRED", "CADD_PHRED_rankscore", "CADD_RawScore", "CADD_RawScore_rankscore", "DANN", "DANN_rankscore",
        "DDIG", "DDIG_rankscore", "eigen", "eigen_rankscore", "EnDSM", "EnDSM_rankscore",
        "fathmm_MKL_coding", "fathmm_MKL_coding_rankscore", "fathmm_xf_coding", "fathmm_xf_coding_rankscore",
        "frDSM", "frDSM_rankscore", "PhD_SNPg", "PhD_SNPg_rankscore", "PrDSM", "PrDSM_rankscore",
        "silva", "silva_rankscore", "syntool", "syntool_rankscore", "usDSM", "usDSM_rankscore",
        "AbSplice_DNA_max", "AbSplice_DNA_max_rankscore", "delta_logit_psi_max", "delta_logit_psi_max_rankscore",
        "delta_psi_max", "delta_psi_max_rankscore", "delta_score", "delta_score_rankscore",
        "spidex_dpsi_max_tissue", "spidex_dpsi_max_tissue_rankscore", "spidex_dpsi_zscore", "spidex_dpsi_zscore_rankscore",
        "Synvepscore_max", "Synvepscore_max_rankscore", "Synvepscore_mean", "Synvepscore_mean_rankscore",
        "Synvepscore_min", "Synvepscore_min_rankscore", "cadd_fitcons", "cadd_mapability_20bp", "cadd_mapability_35bp",
        "cadd_phast_cons_mammalian", "cadd_phast_cons_primate", "cadd_phast_cons_vertebrate",
        "cadd_phylop_mammalian", "cadd_phylop_primate", "cadd_phylop_vertebrate", "gerp_gt2",
        "ALL_sites_2015_08", "AFR_sites_2015_08", "AMR_sites_2015_08", "EAS_sites_2015_08",
        "EUR_sites_2015_08", "SAS_sites_2015_08", "gnomad41_genome_AF", "gnomad41_genome_AF_raw",
        "gnomad41_genome_AF_XX", "gnomad41_genome_AF_XY", "gnomad41_genome_AF_grpmax",
        "gnomad41_genome_faf95", "gnomad41_genome_faf99", "gnomad41_genome_fafmax_faf95_max",
        "gnomad41_genome_fafmax_faf99_max", "gnomad41_genome_AF_afr", "gnomad41_genome_AF_ami",
        "gnomad41_genome_AF_amr", "gnomad41_genome_AF_asj", "gnomad41_genome_AF_eas",
        "gnomad41_genome_AF_fin", "gnomad41_genome_AF_mid", "gnomad41_genome_AF_nfe",
        "gnomad41_genome_AF_remaining", "gnomad41_genome_AF_sas", "gnomad41_exome_AF",
        "gnomad41_exome_AF_raw", "gnomad41_exome_AF_XX", "gnomad41_exome_AF_XY",
        "gnomad41_exome_AF_grpmax", "gnomad41_exome_faf95", "gnomad41_exome_faf99",
        "gnomad41_exome_fafmax_faf95_max", "gnomad41_exome_fafmax_faf99_max", "gnomad41_exome_AF_afr",
        "gnomad41_exome_AF_amr", "gnomad41_exome_AF_asj", "gnomad41_exome_AF_eas",
        "gnomad41_exome_AF_fin", "gnomad41_exome_AF_mid", "gnomad41_exome_AF_nfe",
        "gnomad41_exome_AF_remaining", "gnomad41_exome_AF_sas", "ExAC_ALL", "ExAC_AFR",
        "ExAC_AMR", "ExAC_EAS", "ExAC_FIN", "ExAC_NFE", "ExAC_OTH", "ExAC_SAS", "china_map_AF",
        "delta_CAI", "CAM", "CF", "delta_CUB", "delta_FracOpt", "delta_ICDI", "delta_SCUO",
        "delta_tAI", "#GERP++", "#RSCU", "dRSCU", "#CpG?", "CpG_exon", "#SR-", "SR+",
        "#FAS6-", "FAS6+", "#MES", "dMES", "MES+", "MES-", "MEC-MC?", "MEC-CS?", "MES-KM?",
        "#PESE-", "PESE+", "PESS-", "PESS+", "#f_premrna", "f_mrna"
    ]

    print("\nPreprocessing data...")
    data, train_mean_values = input_mean_overall(data, candidate_features, "train.hg38_multianno.txt")

    print("Removing constant features...")
    candidate_features = remove_constant_features(data, candidate_features)

    print("Preparing data for cross-validation...")
    X = data[candidate_features]
    Y = data['Otherinfo1']

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    all_fold_results = []
    feature_selection_records = {}
    fold_best_metrics = []
    fold_summaries = []

    print(f"\n{'=' * 60}")
    print("Stage 1: Performance Evaluation with Feature Stability")
    print(f"Starting {args.n_splits}-Fold Cross-Validation with GA")
    print(f"{'=' * 60}")

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X, Y), 1):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_num}/{args.n_splits}")
        print(f"{'=' * 60}")
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")

        X_train_fold = X.iloc[train_idx].reset_index(drop=True)
        y_train_fold = Y.iloc[train_idx].reset_index(drop=True)
        X_val_fold = X.iloc[val_idx].reset_index(drop=True)
        y_val_fold = Y.iloc[val_idx].reset_index(drop=True)

        print(f"\nStarting GA feature selection for Fold {fold_num}...")
        print("NOTE: GA uses ONLY training fold data (no data leakage)")

        fold_results, best_features, best_metrics, fold_summary = genetic_algorithm_cv(
            candidate_features, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
            args.best_score, fold_num, args.population_size, args.generations,
            args.mutation_rate, args.early_stop_patience, args.elitism_count,
            args.fitness_weight, args.min_features
        )

        if fold_results:
            all_fold_results.extend(fold_results)

        if best_features:
            feature_selection_records[fold_num] = best_features

        if best_metrics:
            fold_best_metrics.append(best_metrics)

        fold_summaries.append(fold_summary)

    print(f"\n{'=' * 60}")
    print("Stage 1 Completed: Cross-Validation Results")
    print(f"{'=' * 60}")

    results_df = generate_results_table(all_fold_results)

    cv_summary_df = generate_cv_fold_summary(fold_summaries)

    if fold_best_metrics:
        mean_auc = np.mean([m['auc'] for m in fold_best_metrics])
        std_auc = np.std([m['auc'] for m in fold_best_metrics])
        print(f"\n5-Fold CV Performance Summary:")
        print(f"  Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

    print(f"\n{'=' * 60}")
    print("Feature Frequency Analysis")
    print(f"{'=' * 60}")

    df_frequency = generate_feature_frequency_table(feature_selection_records, candidate_features, args.n_splits)

    stable_features = select_stable_features(df_frequency, args.feature_frequency_threshold)

    if len(stable_features) == 0:
        print("\nWarning: No stable features found. Consider lowering the threshold.")
        print("Using features with highest frequency instead...")
        max_freq = df_frequency['frequency'].max()
        stable_features = df_frequency[df_frequency['frequency'] == max_freq]['feature_name'].tolist()
        print(f"Selected {len(stable_features)} features with frequency = {max_freq}")

    final_model = train_final_model(X, Y, stable_features)

    all_test_results = []

    if args.test_data_path:
        for test_idx, test_path in enumerate(args.test_data_path, 1):
            print(f"\n{'=' * 60}")
            print(f"External Test Set Evaluation ({test_idx}/{len(args.test_data_path)})")
            print(f"{'=' * 60}")
            print(f"\nLoading test data from: {test_path}")

            test_data = pd.read_csv(test_path, sep="\t", low_memory=False)

            test_data = deduplicate_test_data(test_data, data, candidate_features)

            test_name = os.path.splitext(os.path.basename(test_path))[0]
            processed_filename = f"test_data_{test_name}.txt"
            test_data = fill_missing_with_train_mean(
                test_data, stable_features, train_mean_values, processed_filename,
                pop_freq_features=POPULATION_FREQUENCY_FEATURES
            )

            X_test = test_data[stable_features]
            y_test = test_data['Otherinfo1']

            test_metrics = evaluate_on_test_set(final_model, X_test, y_test, stable_features)

            test_result = {
                'test_dataset': test_name,
                'test_path': test_path,
                **{f'test_{k}': v for k, v in test_metrics.items()},
                'n_features': len(stable_features)
            }
            all_test_results.append(test_result)

            result_filename = f"test_set_results_{test_name}.csv"
            result_df = pd.DataFrame([test_result])
            result_df.to_csv(result_filename, index=False)
            print(f"Test results saved to: {result_filename}")

        if len(args.test_data_path) > 1:
            all_results_df = pd.DataFrame(all_test_results)
            all_results_filename = "test_set_results_all.csv"
            all_results_df.to_csv(all_results_filename, index=False)
            print(f"\nAll test results summary saved to: {all_results_filename}")
            print(f"\n{'=' * 60}")
            print("Multiple Test Sets Summary")
            print(f"{'=' * 60}")
            print(all_results_df[['test_dataset', 'test_auc', 'test_auprc', 'test_f1']])



if __name__ == "__main__":
    main()
