#!/usr/bin/env python3


import os
import joblib
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, matthews_corrcoef,
    recall_score, precision_score, average_precision_score
)
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from tqdm import tqdm
import json
import random


def input_mean_overall(input_data, input_c, filename):
    df = input_data.copy()
    for col_idx in input_c:
        col_name = col_idx
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        mean_value = df[col_name].mean(skipna=True)
        df[col_name].fillna(mean_value, inplace=True)
        df[col_name] = df[col_name].astype(float)
    output_path = os.path.join(os.getcwd(), "processed." + filename)
    df.to_csv(output_path, sep='\t', index=False)
    return df


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
    for _ in range(population_size):
        r = random.randint(min_features, n)
        individual = random.sample(features, r)
        population.append(individual)
    return population


def evaluate_fitness(individual, X_train, y_train, test_sets, test_targets,best_score):
    X_train_sub = X_train[individual]
    clf = TabPFNClassifier()
    clf.fit(X_train_sub, y_train)

    all_tests_pass = True
    test_results = {}
    for i, test_set in enumerate(test_sets):
        X_test_sub = test_set[individual]
        test_metrics = evaluate_model(clf, X_test_sub, test_targets[i])
        test_results[f'test_set_{i + 1}'] = test_metrics

        print(f"Feature Combination: {individual}, Test Set {i + 1} AUC: {test_metrics['auc']}, AUPRC: {test_metrics['auprc']}")

        if i == 0 and test_metrics['auc'] < best_score:
            all_tests_pass = False
        elif i == 1 and test_metrics['auc'] < best_score:
            all_tests_pass = False
        elif i == 2 and test_metrics['auc'] < best_score:
            all_tests_pass = False
        elif i == 3 and test_metrics['auc'] < best_score:
            all_tests_pass = False

    if all_tests_pass:
        return sum([test_metrics['auc'] for test_metrics in test_results.values()])
    return 0


def crossover(parent1, parent2):
    common_features = list(set(parent1) & set(parent2))
    unique_features1 = [f for f in parent1 if f not in common_features]
    unique_features2 = [f for f in parent2 if f not in common_features]

    crossover_point = random.randint(0, len(unique_features1))
    child1 = common_features + unique_features1[:crossover_point] + unique_features2[crossover_point:]
    child2 = common_features + unique_features2[:crossover_point] + unique_features1[crossover_point:]
    return child1, child2


def mutate(individual, features, mutation_rate=0.1):
    n = len(features)
    for _ in range(len(individual)):
        if random.random() < mutation_rate:
            feature_to_remove = random.choice(individual)
            individual.remove(feature_to_remove)
            new_feature = random.choice([f for f in features if f not in individual])
            individual.append(new_feature)
    return individual


def genetic_algorithm(features, X_train, y_train, test_sets, test_targets, best_score,population_size=100, generations=100):
    population = initialize_population(features, population_size)
    satisfied_models = []
    model_counter = 1

    for generation in tqdm(range(generations), desc="Generations Progress"):
        fitness_scores = []
        # 依次评估每个个体的适应度
        for individual in population:
            fitness = evaluate_fitness(individual, X_train, y_train, test_sets, test_targets,best_score)
            fitness_scores.append(fitness)

        new_population = []
        for _ in range(population_size // 2):
            # 新增：处理适应度总和为0的情况
            if sum(fitness_scores) <= 0:
                # 所有适应度为0时，随机选择父代（等概率）
                parent1, parent2 = random.sample(population, k=2)
            else:
                parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, features)
            child2 = mutate(child2, features)
            new_population.extend([child1, child2])

        population = new_population

        # 检查是否有满足条件的模型，依次处理每个个体
        for individual in population:
            fitness = evaluate_fitness(individual, X_train, y_train, test_sets, test_targets,best_score)
            if fitness > 0:
                X_train_sub = X_train[individual]
                clf = TabPFNClassifier()
                # 依次训练模型
                clf.fit(X_train_sub, y_train)

                # 保存特征组合
                feature_filename = f"feature_{model_counter}.json"
                with open(feature_filename, 'w') as f:
                    json.dump(individual, f)

                # 保存模型
                model_filename = f"bestmodel_{model_counter}.pkl"
                joblib.dump(clf, model_filename)

                test_results = {}
                for i, test_set in enumerate(test_sets):
                    X_test_sub = test_set[individual]
                    test_metrics = evaluate_model(clf, X_test_sub, test_targets[i])
                    test_results[f'test_set_{i + 1}'] = test_metrics

                # 记录结果
                satisfied_models.append({
                    'batch': model_counter,
                    'features': individual.copy(),
                    'model_path': model_filename,
                    'test_results': test_results
                })

                print(f"\n✅ Found satisfying model #{model_counter} with {len(individual)} features")
                print(f"  Saved as: {feature_filename} and {model_filename}")

                model_counter += 1

    return satisfied_models


def generate_results_table(satisfied_models):
    if not satisfied_models:
        print("No satisfying models found.")
        return

    results = []

    for model_info in satisfied_models:
        batch_num = model_info['batch']
        test_results = model_info['test_results']
        row = {'batch': batch_num}

        for i in range(1, 5):
            test_name = f'test_set_{i}'
            if test_name in test_results:
                # 同时保存 AUC 和 AUPRC
                row[f'test{i}_AUC'] = test_results[test_name]['auc']
                row[f'test{i}_AUPRC'] = test_results[test_name]['auprc']

        results.append(row)

    # 转换为 DataFrame 并保存
    df_results = pd.DataFrame(results)

    # 按列排序：批次、测试 1(AUC+AUPRC)、测试 2(AUC+AUPRC)...
    column_order = ['batch']
    for i in range(1, 5):
        column_order.extend([f'test{i}_AUC', f'test{i}_AUPRC'])

    df_results = df_results[column_order]
    output_file = "result_metric_list.csv"
    df_results.to_csv(output_file, index=False)

    print(f"\nResults table saved to: {output_file}")
    print(df_results)

    return df_results


def main():

    # 设置测试集 AUC 阈值
    best_score = 0.80
 

    # 加载和预处理数据
    file_dir = "/path/to/files/"
    data = pd.read_csv(file_dir + "/train_dataset.hg38_multianno.txt", sep="\t", low_memory=False)

    
    test1 = pd.read_csv(file_dir + "/test_dataset1.hg38_multianno.txt", sep="\t", low_memory=False)
    test2 = pd.read_csv(file_dir + "/test_dataset2.hg38_multianno.txt", sep="\t", low_memory=False)
    test3 = pd.read_csv(file_dir + "/test_dataset3.hg38_multianno.txt", sep="\t", low_memory=False)
    test4 = pd.read_csv(file_dir + "/test_dataset4.hg38_multianno.txt", sep="\t", low_memory=False)

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

    # 预处理所有数据集
    print("Preprocessing data...")
    data = input_mean_overall(data, candidate_features, "train.hg38_multianno.txt")
    test1 = input_mean_overall(test1, candidate_features, "testset1.hg38_multianno.txt1")
    test2 = input_mean_overall(test2, candidate_features, "testset2.hg38_multianno.txt1")
    test3 = input_mean_overall(test3, candidate_features, "testset3.hg38_multianno.txt1")
    test4 = input_mean_overall(test4, candidate_features, "testset4.hg38_multianno.txt1")

    # 移除常量特征
    print("Removing constant features...")
    candidate_features = remove_constant_features(data, candidate_features)

    # 准备训练集
    print("Preparing training data...")
    X = data[candidate_features]
    Y = data['Otherinfo1']

    # 准备测试集
    test_sets = [
        test1[candidate_features],
        test2[candidate_features],
        test3[candidate_features],
        test4[candidate_features]
    ]
    test_targets = [
        test1['Otherinfo1'],
        test2['Otherinfo1'],
        test3['Otherinfo1'],
        test4['Otherinfo1']
    ]

    # 执行遗传算法特征搜索
    print("\nStarting genetic algorithm feature search...")
    satisfied_models = genetic_algorithm(
        candidate_features, X, Y, test_sets, test_targets,
        best_score
    )

    # 生成结果表格
    results_df = generate_results_table(satisfied_models)

    # 如果没有找到满足条件的模型，提示用户
    if satisfied_models:
        print(f"\nFound {len(satisfied_models)} models meeting all criteria")
    else:
        print("\nNo models met all AUC criteria. Consider adjusting thresholds or features.")

    print("\nFeature search completed!")


if __name__ == "__main__":
    main()
