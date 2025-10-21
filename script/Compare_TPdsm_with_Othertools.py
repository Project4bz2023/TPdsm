import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Binarizer
import seaborn as sns
import pickle,os,json
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pickle
from joblib import load 
from sklearn.metrics import accuracy_score,precision_recall_curve, average_precision_score, precision_score, f1_score, matthews_corrcoef, multilabel_confusion_matrix, log_loss, roc_curve, auc, recall_score,classification_report, confusion_matrix

model_dir=Path("/path/to/model/")
# 加载模型配置
with open(model_dir / 'feature.json', 'r', encoding='utf-8') as f:
    trainFeatures = json.load(f)
print(trainFeatures)

model = load(model_dir / 'TPdsm.pkl')   

###inputation function
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



file_dir="/path/to/files/"
data=pd.read_table(file_dir+'/train_dataset.hg38_multianno.txt', low_memory=False)
test1=pd.read_table(file_dir+'/test_dataset1.hg38_multianno.txt', low_memory=False)

test2=pd.read_table(file_dir+'/test_dataset2.hg38_multianno.txt', low_memory=False)

test3=pd.read_table(file_dir+'/test_dataset3.hg38_multianno.txt', low_memory=False)
test4=pd.read_table(file_dir+'/test_dataset4.hg38_multianno.txt', low_memory=False)

print("Preprocessing data...")
#trainFeatures=[
#"gnomad41_exome_faf99", "delta_score", "silva_rankscore", "silva", "cadd_mapability_20bp", "delta_psi_max", "delta_score_rankscore", "gnomad41_exome_AF_eas", "CADD_PHRED", "gnomad41_exome_AF_asj", "#RSCU", "syntool_rankscore", "MES-KM?", "gnomad41_genome_fafmax_faf99_max", "ExAC_FIN", "gerp_gt2", "#MES", "CpG_exon", "ExAC_OTH", "gnomad41_genome_AF_eas", "gnomad41_exome_fafmax_faf99_max", "SR+"
#]
CompareList=[
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

Condidate_features=list(set(trainFeatures+CompareList))
data = input_mean_overall(data, Condidate_features, "train.hg38_multianno.txt")
test1 = input_mean_overall(test1, Condidate_features, "testset1.hg38_multianno.txt1")
test2 = input_mean_overall(test2, Condidate_features, "testset2.hg38_multianno.txt1")
test3 = input_mean_overall(test3, Condidate_features, "testset3.hg38_multianno.txt1")
test4 = input_mean_overall(test4, Condidate_features, "testset4.hg38_multianno.txt1")


modelname="SYNmethod_compare_TPdsm"
result="SYNmethod_compare_TPdsm"
os.makedirs(result,exist_ok=True)


 # 使用模型进行预测
colorsn = sns.color_palette("husl", 37)

colors = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#008000', '#000080',
    '#808000', '#800080', '#008080', '#C0C0C0', '#808080', '#FFA500', '#FFC0CB', '#008000', '#FF69B4',
    '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00', '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC',
    '#DC143C', '#00BFFF', '#00FA9A', '#BDB76B', '#9932CC', '#8B4513'
        ]

cmaps = [plt.get_cmap('tab20'), plt.get_cmap('tab20b'), plt.get_cmap('tab20c')]
color60 = []
for cmap in cmaps:
    color60.extend(cmap(range(20)))
print(color60)

def predict(model, data, features):
    model_pred = model.predict_proba(data[features])
    print(model_pred)
    return model_pred[:,1]

def predictBinary(model, data, features):
    model_pred = model.predict(data[features])
    print(model_pred)
    return model_pred
test1["TPdsm"] = predict(model, test1, trainFeatures)
test2["TPdsm"] = predict(model, test2, trainFeatures)
test3["TPdsm"] = predict(model, test3, trainFeatures)
test4["TPdsm"] = predict(model, test4, trainFeatures)
#y_pred_proba = predict(model, Step_test1_1, trainFeatures)

tag_dic={
    "TPdsm": "TPdsm",
    "CADD_RawScore": "CADD",
    "DANN": "DANN",
    "DDIG": "DDIG",
    "eigen": "Eigen",
    "EnDSM": "EnDSM",
    "fathmm_MKL_coding": "Fathmm_MKL_coding",
    "fathmm_xf_coding": "Fathmm_XF_coding",
    "frDSM": "frDSM",
    "PhD_SNPg": "PhD_SNPg",
    "PrDSM": "PrDSM",
    "silva": "SilVA",
    "syntool": "Syntool",
    "usDSM": "usDSM"
}
def pltRoc(y, pred_y, tag, n): 
    if tag == "TPdsm":
        fpr, tpr, thresholds_roc = roc_curve(y, pred_y, pos_label=1, drop_intermediate=False)
    #plt.figure(figsize=(10,10.5))
        plt.plot(fpr, tpr, color=n, lw=2, label=tag_dic[tag]+'(AUC=%0.3f)' % auc(fpr, tpr))
    else:
        y = np.array(y)
        pred_y = np.array(pred_y)
     
    # 找出 pred_y 中既不是 np.nan 也不是字符 "." 的索引
    #valid_indices = ~((pred_y == ".") | np.isnan(pred_y))
        if pred_y.dtype == 'object':
            valid_indices = ~(pred_y == ".")
        elif np.issubdtype(pred_y.dtype, np.floating):
            valid_indices = ~np.isnan(pred_y)
        else:
            raise ValueError("pred_y 的数据类型不被支持")

    # 根据有效索引筛选 y 和 pred_y
        y = y[valid_indices]
        pred_y = pred_y[valid_indices]
        #print(pred_y.shape)
        pred_y = np.asarray(pred_y, dtype=np.float64)
        #print(type(pred_y))
        y=np.asarray(y, dtype=np.float64)
        valid_count = np.sum(valid_indices)
        print(f"{tag} 中为bu NA 或字符bu '.' 的总行数: {valid_count}")
        fpr, tpr, thresholds_roc = roc_curve(y, pred_y, pos_label=1, drop_intermediate=False)
    #plt.figure(figsize=(10,10.5))
        plt.plot(fpr, tpr, color=n, lw=2, label=tag_dic[tag]+'(AUC=%0.3f)' % auc(fpr, tpr))
   



def pltPrc(y, pred_y, tag, n): 

    if tag == "TPdsm":
        precision, recall, thresholds_prc = precision_recall_curve(y, pred_y)
        average_precision = average_precision_score(y, pred_y)
    #print(f"AUPRC (Average Precision Score)_{tag}: {average_precision:.4f}")
    #plt.figure(figsize=(10,10.5))
        plt.step(recall, precision,lw=2, color=(n),
            label=(tag_dic[tag] + '(AUPRC={0:0.3f})'.format(average_precision)), where='post')

    else:
        y = np.array(y)
        pred_y = np.array(pred_y)

    # 找出 pred_y 中既不是 np.nan 也不是字符 "." 的索引
    #valid_indices = ~((pred_y == ".") | np.isnan(pred_y))
        if pred_y.dtype == 'object':
            valid_indices = ~(pred_y == ".")
        elif np.issubdtype(pred_y.dtype, np.floating):
            valid_indices = ~np.isnan(pred_y)
        else:
            raise ValueError("pred_y 的数据类型不被支持")

    # 根据有效索引筛选 y 和 pred_y
        y = y[valid_indices]
        pred_y = pred_y[valid_indices]
        pred_y = np.asarray(pred_y, dtype=np.float64)
        y=np.asarray(y, dtype=np.float64)
        valid_count = np.sum(valid_indices)
        print(f"{tag} 中不为 NA 或字符bu '.' 的总行数: {valid_count}")
        precision, recall, thresholds_prc = precision_recall_curve(y, pred_y)
        average_precision = average_precision_score(y, pred_y)
    #print(f"AUPRC (Average Precision Score)_{tag}: {average_precision:.4f}")
    #plt.figure(figsize=(10,10.5))
        plt.step(recall, precision,lw=2, color=(n),
            label=(tag_dic[tag] + '(AUPRC={0:0.3f})'.format(average_precision)), where='post')
  
   

def Roc(dfTest, compareList,cl, file):
    plt.figure(figsize=(10,10.5))
    for index,i in enumerate(compareList):
        pltRoc(dfTest['Otherinfo1'], dfTest[i],i, cl[index])
    plt.plot([0, 1], [0, 1], color='silver', lw=2, linestyle='--')
    plt.xlim([0.0, 1.00])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate', size = 16)
    plt.ylabel('True positive rate', size = 16)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
    plt.xticks(size = 16)
    plt.yticks(size = 16)
    plt.locator_params()
    plt.savefig(file)
    plt.close()

def Prc(dfTest, compareList,cl, file):
    plt.figure(figsize=(10,10.5))
    for index,i in enumerate(compareList):
        pltPrc(dfTest['Otherinfo1'], dfTest[i],i, cl[index])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', size=16)
    plt.ylabel('Precision', size=16)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
    plt.xticks(size = 16)
    plt.yticks(size = 16)
    plt.locator_params()
    #plt.savefig(file)
    plt.savefig(file)
    plt.close()



CompareList_N=CompareList.copy()
CompareList_N.append('TPdsm')

Roc(test1,CompareList_N,color60,result+"/"+modelname+'_ROCintest1.pdf')
Prc(test1,CompareList_N,color60,result+"/"+modelname+'_PRCintest1.pdf')

Roc(test2,CompareList_N,color60,result+"/"+modelname+'_ROCintest2.pdf')
Prc(test2,CompareList_N,color60,result+"/"+modelname+'_PRCintest2.pdf')

Roc(test3,CompareList_N,color60,result+"/"+modelname+'_ROCintest3.pdf')
Prc(test3,CompareList_N,color60,result+"/"+modelname+'_PRCintest3.pdf')

Roc(test4,CompareList_N,color60,result+"/"+modelname+'_ROCintest4.pdf')
Prc(test4,CompareList_N,color60,result+"/"+modelname+'_PRCintest4.pdf')   

  
