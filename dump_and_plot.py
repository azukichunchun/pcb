import os
import re
import sys
import pdb
import numpy as np
from collections import defaultdict
from scipy.stats import hmean
import matplotlib.pyplot as plt

args = sys.argv

# ディレクトリ構造の基本パス
shots = ["shots_-1"]
seeds = ["seed1", "seed2","seed3"]
trainers = ["ALVLM_CoCoOp_clustering_one_sample"]
datasets = ["eurosat", "dtd", "caltech101",
            "oxford_pets", "oxford_flowers", "fgvc_aircraft", "stanford_cars"]

cfg = "vit_b16_one_time_one_sample"

# accuracyを抽出する関数
def extract_accuracy_from_log(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        # 正規表現を使用してaccuracyを抽出
        match = re.search(r'accuracy: (\d+\.\d+)%', content)
        if match:
            return float(match.group(1))
    return None

# main関数
def main():
    summary = dict()
    differences = {}
    for dataset in datasets:
        print(f"###{dataset}###")
        base_paths = [
            f"output/base2new/train_base/{dataset}/",
            f"output/base2new/test_new/{dataset}/"
            ]
        
        fullres = dict()    
        dump = []

        for trainer in trainers:
            res = defaultdict(list)
            print(f"##{trainer}##")
            
            for base_path in base_paths:
                for shot in shots:
                    accuracies = []
                    for seed in seeds:
                        log_path = os.path.join(base_path, shot, trainer, cfg, seed, "log.txt")
                        if os.path.exists(log_path):
                            accuracy = extract_accuracy_from_log(log_path)
                            if accuracy is not None:
                                #print(f"{log_path} -> accuracy: {accuracy}%")
                                accuracies.append(accuracy)
                            else:
                                print(f"Could not find accuracy in {log_path}")

                    acc_mean = np.round(np.mean(accuracies),2)
                    acc_std = np.round(np.std(accuracies), 2)
                    
                    print(f"{base_path.split('/')[2]},{dataset},{trainer},{cfg},{shot}: -> ${acc_mean}_{{{acc_std}}}$")    
                    res[base_path.split('/')[2]+"_means"].append(acc_mean)
                    res[base_path.split('/')[2]+"_errors"].append(acc_std)
            
            harmonic_means_elementwise = [hmean([res['test_new_means'][i], res['train_base_means'][i]]) for i in range(len(res['test_new_means']))]
            for i, v in enumerate(harmonic_means_elementwise):
                print(f"{trainer}, {shots[i]}: {round(v, 2)}")
            
            res["H"] = harmonic_means_elementwise

            fullres[trainer] = res
                
        summary[dataset] = fullres
    

# スクリプトを実行
main()
