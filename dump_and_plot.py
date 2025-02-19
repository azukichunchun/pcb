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

datasets = ["oxford_flowers", "dtd", "oxford_pets", "eurosat", "caltech101", "stanford_cars", "fgvc_aircraft",]
#datasets = ["eurosat"]
trainers = ["ALVLM_PromptSRC_random","ALVLM_PromptSRC_entropy", "ALVLM_PromptSRC_badge", "ALVLM_PromptSRC_badge_curriculum"]
trainers = ["ALVLM_MaPLe_random","ALVLM_MaPLe_entropy", "ALVLM_MaPLe_badge", "ALVLM_MaPLe_coreset", "ALVLM_MaPLe_random_curriculum"]
trainers = ["ALVLM_CoCoOp_random","ALVLM_CoCoOp_entropy", "ALVLM_CoCoOp_badge", "ALVLM_CoCoOp_coreset"]
#trainers = ["ALVLM_random","ALVLM_entropy", "ALVLM_badge", "ALVLM_coreset"]

begin_signal = "=== Result Overview ==="
end_signal = "======================="


# accuracyを抽出する関数
def extract_accuracy_from_log(file_path):
    good_to_go = False
    stop_to_go = False
    output = dict()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == begin_signal:
                good_to_go = True
            if line == end_signal:
                stop_to_go = True
            
            match = re.compile(r"(\d+): ([\.\deE+-]+)").search(line) # 1: 0.1234
            if match and good_to_go:
                round = int(match.group(1))
                accuracy = float(match.group(2))
                output[round] = accuracy
            
            if stop_to_go:
                break

    return output
    

# main関数
def main():
    for trainer in trainers:
        print(f"##{trainer}##")
        res = []
        agg_ave = []
        for dataset in datasets:
            #print(f"###{dataset}###")
            base_paths = {
                "train_base": f"output/base2new/train_base/{dataset}/",
                "test_new": f"output/base2new/test_new/{dataset}/"
            }
        
            accuracies = defaultdict(lambda: defaultdict(list))
            for style, base_path in base_paths.items():
                for shot in shots:
                    for seed in seeds:
                        if "curriculum" in trainer:
                            cfg = "vit_b16_one_time_miru2025_curriculum_phase_1"
                            trainer_for_path  = trainer.replace("_curriculum", "")
                        else:
                            cfg = "vit_b16_one_time_miru2025"
                            trainer_for_path = trainer
                        log_path = os.path.join(base_path, shot, trainer_for_path , cfg, seed, "log.txt")
                        if os.path.exists(log_path):
                            output = extract_accuracy_from_log(log_path)
                            if output is not None:
                                for rnd, acc in output.items():
                                    accuracies[rnd][style].append(acc)
                        else:
                            print(f"Could not find accuracy in {log_path}")

            # calculate harmonic means
            for rnd, accs in accuracies.items():
                for acc_base, acc_new in zip(accs["train_base"], accs["test_new"]):
                    hmean_val = hmean([acc_base, acc_new])
                    accuracies[rnd]["harmonic_mean"].append(hmean_val)

            # print results
            for rnd, accs in accuracies.items():
                for style, vals in accs.items():
                    avg = np.mean(vals)
                    std = np.std(vals)
                    if style == "harmonic_mean" and rnd == 7:
                    #if style == "harmonic_mean":
                        agg_ave.append(avg)
                        #print(f"{rnd} : {avg:.2f} +- {std:.2f}")
                        backslash="\\"
                        res.append((f"{avg:.2f}{backslash}small{{${backslash}pm${std:.2f}}}"))
        print(" & ".join(res) + f" & {np.mean(agg_ave):.2f}")

# スクリプトを実行
main()
