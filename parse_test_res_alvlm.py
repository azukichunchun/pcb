"""
Goal
---
1. Read test results from log.txt files
2. Compute mean and std across different folders (seeds)

Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    seed1/
        log.txt
    seed2/
        log.txt
    seed3/
        log.txt

Run the following command from the root directory:

$ python tools/parse_test_res.py output/my_experiment

Add --ci95 to the argument if you wanna get 95% confidence
interval instead of standard deviation:

$ python tools/parse_test_res.py output/my_experiment --ci95

If my_experiment/ has the following structure,

my_experiment/
    exp-1/
        seed1/
            log.txt
            ...
        seed2/
            log.txt
            ...
        seed3/
            log.txt
            ...
    exp-2/
        ...
    exp-3/
        ...

Run

$ python tools/parse_test_res.py output/my_experiment --multi-exp
"""
import re
import numpy as np
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden


def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))


def parse_function(metrics, directory="", args=None, begin_signal=None, end_signal=None):
    print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)

    outputs = []
    output = defaultdict(list)

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")
        assert check_isfile(fpath)
        good_to_go = False
        stop_to_go = False
        
        with open(fpath, "r") as f:
            lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line == begin_signal:
                    good_to_go = True

                if line == end_signal:
                    stop_to_go = True
                
                match = metrics.search(line)    
                if match and good_to_go:
                    if not any([(subdir in path) for path in output["file"]]):
                        output["file"].append(fpath)
                    round = int(match.group(1))
                    num = float(match.group(2))
                    output[round].append(num)

                if stop_to_go:
                    break

    assert len(output) > 0, f"Nothing found in {directory}"

    # for key, value in output.items():
    #     msg = ""
    #     if isinstance(value, float):
    #         msg += f"{key}: {value:.2f}%. "
    #     else:
    #         msg += f"{key}: {value}. "
    #     print(msg)

    output_results = OrderedDict()

    print("===")
    print(f"Summary of directory: {directory}")
    for key, values in output.items():
        if key == "file":
            continue
        avg = np.mean(values)
        std = compute_ci95(values) if args.ci95 else np.std(values)
        print(f"* {key}: {avg:.2f}% +- {std:.2f}%")
        output_results[key] = avg
    print("===")

    return output_results


def main(args, begin_signal, end_signal):
    metric = re.compile(r"(\d+): ([\.\deE+-]+)")
    
    if args.multi_exp:
        final_results = defaultdict(list)

        for directory in listdir_nohidden(args.directory, sort=True):
            directory = osp.join(args.directory, directory)
            results = parse_function(
                metric, directory=directory, args=args, end_signal=end_signal
            )

            for key, value in results.items():
                final_results[key].append(value)

        print("Average performance")
        for key, values in final_results.items():
            avg = np.mean(values)
            print(f"* {key}: {avg:.2f}%")

    else:
        parse_function(
            metric, directory=args.directory, args=args, begin_signal=begin_signal, end_signal=end_signal
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="path to directory")
    parser.add_argument(
        "--ci95", action="store_true", help=r"compute 95\% confidence interval"
    )
    parser.add_argument("--test-log", action="store_true", help="parse test-only logs")
    parser.add_argument(
        "--multi-exp", action="store_true", help="parse multiple experiments"
    )
    parser.add_argument(
        "--keyword", default="accuracy", type=str, help="which keyword to extract"
    )
    args = parser.parse_args()

    begin_signal = "=== Result Overview ==="
    end_signal = "======================="

    main(args, begin_signal, end_signal)
