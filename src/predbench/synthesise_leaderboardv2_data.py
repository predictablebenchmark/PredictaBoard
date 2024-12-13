import argparse
import ast
import csv
import glob
import json
import os
import re
import string

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def _load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    results = data.get("results")
    model_name = data.get("model_name")
    return results, model_name

def _clean_results(results):
    try:
        bbh_acc = results.get("leaderboard_bbh").get("acc_norm,none")
        ifeval_acc = results.get("leaderboard_ifeval").get("prompt_level_strict_acc,none")
        math_acc = results.get("leaderboard_math_hard").get("exact_match,none")
        musr_acc = results.get("leaderboard_musr").get("acc_norm,none")
        mmlu_acc = results.get("leaderboard_mmlu_pro").get("acc,none")
        # GPQA is not scraped as prompts are not intended to be publicly available
        return [bbh_acc, ifeval_acc, math_acc, mmlu_acc, musr_acc]
    except:  # noqa: E722
        return []

def load_agg_results(hf_data_folder, save_path):
    files = glob.glob(f'{hf_data_folder}/**/*.json', recursive = True)
    for f in files:
        results, name = _load_json(f)
        if results is not None and name is not None:
            results = _clean_results(results)
            results.append(name)
            match = re.search(r'results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{6})', f)
            results.append(str(match.group(1)))
            csv_path = f"{save_path}/leaderboard_aggregate_results.csv"
            file_exists = os.path.isfile(csv_path)
            if len(results) == 7:
                with open(csv_path, 'a' if file_exists else 'w', newline='') as csv_file:
                    csv_write = csv.writer(csv_file)
                    if not file_exists:
                        csv_write.writerow(['BigBench-Hard', 'IFEval', 'MathLvl5', 'MMLU_Pro', 'MuSR', 'model', 'timestamp'])
                    csv_write.writerow(results)
    print("Complete!")

def _load_model_names(save_path):
    models = []
    stamps = []
    with open(f"{save_path}/leaderboard_aggregate_results.csv") as csvfile:
        csv_read = csv.reader(csvfile)
        for row in csv_read:
            if row[0] != 'model':
                model = row[5].replace("/", "__")
                models.append(model)
                stamps.append(row[6])
    return models, stamps

def _check_model_store(model, stamp, path):
    file_exists = os.path.isfile(path)
    if file_exists:
        with open(path) as csvfile:
            csv_read = csv.reader(csvfile)
            for row in csv_read:
                if row[0] == model and row[1] == stamp:
                    return True
                else:
                    pass
            return False
    else:
        return False


def _load_hf_data(model, stamp, save_path):
    bm_subsets = ["bbh_boolean_expressions", "bbh_causal_judgement", "bbh_date_understanding", "bbh_disambiguation_qa", "bbh_formal_fallacies", "bbh_geometric_shapes", "bbh_hyperbaton", "bbh_logical_deduction_five_objects",
                  "bbh_logical_deduction_seven_objects", "bbh_logical_deduction_three_objects", "bbh_movie_recommendation", "bbh_navigate", "bbh_object_counting", "bbh_penguins_in_a_table", "bbh_reasoning_about_colored_objects",
                  "bbh_ruin_names", "bbh_salient_translation_error_detection", "bbh_snarks", "bbh_sports_understanding", "bbh_temporal_sequences", "bbh_tracking_shuffled_objects_five_objects", "bbh_tracking_shuffled_objects_seven_objects",
                  "bbh_tracking_shuffled_objects_three_objects", "bbh_web_of_lies", 
                  "ifeval", 
                  "math_algebra_hard", "math_counting_and_prob_hard", "math_geometry_hard", "math_intermediate_algebra_hard", "math_num_theory_hard", "math_prealgebra_hard", "math_precalculus_hard", 
                  "mmlu_pro",
                  "musr_murder_mysteries", "musr_object_placements", "musr_team_allocation"]
    print(f"\n\nScraping data for {model} ({stamp})...")
    for subset in tqdm(bm_subsets):
        check = _check_model_store(model, stamp, f"{save_path}/{subset}_results.csv")
        if not check:
            try:
                data = load_dataset(f"open-llm-leaderboard/{model}-details", name=f"{model}__leaderboard_{subset}", split=f"{stamp}")
            except:  # noqa: E722
                try:
                    data = load_dataset(f"open-llm-leaderboard/{model}-details", name=f"{model}__leaderboard_{subset}", split="latest")
                except:  # noqa: E722
                    data = None
                    complete = False
                    print(f"Cannot load data for {model}. Visit https://huggingface.co/datasets/open-llm-leaderboard/{model}-details to ask for access.")
                    break
            prompt_data_path = f"{save_path}/{subset}_prompts.csv"
            prompt_data_exists = os.path.isfile(prompt_data_path)
            if data is not None:
                if not prompt_data_exists:
                    if "ifeval" in subset:
                        prompts = {'id': data['doc_id'], 'prompt': [pr['prompt'] for pr in data['doc']], 'target': data['target']}
                    elif "math" in subset:
                        prompts = {'id': data['doc_id'], 'prompt': [pr['problem'] for pr in data['doc']], 'target': data['target']}
                    elif "musr" in subset:
                        choices = [ast.literal_eval(pr['choices']) for pr in data['doc']]
                        choices = ["".join([f"{i+1} - {choice}\n" for i, choice in enumerate(c)]) + 'Answer:' for c in choices]
                        prompts = {'id': data['doc_id'], 'prompt': [pr['narrative'] + "\n\n" + pr["question"] + "\n\n" + choices[i] for i, pr in enumerate(data['doc'])], 'target': data['target']}
                    elif "mmlu" in subset:
                        choices = [ast.literal_eval(str(pr['options'])) for pr in data['doc']]
                        choices = ["".join([f"{string.ascii_uppercase[i]}. {choice}\n" for i, choice in enumerate(c)]) + 'Answer:' for c in choices]
                        prompts = {'id': data['doc_id'], 'prompt': [pr['question'] + "\n" + choices[i] for i, pr in enumerate(data['doc'])], 'target': data['target']}
                    elif "bbh" in subset:
                        prompts = {'id': data['doc_id'], 'prompt': [pr['input'] for pr in data['doc']], 'target': data['target']}
                    else:
                        raise ValueError("Benchmark subset not recognised.")
                    dataframe = pd.DataFrame(prompts)
                    dataframe.to_csv(prompt_data_path, index=False)
                llm_results_path = f"{save_path}/{subset}_results.csv"
                llm_results_exists = os.path.isfile(llm_results_path)
                with open(llm_results_path, 'a' if llm_results_exists else 'w', newline='') as csv_file:
                    csv_write = csv.writer(csv_file)
                    if not llm_results_exists:
                        title_row = data['doc_id']
                        title_row.insert(0, "model")
                        title_row.insert(1, "timestamp")
                        csv_write.writerow(title_row)
                    if "ifeval" in subset:
                        llm_row = [int(i) for i in data['prompt_level_strict_acc']]
                    elif "math" in subset:
                        llm_row = data['exact_match']
                    elif "mmlu" in subset:
                        llm_row = [int(i) for i in data['acc']]
                    else:
                        llm_row = [int(i) for i in data['acc_norm']]
                    llm_row.insert(0, model)
                    llm_row.insert(1, stamp)
                    csv_write.writerow(llm_row)
                complete = True
            else:
                complete = False
                break
    else:
        complete = True
    return complete

def aggregate_instance_results(save_path, limit=10):
    models, stamps = _load_model_names(save_path)
    count = 0
    for (m, s) in zip(models, stamps):
        completed = _load_hf_data(m, s, save_path)
        if not completed:
            count += 1
        if count >= limit:
            print(f"You have reached the limit of {limit} inaccess errors. Please gain access to the relevant repositories and try again.")
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outputdir', type=str, default="./data/open-llm-leaderboard-v2")
    parser.add_argument('-d', '--datadir', type=str, default="../hf_files/")
    parser.add_argument('-l', '--limit', type=int, default=10)
    parser.add_argument('-r', '--reaggregate', action='store_true')
    args = parser.parse_args()
    if args.reaggregate:
        print("Loading aggregated leaderboard results!")
        if os.path.isfile(f"{args.outputdir}/leaderboard_aggregate_results.csv"):
            os.remove(f"{args.outputdir}/leaderboard_aggregate_results.csv")
        load_agg_results(args.datadir, args.outputdir)
    aggregate_instance_results(args.outputdir, limit=args.limit)

    print("\n\n\n Be sure to clear out your ~/.cache. All the datasets are stored there, and it can be a bit gross to leave it all unchecked.")

if __name__ == '__main__':
    main()


