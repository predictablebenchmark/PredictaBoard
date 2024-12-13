import pandas as pd

from datasets import Dataset

from src.predbench.generate_train_test_split import process_benchmarks

def create_dataset(llm: str) -> Dataset:
    """Combine the metabench datasets into a single dataset."""
    def _process_dataset(doc, test: bool):
        answer = "Correct" if doc["response"] == 1 else "Incorrect"
        if test:
            doc["prompt"] = f"You must predict whether a language model will answer a question correctly.\n\n{doc['prompt']}\n\nDoes the model answer correctly or incorrectly?\n\n"
        else:
            doc["prompt"] = f"You must predict whether a language model will answer a question correctly.\n\n{doc['prompt']}\n\nDoes the model answer correctly or incorrectly?\n\n<<<<{answer}>>>>"
        return doc
    pred_data = process_benchmarks(llm)
    math_sets = ["math_algebra_hard",
                 "math_counting_and_prob_hard",
                 "math_geometry_hard",
                 "math_intermediate_algebra_hard",
                 "math_num_theory_hard",
                 "math_prealgebra_hard",
                 "math_precalculus_hard",
                 ]
    mmlu_pro_train, mmlu_pro_test = Dataset.from_pandas(pred_data["mmlu_pro"]["train"], split="train"), Dataset.from_pandas(pred_data["mmlu_pro"]["test"], split="test")
    ifeval_train, ifeval_test = pred_data["ifeval"]["train"], pred_data["ifeval"]["test"]
    ifeval_set = pd.concat([ifeval_train, ifeval_test], ignore_index=True)
    ifeval_set = Dataset.from_pandas(ifeval_set, split="test")
    math_train, math_test = pred_data[math_sets[0]]["train"], pred_data[math_sets[0]]["test"]
    math_set = pd.concat([math_train, math_test], ignore_index=True)
    math_set['subset'] = math_sets[0]
    for ms in range(1, len(math_sets)):
        math_subset_train = pred_data[math_sets[ms]]["train"]
        math_subset_test = pred_data[math_sets[ms]]["test"]
        math_subset = pd.concat([math_subset_train, math_subset_test], ignore_index=True)
        math_subset['subset'] = math_sets[ms]
        math_set = pd.concat([math_set, math_subset], ignore_index=True)
    math_set = Dataset.from_pandas(math_set, split="test")
    output = {
        "mmlu_pro_train" : mmlu_pro_train.map(_process_dataset, fn_kwargs={"test" : False}),
        "mmlu_pro_test" : mmlu_pro_test.map(_process_dataset, fn_kwargs={"test" : True}),
        "math" : math_set.map(_process_dataset, fn_kwargs={"test" : True}),
        "ifeval" : ifeval_set.map(_process_dataset, fn_kwargs={"test" : True})
    }
    return output