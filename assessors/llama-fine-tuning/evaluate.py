import argparse  # noqa: D100, INP001
import pandas as pd

from unsloth import (
    FastLanguageModel
)

from utils import create_dataset

def evaluate(llm: str, adapter_checkpoint: str) -> None:
    pred_data = create_dataset(llm)
    mmlu_test_data = pred_data["mmlu_pro_test"]
    math_test_data = pred_data["math"]
    ifeval_test_data = pred_data["ifeval"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_checkpoint,
        max_seq_length = 32768,
        dtype = None,
        load_in_4bit = True,
    )

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    model.eval()
    FastLanguageModel.for_inference(model)

    ids = []
    llm_corrects = []
    assessor_corrects = []
    for doc in mmlu_test_data:
        ids.append(doc["id"])
        llm_corrects.append(int(doc["response"]))

        messages = [
            {"role": "user",
             "content": doc["prompt"]},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")
        
        sum_pred = 0
        for i in range(1, 101):
            outputs = model.generate(input_ids = inputs, max_new_tokens = 10, use_cache = True,
                                 temperature = 2.0, min_p = 0.1)
        
            out_string = tokenizer.batch_decode(outputs)[0].replace(doc["prompt"], "")

            if "incorrect" in out_string.lower():
                if doc["response"] == 0:
                    pred = 1
                else:
                    pred = 0
            elif "correct" in out_string.lower():
                if doc["response"] == 0:
                    pred = 0
                else:
                    pred = 1
            else:
                pred = 0

            sum_pred += pred
            mean = sum_pred/i

        assessor_corrects.append(mean)

    mmlu_df = pd.DataFrame({
        "id": ids,
        "llm_correct": llm_corrects,
        "assessor_correct": assessor_corrects,
    })

    mmlu_df.to_csv(f"./results/{llm}_assessor_llama_3.2_1B_mmlu_pro.csv", index=False)

    ids = []
    subsets = []
    llm_corrects = []
    assessor_corrects = []
    for doc in math_test_data:
        ids.append(doc["id"])
        llm_corrects.append(doc["response"])
        subsets.append(doc["subset"])

        messages = [
            {"role": "user",
             "content": doc["prompt"]},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")

        sum_pred = 0
        for i in range(1, 101):
            outputs = model.generate(input_ids = inputs, max_new_tokens = 10, use_cache = True,
                                 temperature = 2.0, min_p = 0.1)
        
            out_string = tokenizer.batch_decode(outputs)[0].replace(doc["prompt"], "")

            if "incorrect" in out_string.lower():
                if doc["response"] == 0:
                    pred = 1
                else:
                    pred = 0
            elif "correct" in out_string.lower():
                if doc["response"] == 0:
                    pred = 0
                else:
                    pred = 1
            else:
                pred = 0

            sum_pred += pred
            mean = sum_pred/i

        assessor_corrects.append(mean)

    math_df = pd.DataFrame({
        "id": ids,
        "subset": subsets,
        "llm_correct": llm_corrects,
        "assessor_correct": assessor_corrects,
    })

    math_df.to_csv(f"./results/{llm}_assessor_llama_3.2_1B_math.csv", index=False)

    ids = []
    llm_corrects = []
    assessor_corrects = []
    for doc in ifeval_test_data:
        ids.append(doc["id"])
        llm_corrects.append(int(doc["response"]))

        messages = [
            {"role": "user",
             "content": doc["prompt"]},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")
        
        sum_pred = 0
        for i in range(1, 101):
            outputs = model.generate(input_ids = inputs, max_new_tokens = 10, use_cache = True,
                                 temperature = 2.0, min_p = 0.1)
        
            out_string = tokenizer.batch_decode(outputs)[0].replace(doc["prompt"], "")

            if "incorrect" in out_string.lower():
                if doc["response"] == 0:
                    pred = 1
                else:
                    pred = 0
            elif "correct" in out_string.lower():
                if doc["response"] == 0:
                    pred = 0
                else:
                    pred = 1
            else:
                pred = 0

            sum_pred += pred
            mean = sum_pred/i

        assessor_corrects.append(mean)

    ifeval_df = pd.DataFrame({
        "id": ids,
        "llm_correct": llm_corrects,
        "assessor_correct": assessor_corrects,
    })

    ifeval_df.to_csv(f"./results/{llm}_assessor_llama_3.2_1B_ifeval.csv", index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str)
    args = parser.parse_args()

    llm_string = args.checkpoint.replace("assessors/llama-fine-tuning/models/Llama-3.2-1B-Instruct-bnb-4bit_", "")
    llm_string = llm_string.replace("_mmlu_pro_train_set/checkpoint-2130", "")

    evaluate(llm_string, args.checkpoint)