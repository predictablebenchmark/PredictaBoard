import argparse  # noqa: D100, INP001

from trl import DataCollatorForCompletionOnlyLM
from unsloth import (
    FastLanguageModel,
    UnslothTrainer,
    UnslothTrainingArguments,
    is_bfloat16_supported,
)
from unsloth.chat_templates import get_chat_template

from utils import create_dataset

def finetune(llm: str, model_hf_path: str = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit") -> None:
    """Finetune model."""
    train_data = create_dataset(llm)["mmlu_pro_train"]

    max_seq_length = 8192
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_hf_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    l_id = tokenizer("<<<<").input_ids[1:]
    r_id = tokenizer(">>>>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template=l_id,
                                               instruction_template=r_id,
                                               tokenizer=tokenizer,
                                               )

    model_name = model_hf_path[model_hf_path.index("/"):]
    # trainer
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_data,
        dataset_text_field = "prompt",
        max_seq_length = max_seq_length,
        dataset_num_proc = 8,
        data_collator=collator,
        args = UnslothTrainingArguments(
            per_device_train_batch_size = 1,
            per_device_eval_batch_size =  1,
            gradient_accumulation_steps = 32,
            warmup_steps = 100,
            num_train_epochs = 10,
            learning_rate = 5e-5,
            embedding_learning_rate = 5e-5 / 10,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            log_level = "info",
            logging_strategy = "steps",
            logging_steps = 1,
            evaluation_strategy = "steps",
            eval_steps = 999999,
            save_strategy = "steps",
            save_steps = 100,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 100,
            output_dir = f"./assessors/llama-fine-tuning/models/{model_name}_{llm}_mmlu_pro_train_set",
        ),
    )

    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()

    trainer.train(resume_from_checkpoint=None)

    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--llm", type=str)
    args = parser.parse_args()

    finetune(args.llm)
