import argparse
import os
import random

import datasets
import numpy as np
import torch
from model import VarMisUseTrainer, compute_metrics
from preprocess import (DataCollatorForVarMisuse, preprocess_typecheck,
                        tokenize_and_align_labels)
from transformers import (AutoConfig, AutoModelForTokenClassification,
                          AutoTokenizer, EarlyStoppingCallback,
                          TrainingArguments)
from utils import load_dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main(config):
    set_seed(config.seed)
    DATASET_VAL = f"{config.dataset_path}/dev"
    DATASET_TRAIN = f"{config.dataset_path}/train"
    DATASET_TEST = f"{config.dataset_path}/eval"
    TYPECHECK_TRAIN = f"{config.dataset_path}/typecheck/train"
    OVERSAMPLE_PATH = f"{config.dataset_path}/oversample.csv"
    DO_TYPECHECK = config.do_typecheck
    DO_RANDOM_FILTER = config.do_random_filter

    TOKENIZER = AutoTokenizer.from_pretrained(config.model_name, add_prefix_space=True)
    
    if not config.is_baseline:
        MODEL = AutoModelForTokenClassification.from_pretrained(
            config.model_name, num_labels=2
        )
    else:
        model_config = AutoConfig.from_pretrained(config.model_name)
        model_config.num_hidden_layers = config.baseline_layers
        model_config.num_labels = 2
        MODEL = AutoModelForTokenClassification.from_config(model_config)
    
    if config.preprocess:
        num_files = None

        if not DO_TYPECHECK and not DO_RANDOM_FILTER:
            print("Using the full dataset")
            dataset_train = load_dataset(DATASET_TRAIN, num_files=num_files)
        elif DO_RANDOM_FILTER:
            print("Using the random filtered dataset")
            dataset_train = load_dataset(
                DATASET_TRAIN,
                random_split_path=f"{config.dataset_path}/random_filter/train.csv",
                num_files=num_files,
            )
        else:
            print("Using the typecheck filtered dataset")
            dataset_train = load_dataset(
                DATASET_TRAIN, typecheck_path=TYPECHECK_TRAIN, num_files=num_files
            )

        dataset_val = load_dataset(DATASET_VAL, num_files=num_files)
        dataset_test = load_dataset(DATASET_TEST, num_files=num_files)
        dataset = datasets.DatasetDict(
            {"train": dataset_train, "val": dataset_val, "test": dataset_test}
        )

        tokenized_datasets = dataset.map(
            lambda x: tokenize_and_align_labels(x, TOKENIZER),
            batched=True,
            remove_columns=["tokens", "has_bug"],
            num_proc=8,
        )
        tokenized_datasets.save_to_disk("~/autodl/varmisuse-full-unixcoder")
        tokenized_datasets["val"] = tokenized_datasets["val"].shuffle()[:config.max_dev_samples] 
    else:
        tokenized_datasets = datasets.load_dataset(config.huggingface_dataset_path)
        
        if DO_TYPECHECK:
            print("Performing training with typechecking")
            if config.oversample:
                typecheck_dataset_ids = preprocess_typecheck(TYPECHECK_TRAIN, OVERSAMPLE_PATH)
            else:
                typecheck_dataset_ids = preprocess_typecheck(TYPECHECK_TRAIN)

            tokenized_datasets["train"] = tokenized_datasets["train"].select(typecheck_dataset_ids)
            
        
        tokenized_datasets["val"] = tokenized_datasets["validation"].shuffle().select(range(config.max_dev_samples))
    data_collator = DataCollatorForVarMisuse(tokenizer=TOKENIZER)

    if not DO_TYPECHECK:
        model_dir = f"{config.model_name.replace('/', '-')}-finetune"
    else:
        model_dir = f"{config.model_name.replace('/', '-')}-finetune-typecheck"
    print("Saving to ", model_dir)
    args = TrainingArguments(
        model_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        weight_decay=0.01,
        push_to_hub=False,
        logging_steps=100,
        eval_steps=4000,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        load_best_model_at_end=True,
        gradient_accumulation_steps=4,
        metric_for_best_model="classification_accuracy",
        save_total_limit=2,
        save_steps=4000,
    )

    trainer = VarMisUseTrainer(
        model=MODEL,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        data_collator=data_collator,
        tokenizer=TOKENIZER,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    
    # save the last model
    trainer.save_model(f"{model_dir}/last") 

    # predictions = trainer.predict(tokenized_datasets["test"])
    # print(predictions.metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/codebert-base", type=str)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--do_typecheck", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--do_random_filter", action="store_true")
    parser.add_argument("--huggingface_dataset_path", type=str, required=False)
    parser.add_argument("--is_baseline", action="store_true")
    parser.add_argument("--baseline_layers", type=int, default=6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--max_dev_samples", type=int, default=25000)

    args = parser.parse_args()
    main(args)
