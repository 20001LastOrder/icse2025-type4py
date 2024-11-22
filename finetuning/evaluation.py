from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from preprocess import DataCollatorForVarMisuse, tokenize_and_align_labels
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer
from utils import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser()

parser.add_argument("--model_name", type=str)
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--output_filename", type=str)

config = parser.parse_args()
model_name = config.model_name
DATASET_PATH = config.dataset_path
MODEL_PATH = config.model_path
OUTPUT_FILENAME = config.output_filename

save_every_n_epoch = 1000
TOKENIZER = AutoTokenizer.from_pretrained(
    f"microsoft/{model_name}-base", add_prefix_space=True
)
MODEL = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, num_labels=2)

dataset_test = load_dataset(DATASET_PATH, max_samples=10000)
tokenized_datasets = dataset_test.map(
    lambda x: tokenize_and_align_labels(x, TOKENIZER),
    batched=True,
    remove_columns=["tokens", "has_bug", "error_location", "repair_locations"],
    # num_proc=8
)

all_word_ids = [word_ids for word_ids in tokenized_datasets["word_ids"]]
tokenized_datasets = tokenized_datasets.remove_columns("word_ids")

data_collator = DataCollatorForVarMisuse(tokenizer=TOKENIZER)

dataloader = DataLoader(
    tokenized_datasets,
    batch_size=64,
    collate_fn=data_collator.torch_call,
    shuffle=False,
    # num_workers=8
)


MODEL = MODEL.to(device)
MODEL.eval()
softmax = torch.nn.Softmax(dim=1)
results = []
count = 0
epoch_counter = 0
with torch.no_grad():
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        batch_size = batch.input_ids.shape[0]

        labels = batch.pop("labels")
        outputs = MODEL(**batch)

        # classification prediction
        has_bug = outputs.logits[:, 0, 0] > 0.5

        # localization prediction
        localization_labels = labels[:, 1:, 0]
        localization_logits = outputs.logits[:, 1:, 0]
        localization_logits[localization_labels == -100] = float("-inf")

        localizations_pred = torch.argmax(localization_logits, axis=1)
        localizations_label = torch.argmax(localization_labels, axis=1)

        # repair prediction
        repair_labels = labels[:, 1:, 1]
        repair_logits = outputs.logits[:, 1:, 1]

        repair_logits[repair_labels == -100] = float("-inf")
        repair_logits = softmax(repair_logits)
        repair_pred = torch.argmax(repair_logits, axis=1)
        repair_prob = (repair_logits * (repair_labels == 1).float()).sum(axis=1)

        for i in range(batch_size):
            word_ids = all_word_ids[count]
            result = {}
            if not has_bug[i]:
                result["loc_pred"] = 0
            else:
                result["loc_pred"] = word_ids[localizations_pred[i].item()]
                if result["loc_pred"]:
                    # considers both the first CLS token and the new line token
                    result["loc_pred"] += 2
                else:
                    result["loc_pred"] = 0
            # result["target_probs"] = repair_prob[i].item()
            # result["repair_pred"] = repair_pred[i].item()
            result["gt_error_loc"] = word_ids[localizations_label[i].item()]

            if result["gt_error_loc"]:
                result["gt_error_loc"] += 2
            else:
                result["gt_error_loc"] = 0
            results.append(result)
            count += 1
        epoch_counter += 1
        if epoch_counter % save_every_n_epoch == 0:
            df = pd.DataFrame(results)
            df.to_csv(OUTPUT_FILENAME)


df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILENAME)
