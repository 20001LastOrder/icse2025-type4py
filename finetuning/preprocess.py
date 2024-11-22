import glob

import networkx as nx
import pandas as pd
from transformers import DataCollatorForTokenClassification


def generate_nx_graph(data):
    # we ignore the first token
    g = nx.DiGraph()
    g.graph["has_bug"] = data["has_bug"]
    g.graph["repair_candidates"] = [j - 1 for j in data["repair_candidates"]]
    g.graph["error_location"] = data["error_location"] - 1
    g.graph["repair_targets"] = [j - 1 for j in data["repair_targets"]]
    for j, t in enumerate(data["source_tokens"][1:]):
        g.add_node(j, token=t)
    for source, target, t, _ in data["edges"]:
        g.add_edge(source - 1, target - 1, type=t)
    return g


def generate_data_token_classification(data):
    tokens = data["source_tokens"][1:]
    has_bug = data["has_bug"]
    error_location = [0] * len(tokens)
    if has_bug:
        error_location[data["error_location"] - 1] = 1
    repair_locations = [-100] * len(tokens)
    if has_bug:
        for j in data["repair_candidates"]:
            repair_locations[j - 1] = 0
        for j in data["repair_targets"]:
            repair_locations[j - 1] = 1
    return {
        "tokens": tokens,
        "has_bug": has_bug,
        "error_location": error_location,
        "repair_locations": repair_locations,
    }


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = -100  # labels[word_id]
            new_labels.append(label)
    return new_labels


def tokenize_and_align_labels(examples, tokenizer, max_length=512):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=max_length
    )
    error_location = examples["error_location"]
    repair_locations = examples["repair_locations"]
    has_bug = examples["has_bug"]
    all_labels = []
    for i, (error, h, repair) in enumerate(
        zip(error_location, has_bug, repair_locations)
    ):
        word_ids = tokenized_inputs.word_ids(i)
        alignment = align_labels_with_tokens(error, word_ids)
        alignment[0] = 1 if h else 0
        alignment_repair = align_labels_with_tokens(repair, word_ids)
        all_labels.append([alignment, alignment_repair])

    tokenized_inputs["labels"] = all_labels
    tokenized_inputs["word_ids"] = [
        tokenized_inputs.word_ids(i) for i in range(len(all_labels))
    ]
    return tokenized_inputs


def typecheck_filter(example):
    return not example["type_check_failed"]


class DataCollatorForVarMisuse(DataCollatorForTokenClassification):
    def torch_call(self, features):
        import torch
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )

        no_labels_features = [
            {k: v for k, v in feature.items() if k != label_name}
            for feature in features
        ]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                [
                    to_list(l1)
                    + [self.label_pad_token_id] * (sequence_length - len(l1)),
                    to_list(l2)
                    + [self.label_pad_token_id] * (sequence_length - len(l2)),
                ]
                for l1, l2 in labels
            ]
        else:
            batch[label_name] = [
                [
                    [self.label_pad_token_id] * (sequence_length - len(l1))
                    + to_list(l1),
                    [self.label_pad_token_id] * (sequence_length - len(l2))
                    + to_list(l2),
                ]
                for l1, l2 in labels
            ]

        batch[label_name] = torch.permute(
            torch.tensor(batch[label_name], dtype=torch.float32), (0, 2, 1)
        )
        return batch

    
def preprocess_typecheck(typecheck_path=None, oversampling_file=None):
    if typecheck_path is not None:
        typecheck_files = sorted(glob.glob(typecheck_path + "/*.csv"))

        typecheck_failed = []
        for filename in typecheck_files:
            df = pd.read_csv(filename)
            typecheck_failed.extend(df["type_check_failed"].tolist())

        if oversampling_file is not None:
            over_sampling_data = pd.read_csv(oversampling_file)["ids"].tolist()
        else:
            over_sampling_data = []
            
        included_ids = []
        for idx, typecheck_label in enumerate(typecheck_failed):
            if not typecheck_label:
                included_ids.append(idx)
        
        included_ids.extend(over_sampling_data)
        
        return included_ids