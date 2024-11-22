import glob
import os
import random
import tempfile

import datasets
import orjson
import pandas as pd
import tensorflow as tf
import torch
from datasets import Features, Sequence, Value
from tqdm import tqdm
import json

# Edge types to be used in the models, and their (renumbered) indices -- the data files contain
# reserved indices for several edge types that do not occur for this problem (e.g. UNSPECIFIED)
EDGE_TYPES = {
    "enum_CFG_NEXT": 0,
    "enum_LAST_READ": 1,
    "enum_LAST_WRITE": 2,
    "enum_COMPUTED_FROM": 3,
    "enum_RETURNS_TO": 4,
    "enum_FORMAL_ARG_NAME": 5,
    "enum_FIELD": 6,
    "enum_SYNTAX": 7,
    "enum_NEXT_SYNTAX": 8,
    "enum_LAST_LEXICAL_USE": 9,
    "enum_CALLS": 10,
}


class DataLoader:
    def __init__(self, data_path, data_config, vocabulary):
        self.data_path = data_path
        self.config = data_config
        self.vocabulary = vocabulary

        if "typechecking_path" in self.config:
            self.typechecking_path = self.config["typechecking_path"]
        else:
            self.typechecking_path = False

    def batcher(self, mode="train"):
        def parse_sample(sample):
            sample["edges"] = [
                [2 * EDGE_TYPES[rel[3]], rel[0], rel[1]]
                for rel in sample["edges"]
                if rel[3] in EDGE_TYPES
            ]

            sample["repair_candidates"] = [
                t for t in sample["repair_candidates"] if isinstance(t, int)
            ]

            del obj["provenances"]
            del obj["bug_kind_name"]
            del obj["bug_kind"]
            del obj["has_bug"]

            return sample

        data_path = self.get_data_path(mode)
        file_names = glob.glob(data_path + "/*.txt*")

        if self.typechecking_path is not None:
            typecheck_files = sorted(glob.glob(self.typechecking_path + "/*.csv"))

            typecheck_dataset = []
            for filename in typecheck_files:
                typecheck_dataset.append(
                    pd.read_csv(filename, index_col=False)
                )

            typecheck_dataset = pd.concat(typecheck_dataset)
            typecheck_dataset.fillna("", inplace=True)
            typecheck_dataset = typecheck_dataset.to_dict(orient="records")
        else:
            typecheck_dataset = None

        print("Load files...")
        dataset = []
        sample_id = 0
        _, filename = tempfile.mkstemp()
        with open(filename, "w") as fwrite:
            for file_name in tqdm(file_names):
                with open(file_name, "r") as f:
                    for line in f.readlines():
                        obj = orjson.loads(line)
                        obj["sample_id"] = sample_id
                        sample_id += 1

                        if len(obj["source_tokens"]) > self.config["max_sequence_length"]:
                            continue

                        if mode != "dev" and typecheck_dataset is not None:
                            if typecheck_dataset[obj["sample_id"]]["type_check_failed"]:
                                continue

                        if mode == 'dev' and sample_id >= self.config["max_valid_samples"]:
                            break

                        obj = parse_sample(obj)
                        fwrite.write(json.dumps(obj) + "\n")

        
            # for sample in tqdm(dataset):
                
        dataset = datasets.load_dataset("json", data_files=filename)["train"]

        dataset = dataset.map(
            self.to_sample, num_proc=8, remove_columns=["sample_id", "source_tokens"], cache_file_name=f"cache/{mode}_cache.json"
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=48,
            shuffle=mode == "train",
            num_workers=8,
            collate_fn=self.make_batch_torch,
        )

        return dataloader

    def convert_data(self, sample):
        return self.to_sample(orjson.loads(sample))

    def get_data_path(self, mode):
        if mode == "train":
            return os.path.join(self.data_path, "train")
        elif mode == "dev":
            return os.path.join(self.data_path, "dev")
        elif mode == "eval":
            return os.path.join(self.data_path, "eval")
        else:
            raise ValueError(
                'Mode % not supported for batching; please use "train", "dev", or "eval".'
            )

    # Creates a simple Python sample from a JSON object.
    def to_sample(self, json_data):
        def parse_edges(edges):
            # Reorder edges to [edge type, source, target] and double edge type index to allow reverse edges
            # relations = [
            #     [2 * EDGE_TYPES[rel[3]], rel[0], rel[1]]
            #     for rel in edges
            #     if rel[3] in EDGE_TYPES
            # ]  # Note: we reindex edge types to be 0-based and filter unsupported edge types (useful for ablations)
            relations = edges
            relations += [
                [rel[0] + 1, rel[2], rel[1]] for rel in relations
            ]  # Add reverse edges
            return relations

        tokens = [
            self.vocabulary.translate(t)[: self.config["max_token_length"]]
            for t in json_data["source_tokens"]
        ]
        for sublist in tokens:
            sublist.extend([0] * (self.config["max_token_length"] - len(sublist)))

        edges = parse_edges(json_data["edges"])
        error_location = json_data["error_location"]
        repair_targets = json_data["repair_targets"]
        repair_candidates = json_data["repair_candidates"]

        return {
            "tokens": tokens,
            "edges": edges,
            "error_location": error_location,
            "repair_targets": repair_targets,
            "repair_candidates": repair_candidates,
        }

    def make_batch_torch(self, batch):
        # Pad all tokens to max length
        max_rows = max(len(sample["tokens"]) for sample in batch)
        num_columns = len(batch[0]["tokens"][0])
        tokens = torch.zeros((len(batch), max_rows, num_columns), dtype=torch.long)
        for i, sample in enumerate(batch):
            tokens[i, : len(sample["tokens"]), :] = torch.tensor(sample["tokens"])

        batch_dim = len(batch)
        edge_batches = torch.repeat_interleave(
            torch.arange(batch_dim),
            torch.tensor([len(sample["edges"]) for sample in batch]),
        )
        edge_tensor = torch.cat([torch.tensor(sample["edges"]) for sample in batch], dim=0)
        edge_tensor = torch.stack(
            [edge_tensor[:, 0], edge_batches, edge_tensor[:, 1], edge_tensor[:, 2]],
            dim=1,
        )
        
        # Error location is just a simple constant list
        error_location = torch.tensor([sample["error_location"] for sample in batch], dtype=torch.long)

        # Targets and candidates both have an added batch dimension as well, and are otherwise just a list of indices
        target_batches = torch.repeat_interleave(
            torch.arange(batch_dim),
            torch.tensor([len(sample["repair_targets"]) for sample in batch]),
        )

        repair_targets = torch.cat([torch.tensor(sample["repair_targets"]) for sample in batch], dim=0)
        repair_targets = torch.stack([target_batches, repair_targets], dim=1)

        candidates_batches = torch.repeat_interleave(
            torch.arange(batch_dim),
            torch.tensor([len(sample["repair_candidates"]) for sample in batch]),
        )
        repair_candidates = torch.cat([torch.tensor(sample["repair_candidates"]) for sample in batch], dim=0)
        repair_candidates = torch.stack([candidates_batches, repair_candidates], dim=1)

        return {
            "tokens": tokens,
            "edges": edge_tensor,
            "error_location": error_location,
            "repair_targets": repair_targets,
            "repair_candidates": repair_candidates,
        }

def sample_len(sample):
    return len(sample[0])
