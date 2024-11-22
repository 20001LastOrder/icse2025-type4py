import glob
import json
import os
import random

import pandas as pd
import tensorflow as tf

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
    "enum_HYPER": 11,
}


class DataLoader:

    def __init__(self, data_path, data_config, vocabulary):
        self.data_path = data_path
        self.config = data_config
        self.vocabulary = vocabulary

        self.typecheck_dataset = None
        self.random_filter_dataset = None

        if self.config.get("typechecking", False):
            data_path = os.path.join(self.data_path, "typecheck/train/")
            files = sorted(glob.glob(data_path + "*.csv"))
            typecheck_dataset = []
            for filename in files:
                typecheck_dataset.append(pd.read_csv(filename, index_col=False))
            typecheck_dataset = pd.concat(typecheck_dataset)
            typecheck_dataset.fillna("", inplace=True)
            typecheck_dataset = typecheck_dataset.to_dict(orient="records")
            print(f"Loaded {len(typecheck_dataset)} type checking records")

            self.typecheck_dataset = typecheck_dataset

        if self.config.get("random_filter", False):
            data_path = os.path.join(self.data_path, "random_filter/train.csv")
            random_filter_dataset = pd.read_csv(data_path)
            random_filter_dataset = random_filter_dataset.to_dict(orient="records")
            print(f"Loaded {len(random_filter_dataset)} random filtering records")

            self.random_filter_dataset = random_filter_dataset

    def batcher(self, mode="train"):
        data_path = self.get_data_path(mode)
        dataset = tf.data.Dataset.list_files(
            data_path + "/*.json*", shuffle=mode != "eval", seed=42
        )
        dataset = dataset.interleave(
            lambda x: (
                tf.data.TextLineDataset(x, num_parallel_reads=8).shuffle(
                    buffer_size=100000, seed=42
                )
                if mode != "eval"
                else tf.data.TextLineDataset(x)
            ),
            cycle_length=(
                4 if mode != "eval" else 1
            ),  # make the test input to be the same order as files
            block_length=16,
        )
        dataset = dataset.prefetch(1)
        if mode == "train":
            dataset = dataset.repeat()

        # To batch similarly-sized sequences together, we need buffering support, which TF doesn't really have.
        # Instead, turn the dataset into a generator, run it through our own buffering function, and return a
        # dataset of batches from that function. Could probably be more efficient, PRs welcome :)
        ds = tf.data.Dataset.from_generator(
            lambda mode: self.to_batch(dataset, mode),
            output_types=(
                tf.dtypes.int32,
                tf.dtypes.int32,
                tf.dtypes.int32,
                tf.dtypes.int32,
            ),
            args=(mode,),
        )
        ds = ds.prefetch(1)
        return ds

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
            relations = [
                [2 * EDGE_TYPES[rel[3]], rel[0], rel[1]]
                for rel in edges
                if rel[3] in EDGE_TYPES
            ]  # Note: we reindex edge types to be 0-based and filter unsupported edge types (useful for ablations)
            relations += [
                [rel[0] + 1, rel[2], rel[1]] for rel in relations
            ]  # Add reverse edges
            return relations

        tokens = [
            self.vocabulary.translate(t)[: self.config["max_token_length"]]
            for t in json_data["source_tokens"]
        ]
        edges = parse_edges(json_data["edges"])

        # # Add hyper edge to the first node
        # for i in range(1, len(tokens)):
            # edges.append([EDGE_TYPES["enum_HYPER"] * 2, 0, i])
            # edges.append([EDGE_TYPES["enum_HYPER"] * 2 + 1, i, 0])

        label = 0 if json_data["label"] == "Correct" else 1
        sample_id = json_data["idx"]

        return (
            tokens,
            edges,
            label,
            sample_id,
        )

    # Creates Tensor batches from a set of files
    def to_batch(self, sample_generator, mode):
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")

        def sample_len(sample):
            return len(sample[0])

        # Generates a batch with similarly-sized sequences for efficiency
        def make_batch(buffer):
            pivot = sample_len(random.choice(buffer))
            if mode != "eval":
                buffer = sorted(buffer, key=lambda b: abs(sample_len(b) - pivot))
            batch = []
            max_seq_len = 0
            for sample in buffer:
                max_seq_len = max(max_seq_len, sample_len(sample))
                if max_seq_len * (len(batch) + 1) > self.config["max_batch_size"]:
                    break
                batch.append(sample)
            batch_dim = len(batch)
            buffer = buffer[batch_dim:]
            batch = list(zip(*batch))

            # Pad all tokens to max length using ragged tensors
            token_tensor = tf.ragged.constant(
                batch[0], dtype=tf.dtypes.int32
            ).to_tensor(
                shape=(
                    len(batch[0]),
                    max(len(b) for b in batch[0]),
                    self.config["max_token_length"],
                )
            )

            # Add batch axis to all edges and flatten
            edge_batches = tf.repeat(
                tf.range(batch_dim), [len(edges) for edges in batch[1]]
            )
            edge_tensor = tf.concat(batch[1], axis=0)
            edge_tensor = tf.stack(
                [edge_tensor[:, 0], edge_batches, edge_tensor[:, 1], edge_tensor[:, 2]],
                axis=1,
            )

            # Label is just a simple constant list
            label = tf.constant(batch[2], dtype=tf.dtypes.int32)

            sample_ids = tf.constant(batch[3], dtype=tf.dtypes.int32)

            return buffer, (token_tensor, edge_tensor, label, sample_ids)

        # Keep samples in a buffer that is (ideally) much larger than the batch size to allow efficient batching
        buffer = []
        num_samples = 0
        counter = 0
        for line in sample_generator:
            json_sample = json.loads(line.numpy())

            if "idx" not in json_sample:
                json_sample["idx"] = counter
                counter += 1

            sample = self.to_sample(json_sample)
            if sample_len(sample) > self.config["max_sequence_length"]:
                continue

            if (
                mode == "train"
                and self.typecheck_dataset is not None
                and self.typecheck_dataset[json_sample["idx"]]["type_check_failed"]
            ):
                continue

            if (
                mode == "train"
                and self.random_filter_dataset is not None
                and not self.random_filter_dataset[json_sample["idx"]]["included"]
            ):
                continue

            buffer.append(sample)
            num_samples += 1
            if mode == "dev" and num_samples >= self.config["max_valid_samples"]:
                break
            if (
                sum(sample_len(sample) for l in buffer)
                > self.config["max_buffer_size"] * self.config["max_batch_size"]
            ):
                buffer, batch = make_batch(buffer)
                yield batch
        # Drain the buffer upon completion
        while buffer:
            buffer, batch = make_batch(buffer)
            yield batch
