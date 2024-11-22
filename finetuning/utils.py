import json
import os
import tempfile

import datasets
import pandas as pd
from preprocess import generate_data_token_classification
from tqdm import tqdm


def load_dataset(path, num_files=None, typecheck_path=None, random_split_path=None, max_samples=None):
    filenames = sorted(os.listdir(path))
    if num_files is not None:
        filenames = filenames[0:num_files]
    dataset = []

    count = 0
    for filename in tqdm(filenames, desc="Loading files"):
        with open(os.path.join(path, filename), "r") as f:
            lines = f.readlines()
            for line in lines:
                dataset.append(line)
                count += 1
                if max_samples is not None and count >= max_samples:
                    break

        if max_samples is not None and count >= max_samples:
            break

    dataset = [
        generate_data_token_classification(json.loads(d))
        for d in tqdm(dataset, desc="Generating data")
    ]

    if typecheck_path is not None:
        typecheck_files = sorted(os.listdir(typecheck_path))
        if num_files is not None:
            typecheck_files = typecheck_files[0:num_files]

        typecheck_dataset = []
        for filename in typecheck_files:
            typecheck_dataset.append(
                pd.read_csv(os.path.join(typecheck_path, filename), index_col=False)
            )

        typecheck_dataset = pd.concat(typecheck_dataset)
        typecheck_dataset.fillna("", inplace=True)
        typecheck_dataset = typecheck_dataset.to_dict(orient="records")

        filtered_dataset = []
        for i, d in enumerate(tqdm(typecheck_dataset, desc="Concate typecheck data")):
            if d["type_check_failed"]:
                continue
            filtered_dataset.append(dataset[i])
        del typecheck_dataset
        dataset = filtered_dataset

    if random_split_path is not None:
        include_datapoints = pd.read_csv(random_split_path, index_col=False)
        included = include_datapoints["included"].to_list()
        filtered_dataset = []

        for i, d in enumerate(tqdm(dataset, desc="Filtering data")):
            if included[i]:
                filtered_dataset.append(d)
        dataset = filtered_dataset

    _, filename = tempfile.mkstemp()
    with open(filename, "w") as f:
        for d in tqdm(dataset, desc="Writing to file"):
            f.write(json.dumps(d) + "\n")
    dataset = datasets.load_dataset("json", data_files=filename)

    return dataset["train"]  # train is the default dataset when loading
