import argparse
import glob
import json

import orjson
import pandas as pd
from tqdm import tqdm


def load_files(path):
    file_names = sorted(glob.glob(f"{path}/*.csv"))
    data = []
    for file_name in file_names:
        data.append(pd.read_csv(file_name))

    return pd.concat(data, ignore_index=True)


def main(config):
    count = 0
    files = sorted(glob.glob(f"{config.dataset_path}/*txt*"))

    for file in tqdm(files):
        data = []
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                data.append(orjson.loads(line))

        for d in data:
            d["id"] = count
            count += 1

        with open(file, "w") as f:
            for d in data:
                json.dump(d, f)
                f.write("\n")
    print(count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)

    args = parser.parse_args()
    main(args)
