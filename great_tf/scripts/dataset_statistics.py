import argparse
import glob

import orjson
import pandas as pd
from tqdm import tqdm


def main(args):
    files = sorted(glob.glob(args.folder_path + "/*.txt*"))

    results = []
    pbar = tqdm()
    for file in files:
        with open(file) as f:
            programs = []
            for line in f:
                programs.append(orjson.loads(line))

        for program in programs:
            results.append(
                {
                    "error_location": program["error_location"],
                    "num_repair_targets": len(program["repair_targets"]),
                    "num_repair_candidates": len(program["repair_candidates"]),
                    "num_tokens": len(program["source_tokens"]),
                }
            )
            pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="./great/eval")
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()
    main(args)
