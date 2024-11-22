import argparse
import glob

import orjson
import pandas as pd
import yaml
from tqdm import tqdm

# import json


# from data import data_loader, vocabulary


def create_candidate_group(tokens, repair_targets, repair_candidates):
    """
    Find the variable for each repair candiate. Group 0 is for the groud truth variable
    """
    group_counter = 0
    group_map = {}
    results = []

    if len(repair_targets) > 0:
        repair_tokens = tuple(tokens[repair_targets[0]])
        group_map[repair_tokens] = group_counter
    group_counter += 1

    for candidate in repair_candidates:
        candidate_tokens = tuple(tokens[candidate])
        if candidate_tokens not in group_map:
            group_map[candidate_tokens] = group_counter
            group_counter += 1
        results.append(group_map[candidate_tokens])
    return results


def main(args, config):
    files = sorted(glob.glob(args.folder_path + "/*.txt*"))
    ground_truth = []

    pbar = tqdm()

    for file in files:
        with open(file) as f:
            programs = []
            for line in f:
                programs.append(orjson.loads(line))

        for program in programs:
            tokens = program["source_tokens"]
            repair_targets = program["repair_targets"]
            error_loc = program["error_location"]

            repair_candidates = [
                t for t in program["repair_candidates"] if isinstance(t, int)
            ]

            candidate_group = create_candidate_group(
                tokens, repair_targets, repair_candidates
            )
            ground_truth.append(
                {
                    "error_location": error_loc,
                    "repair_targets": repair_targets,
                    "repair_candidates": repair_candidates,
                    "candidate_group": candidate_group,
                }
            )
            pbar.update(1)

    df = pd.DataFrame(ground_truth)
    df.to_csv(args.output_path)


# def main_old(args, config):
#     loader = data_loader.DataLoader(
#         args.folder_path, config["data"], vocabulary.Vocabulary(args.vocab_path)
#     )
#     ground_truth = []

#     pbar = tqdm()
#     for batch in loader.batcher(mode="eval"):
#         (
#             tokens,
#             edges,
#             error_loc,
#             repair_targets,
#             repair_candidates,
#             changes,
#             sample_id,
#         ) = batch
#         batch_size = tf.shape(tokens)[0].numpy()

#         targets = repair_targets
#         candidates = repair_candidates
#         for i in range(batch_size):
#             targets_i = targets[targets[:, 0] == i][:, 1]
#             candidates_i = candidates[candidates[:, 0] == i][:, 1]

#             candidate_group = create_candidate_group(
#                 tokens[i].numpy().tolist(),
#                 targets_i.numpy().tolist(),
#                 candidates_i.numpy().tolist(),
#             )
#             ground_truth.append(
#                 {
#                     "error_location": error_loc[i].numpy().tolist(),
#                     "repair_targets": targets_i.numpy().tolist(),
#                     "repair_candidates": candidates_i.numpy().tolist(),
#                     "candidate_group": candidate_group,
#                     "sample_id": sample_id[i].numpy().tolist(),
#                 }
#             )
#         pbar.update(batch_size)

#     with open(args.output_path, "w") as f:
#         json.dump(ground_truth, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.yml")
    parser.add_argument("--folder_path", type=str, default="./great/eval")
    parser.add_argument("--vocab_path", type=str, default="./vocab.txt")
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    main(args, config)
