import json
import unittest

import datasets
import pandas as pd
from preprocess import (DataCollatorForVarMisuse, align_labels_with_tokens,
                        generate_data_token_classification, generate_nx_graph,
                        tokenize_and_align_labels)
from transformers import AutoTokenizer

EXAMPLE = """{"has_bug": true, "bug_kind": 1, "bug_kind_name": "VARIABLE_MISUSE", "source_tokens": ["#NEWLINE#", 
"def _read_2byte(", "img_data", ",", "pos", ")", ":", "#NEWLINE#", "#INDENT#", "return", "(", "(", "ord", "(", 
"img_data", "[", "pos", "]", ")", "<<", "8", ")", "+", "ord", "(", "img_data", "[", "(", "img_data", "+", "1", ")", 
"]", ")", ")"], "edges": [[1, 3, 7, "enum_FIELD"], [9, 22, 7, "enum_FIELD"], [22, 19, 7, "enum_FIELD"], [22, 24, 7, 
"enum_FIELD"], [19, 13, 7, "enum_FIELD"], [19, 20, 7, "enum_FIELD"], [24, 23, 7, "enum_FIELD"], [13, 12, 7, 
"enum_FIELD"], [26, 25, 7, "enum_FIELD"], [15, 14, 7, "enum_FIELD"], [29, 28, 7, "enum_FIELD"], [29, 30, 7, 
"enum_FIELD"], [1, 2, 9, "enum_NEXT_SYNTAX"], [2, 3, 9, "enum_NEXT_SYNTAX"], [3, 4, 9, "enum_NEXT_SYNTAX"], [4, 5, 9, 
"enum_NEXT_SYNTAX"], [5, 6, 9, "enum_NEXT_SYNTAX"], [6, 9, 9, "enum_NEXT_SYNTAX"], [9, 10, 9, "enum_NEXT_SYNTAX"], 
[10, 11, 9, "enum_NEXT_SYNTAX"], [11, 12, 9, "enum_NEXT_SYNTAX"], [12, 13, 9, "enum_NEXT_SYNTAX"], [14, 2, 10, 
"enum_LAST_LEXICAL_USE"], [13, 14, 9, "enum_NEXT_SYNTAX"], [14, 15, 9, "enum_NEXT_SYNTAX"], [16, 4, 10, 
"enum_LAST_LEXICAL_USE"], [15, 16, 9, "enum_NEXT_SYNTAX"], [16, 17, 9, "enum_NEXT_SYNTAX"], [17, 18, 9, 
"enum_NEXT_SYNTAX"], [18, 19, 9, "enum_NEXT_SYNTAX"], [19, 20, 9, "enum_NEXT_SYNTAX"], [20, 21, 9, 
"enum_NEXT_SYNTAX"], [21, 22, 9, "enum_NEXT_SYNTAX"], [23, 12, 10, "enum_LAST_LEXICAL_USE"], [22, 23, 9, 
"enum_NEXT_SYNTAX"], [23, 24, 9, "enum_NEXT_SYNTAX"], [25, 14, 10, "enum_LAST_LEXICAL_USE"], [24, 25, 9, 
"enum_NEXT_SYNTAX"], [25, 26, 9, "enum_NEXT_SYNTAX"], [26, 27, 9, "enum_NEXT_SYNTAX"], [28, 25, 10, 
"enum_LAST_LEXICAL_USE"], [27, 28, 9, "enum_NEXT_SYNTAX"], [28, 29, 9, "enum_NEXT_SYNTAX"], [29, 30, 9, 
"enum_NEXT_SYNTAX"], [30, 31, 9, "enum_NEXT_SYNTAX"], [31, 32, 9, "enum_NEXT_SYNTAX"], [32, 33, 9, 
"enum_NEXT_SYNTAX"], [33, 34, 9, "enum_NEXT_SYNTAX"], [3, 9, 1, "enum_CFG_NEXT"], [14, 2, 3, "enum_LAST_WRITE"], [16, 
4, 3, "enum_LAST_WRITE"], [23, 12, 2, "enum_LAST_READ"], [25, 14, 2, "enum_LAST_READ"], [25, 2, 3, 
"enum_LAST_WRITE"], [28, 25, 2, "enum_LAST_READ"], [28, 2, 3, "enum_LAST_WRITE"]], "error_location": 28, 
"repair_targets": [4, 16], "repair_candidates": [2, 14, 25, 28, 4, 16], "provenances": [{"datasetProvenance": {
"datasetName": "ETHPy150Open", "filepath": "CollabQ/CollabQ/common/imageutil.py", "license": "apache-2.0", 
"note": "license: bigquery_api"}}]}"""
DATA = json.loads(EXAMPLE)
TOKENIZER = AutoTokenizer.from_pretrained("microsoft/codebert-base", add_prefix_space=True)

class Tests(unittest.TestCase):
    def test_preproces_graph(self):
        g = generate_nx_graph(DATA)
        # print(g.edges(data=True))

    def test_preprocess_token_class(self):
        data = generate_data_token_classification(DATA)
        # print(data)

        words = data["tokens"]
        line1 = ""
        line2 = ""
        line3 = ""
        for word, label1, label2 in zip(words, data["error_location"], data["repair_locations"]):
            full_label1 = str(label1)
            full_label2 = str(label2)
            max_length = max(len(word), len(full_label1))
            line1 += word + " " * (max_length - len(word) + 1)
            line2 += full_label1 + " " * (max_length - len(full_label1) + 1)
            line3 += full_label2 + " " * (max_length - len(full_label2) + 1)

        print()
        print(line1)
        print(line2)
        print(line3)

        inputs = TOKENIZER(data["tokens"], is_split_into_words=True)
        print(inputs.tokens())
        print(inputs.word_ids())

        labels = data["error_location"]
        word_ids = inputs.word_ids()
        print(labels)
        alignment = align_labels_with_tokens(labels, word_ids)
        alignment[0] = int(data["has_bug"])
        alignment_2 = align_labels_with_tokens(data["repair_locations"], word_ids)
        for t, a, b in zip(inputs.tokens(), alignment, alignment_2):
            print(t, a, b)

    def test_datacollator(self):
        data = generate_data_token_classification(DATA)
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=[data, data, data]))
        print(dataset[0])
        tokenized_datasets = dataset.map(
            lambda x: tokenize_and_align_labels(x, TOKENIZER),
            batched=True,
            remove_columns=dataset.column_names
        )
        print(tokenized_datasets[0])

        data_collator = DataCollatorForVarMisuse(tokenizer=TOKENIZER)
        batch = data_collator(tokenized_datasets)
        print(batch["labels"].shape)
        print(batch["labels"][0][0])
        print('------------------')
        print(batch["labels"][0][1])


if __name__ == '__main__':
    unittest.main()
