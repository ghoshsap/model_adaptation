"""Dataset helpers"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "name": None,
        "text_column": "text",
        "train_split": "train",
        "eval_split": "validation",
    },
    "wikitext2": {
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "text_column": "text",
        "train_split": "train",
        "eval_split": "validation",
    },
}


@dataclass
class DatasetArguments:
    dataset: str = "tinystories"
    block_size: int = 512
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


def _group_texts(examples: Dict[str, list], block_size: int) -> Dict[str, list]:
    # Concatenate texts then split into blocks of block_size tokens
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [
            concatenated_examples[k][i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        for k in concatenated_examples.keys()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def load_and_prepare_dataset(
    args: DatasetArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> DatasetDict:
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset '{args.dataset}'")

    config = DATASET_CONFIGS[args.dataset]
    text_column = config["text_column"]
    train_split = config["train_split"]
    eval_split = config["eval_split"]

    raw_dataset = load_dataset(config["path"], config["name"])

    train_dataset = raw_dataset[train_split]
    eval_dataset = raw_dataset[eval_split]

    if args.max_train_samples:
        train_dataset = train_dataset.select(
            range(min(args.max_train_samples, len(train_dataset)))
        )
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(
            range(min(args.max_eval_samples, len(eval_dataset)))
        )

    dataset = DatasetDict({"train": train_dataset, "validation": eval_dataset})

    tokenizer.pad_token = tokenizer.eos_token
    remove_columns = dataset["train"].column_names

    def tokenize_function(batch):
        return tokenizer(batch[text_column])

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_columns,
        desc="Tokenizing",
    )

    lm_datasets = tokenized.map(
        lambda batch: _group_texts(batch, args.block_size),
        batched=True,
        desc="Grouping texts",
    )

    return DatasetDict(
        {
            "train": lm_datasets["train"],
            "validation": lm_datasets["validation"],
        }
    )
