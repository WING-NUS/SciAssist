import csv
from typing import Optional

import datasets
from datasets import Dataset, DatasetDict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from src.models.components.bart_summarization import BartForSummarization
from src.models.components.bart_tokenizer import bart_tokenizer
from src.utils.pad_for_seq2seq import tokenize_and_align_labels


class MupDataModule(LightningDataModule):
    def __init__(
        self,
        data_repo: str,
        train_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_cache_dir: str = ".cache",
        seed: int = 777
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_collator = DataCollatorForSeq2Seq(bart_tokenizer, model=BartForSummarization, pad_to_multiple_of=8)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> DatasetDict:
        raw_datasets = datasets.load_dataset(
            self.hparams.data_repo,
            cache_dir=self.hparams.data_cache_dir
        )

        # Get test from csv
        test_set = {"paper_name":[], "text":[], "summary":[],"paper_id":[]}
        with open("data/mup/test-release.csv",'r', newline='') as f:
            rows = csv.reader(f)
            # Skip title line
            next(rows)
            for row in rows:
                test_set["paper_name"].append(row[1])
                test_set["text"].append(row[2])
                test_set["summary"].append(row[3])
                test_set["paper_id"].append(row[4])
        test_set = Dataset.from_dict(test_set)
        raw_datasets["test"] = test_set
        return raw_datasets

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            processed_datasets = self.prepare_data()
            tokenized_datasets = processed_datasets.map(
                lambda x: tokenize_and_align_labels(x, inputs_column="text", labels_column="summary"),
                batched=True,
                remove_columns=processed_datasets["train"].column_names,
                load_from_cache_file=True
            )
            self.data_train = tokenized_datasets["train"]
            self.data_val = tokenized_datasets["validation"]
            self.data_test = tokenized_datasets["test"]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_collator,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_collator,
            shuffle=False,
        )

