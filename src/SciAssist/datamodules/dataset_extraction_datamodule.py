from pathlib import Path
from typing import Optional

import datasets
from datasets import Dataset, DatasetDict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from SciAssist.utils.data_reader import csv_reader
from SciAssist.utils.data_utils import DataUtilsForDatasetExtraction


class DatasetExtractionModule(LightningDataModule):
    def __init__(
        self,
        data_repo: str,
        train_batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_cache_dir: str = ".cache",
        seed: int = 176,
        data_utils = DataUtilsForDatasetExtraction
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_cache_dir = Path(self.hparams.data_cache_dir)
        self.data_utils = self.hparams.data_utils

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            data_files = {"data": "sentences.txt", "labels": "labs.txt", "tags": "tags.txt"}
            print(self.hparams.data_repo + '/train/')
            raw_train_dataset = datasets.load_dataset(
                path='text',
                data_dir=self.hparams.data_repo + '/train/',
                data_files=data_files,
                cache_dir=self.data_cache_dir
            )
            raw_val_dataset = datasets.load_dataset(
                path='text',
                data_dir=self.hparams.data_repo + '/val/',
                data_files=data_files,
                cache_dir=self.data_cache_dir
            )
            raw_test_dataset = datasets.load_dataset(
                path='text',
                data_dir=self.hparams.data_repo + '/test/',
                data_files=data_files,
                cache_dir=self.data_cache_dir
            )

            self.data_train = self.data_utils.tokenize_and_align_labels(raw_train_dataset)
            self.data_val = self.data_utils.tokenize_and_align_labels(raw_val_dataset)
            self.data_test = self.data_utils.tokenize_and_align_labels(raw_test_dataset)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_train.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_val.collate_fn,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_test.collate_fn,
            shuffle=False,
        )

