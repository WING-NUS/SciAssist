import datasets
from datasets import Dataset, DatasetDict

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.datamodules.components.cora_label import num_labels, label2id

from src.utils.pad_for_token_level import tokenize_and_align_labels, pad
from typing import Optional




class CoraDataModule(LightningDataModule):
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
        self.data_collator = pad
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return num_labels

    def prepare_data(self) -> DatasetDict:
        raw_datasets = datasets.load_dataset(
            self.hparams.data_repo,
            cache_dir=self.hparams.data_cache_dir
        )
        return raw_datasets

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            processed_datasets = self.prepare_data()
            tokenized_datasets = processed_datasets.map(
                lambda x: tokenize_and_align_labels(x, label2id),
                batched=True,
                remove_columns=processed_datasets["train"].column_names,
                load_from_cache_file=True
            )
            self.data_train = tokenized_datasets["train"]
            self.data_val = tokenized_datasets["val"]
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
