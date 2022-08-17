import datasets
from datasets import Dataset, DatasetDict

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.datamodules.components.synthetic_label import num_labels, label2id
from src.datamodules.components.process import preprocess, tokenize_and_align_labels, pad
from typing import Optional, Tuple


class SyntheticDataModule(LightningDataModule):
    def __init__(
        self,
        data_repo: str,
        train_val_test_split: Tuple[int] = (100_000, 5_000, 10_000),
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
        raw_train_val_set = datasets.load_dataset(
            self.hparams.data_repo,
            split="train",
            cache_dir=self.hparams.data_cache_dir
        )
        raw_test_set = datasets.load_dataset(
            self.hparams.data_repo,
            split="test",
            cache_dir=self.hparams.data_cache_dir
        )
        shuffled_raw_train_val_set = raw_train_val_set.shuffle(seed=self.hparams.seed)
        selected_indices = list(range(self.hparams.train_val_test_split[0] + self.hparams.train_val_test_split[1]))
        selected_train_data = shuffled_raw_train_val_set.select(selected_indices[:self.hparams.train_val_test_split[0]])
        selected_val_data = shuffled_raw_train_val_set.select(selected_indices[self.hparams.train_val_test_split[0]:])
        selected_test_data = raw_test_set

        dataset_dict = DatasetDict()
        dataset_dict['train'] = selected_train_data
        dataset_dict['val'] = selected_val_data
        dataset_dict['test'] = selected_test_data
        return dataset_dict

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            dataset_dict = self.prepare_data()
            processed_datasets = dataset_dict.map(
                preprocess,
                batched=True,
                remove_columns=dataset_dict["train"].column_names,
                load_from_cache_file=True
            )
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
