# main developer: Yixi Ding <dingyixi@hotmail.com>
import csv
import math
import os
from pathlib import Path
from typing import Optional

import datasets
from datasets import Dataset, DatasetDict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from SciAssist.utils.data_utils import DataUtilsForSeq2Seq


class MupSciSummDataModule(LightningDataModule):
    def __init__(
            self,
            data_repo: str,
            train_batch_size: int = 8,
            num_workers: int = 0,
            pin_memory: bool = False,
            data_cache_dir: str = ".cache",
            seed: int = 777,
            data_utils=DataUtilsForSeq2Seq
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_cache_dir = Path(self.hparams.data_cache_dir)
        self.data_utils = self.hparams.data_utils
        self.data_collator = self.data_utils.collator()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> DatasetDict:
        mup_datasets = datasets.load_dataset(
            self.hparams.data_repo,
            cache_dir=self.data_cache_dir
        )
        # Prepare keywords
        id2kw = {}
        filter_files = ['C04-1046', 'J93-2004', 'C00-1072', 'J94-4004', 'C04-1100', 'C88-1016', 'M95-1005', 'J05-1004',
                        'H94-1046', 'C86-1016', 'P90-1010', 'C96-1079', 'P89-1009', 'C92-3150', 'C00-2136', 'C92-1025',
                        'H94-1048', 'M95-1012', 'H92-1026', 'P85-1011', 'C96-2183', 'C94-1042', 'C92-2082', 'C02-2025',
                        'H05-2018', 'W08-2123', 'C96-1058', 'H91-1060', 'C88-2128', 'C02-1139', 'C00-1007', 'C92-3126',
                        'I05-3025', 'J06-1003', 'J08-2005', 'H91-1026', 'C94-1032', 'C94-1027', 'H93-1051', 'C88-2147',
                        'H92-1045', 'C90-3063', 'C90-3044', 'C94-2174', 'D07-1074', 'H93-1052', 'N03-2021', 'C88-2121',
                        'C92-2070', 'C00-2137', 'W97-0703', 'H94-1020', 'C90-3030', 'H01-1035', 'C96-1055', 'H93-1061',
                        'C96-1005', 'C00-1044', 'C92-1038', 'C00-2163', 'C90-3052', 'J94-2003', 'C04-1073']
        texts = []
        summaries = []
        keywords = []
        with open("/home/yixi/project/scisumm-corpus/data/Training-Set-2019/Task2/From-ScisummNet-2019/scisumm.csv",
                  'r', newline='', encoding='ISO-8859-1') as f:
            rows = csv.reader(f)
            # Get Column names
            keys = next(rows)
            # Add values by column
            for row in rows:
                k_list = row[1].split(",")
                id2kw[row[0]] = [k.strip() for k in k_list]

        file_list = []
        root_dir = "/home/yixi/project/scisumm-corpus/data/Training-Set-2019/Task2/From-ScisummNet-2019"
        for dirpath, dirnames, files in os.walk(root_dir):
            file_list = dirnames
            break

        for file in file_list:
            if file in filter_files:
                continue
            with open(os.path.join(root_dir, file, "summary", file + ".scisummnet_human.txt"), "r") as f:
                summary = f.readlines()
                summary = " ".join(summary[1:])
                summaries.append(summary)
            with open(os.path.join(root_dir, file, file + ".txt"), "r") as f:
                text = f.readlines()
                text = " ".join(text)
                texts.append(text)
            keywords.append(id2kw[file])
        scisumm_datasets = {
            "text": texts,
            "summary": summaries,
            "keywords": keywords,
            "length": [None for i in texts]
        }

        lengths = [len(s.split(" ")) for s in mup_datasets["train"]["summary"]]
        lengths = [50 * math.ceil(s / 50) for s in lengths]

        raw_datasets = DatasetDict()
        raw_datasets["train"] = {
            "text": scisumm_datasets["text"] + mup_datasets["train"]["text"],
            "summary": scisumm_datasets["summary"] + mup_datasets["train"]["summary"],
            "keywords": keywords + [None for i in mup_datasets["train"]["text"]],
            "length": scisumm_datasets["length"] + lengths
        }

        raw_datasets["train"] = Dataset.from_dict(raw_datasets["train"])
        # raw_datasets["test"] = Dataset.from_dict(mup_datasets)
        # raw_datasets["validation"] = Dataset.from_dict(raw_datasets["validation"])
        # raw_datasets = Dataset.from_dict(raw_datasets)
        return raw_datasets

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            processed_datasets = self.prepare_data()
            tokenized_datasets = processed_datasets.map(
                lambda x: self.data_utils.tokenize_and_align_labels(x, inputs_column="text", labels_column="summary"),
                batched=True,
                remove_columns=processed_datasets["train"].column_names,
                load_from_cache_file=True
            )
            # self.data_train = tokenized_datasets["train"]
            # self.data_val = tokenized_datasets["validation"]
            length = len(tokenized_datasets["train"])
            train_size = int(0.9 * length)
            validate_size = length - train_size
            self.data_train, self.data_val = random_split(tokenized_datasets["train"], [train_size, validate_size])
            # self.data_test = tokenized_datasets["test"].select(range(100))
            # If labels are not provided, delete the column "labels"
            # self.data_test = tokenized_datasets["test"]

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
    #
    # def test_dataloader(self):
    #     return DataLoader(
    #         dataset=self.data_test,
    #         batch_size=self.hparams.train_batch_size,
    #         num_workers=self.hparams.num_workers,
    #         pin_memory=self.hparams.pin_memory,
    #         collate_fn=self.data_collator,
    #         shuffle=False,
    #     )
    #
