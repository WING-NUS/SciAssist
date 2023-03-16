import os
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import seaborn as sn
import torch
from pytorch_lightning import LightningModule
import sklearn

from SciAssist.models.components.bert_dataset_extraction import BertForDatasetExtraction
from SciAssist.utils.data_utils import DataUtilsForDatasetExtraction
from seqeval.metrics import f1_score, accuracy_score, classification_report
from seqeval.scheme import IOB2

class DatasetExtractionModule(LightningModule):
    def __init__(
        self,
        model: BertForDatasetExtraction,
        data_utils: DataUtilsForDatasetExtraction,
        lr: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_utils = data_utils
        self.model = model

        self.val_cls_acc = 0
        self.test_cls_acc = 0
        self.val_f1 = 0
        self.test_f1 = 0
        self.val_cls_f1 = 0
        self.test_cls_f1 = 0
        self.val_ner_acc = 0
        self.test_ner_acc = 0

        self.val_f1_best = 0
        self.save_path = Path(os.path.join(self.model.model_dir,self.model.save_name))


    def forward(self, inputs):
        return self.hparams.model(**inputs)


    def on_train_start(self):
        self.val_acc_best = 0
        self.val_f1_best = 0


    def step(self, batch: Any):
        inputs, batch_tags, batch_labels = batch, batch["ner_tags"], batch["cls_labels"]
        outputs = self.forward(inputs)

        loss = outputs[0]
        ner_output = outputs[1]
        cls_output = outputs[2]

        return loss, ner_output, cls_output, batch_tags, batch_labels


    def training_step(self, batch: Any, batch_idx: int):
        loss, ner_output, cls_output, batch_tags, batch_labels = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss}


    def training_epoch_end(self, outputs: List[Any]):
        pass


    def validation_step(self, batch: Any, batch_idx: int):
        loss, ner_output, cls_output, batch_tags, batch_labels = self.step(batch)
        pred_tags, true_tags, pred_labels, true_labels = self.data_utils.postprocess(ner_output, cls_output, batch_tags, batch_labels)
        self.val_f1 = f1_score(true_tags, pred_tags, mode='strict', scheme=IOB2)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "pred_tags": pred_tags, "true_tags": true_tags, "pred_labels": pred_labels, "true_labels": true_labels}


    def validation_epoch_end(self, outputs):
        true_tags = []
        pred_tags = []
        pred_labels = []
        true_labels = []
        for output in outputs:
            pred_tags.extend(output["pred_tags"])
            true_tags.extend(output["true_tags"])
            pred_labels.extend(output["pred_labels"])
            true_labels.extend(output["true_labels"])

        self.val_f1 = f1_score(true_tags, pred_tags, mode='strict', scheme=IOB2)
        self.val_cls_f1 = sklearn.metrics.f1_score(true_labels, pred_labels)
        self.val_cls_acc = accuracy_score(true_labels, pred_labels)
        self.val_ner_acc = accuracy_score(true_tags, pred_tags)

        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val/cls_f1", self.val_cls_f1, on_epoch=True, prog_bar=True)
        self.log("val/cls_acc", self.val_cls_acc, on_epoch=True, prog_bar=True)

        if self.val_f1 > self.val_f1_best:
            print(f"The f1 on validation set: {self.val_f1:.3f}, higher than the previous best one {self.val_f1_best:.3f}")
            print(f"Epoch: {self.current_epoch}: Found better model! Save the current model as the best model!")
            self.val_f1_best = self.val_f1
            torch.save(self.model.state_dict(), self.save_path)
        else:
            print(f"Epoch: {self.current_epoch}: No improvement! The current best model is still the best.")
        self.log("val/f1_best", self.val_f1_best, on_epoch=True, prog_bar=True)
        print("Epoch:", self.current_epoch, ", val/f1:", self.val_f1, ", val/ner_acc:", self.val_ner_acc, ", val/cls_f1:", self.val_cls_f1, ", val/cls_acc:", self.val_cls_acc, ", val/best_f1:", self.val_f1_best)
        print(classification_report(true_tags, pred_tags, mode='strict', scheme=IOB2))


    def test_step(self, batch: Any, batch_idx: int):
        loss, ner_output, cls_output, batch_tags, batch_labels = self.step(batch)
        pred_tags, true_tags, pred_labels, true_labels = self.data_utils.postprocess(ner_output, cls_output, batch_tags, batch_labels)
        self.test_f1 = f1_score(true_tags, pred_tags, mode='strict', scheme=IOB2)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "pred_tags": pred_tags, "true_tags": true_tags, "pred_labels": pred_labels, "true_labels": true_labels}


    def test_epoch_end(self, outputs):
        true_tags = []
        pred_tags = []
        pred_labels = []
        true_labels = []
        for output in outputs:
            pred_tags.extend(output["pred_tags"])
            true_tags.extend(output["true_tags"])
            pred_labels.extend(output["pred_labels"])
            true_labels.extend(output["true_labels"])

        self.test_f1 = f1_score(true_tags, pred_tags, mode='strict', scheme=IOB2)
        self.test_ner_acc = accuracy_score(true_tags, pred_tags)
        self.test_cls_f1 = sklearn.metrics.f1_score(true_labels, pred_labels)
        self.test_cls_acc = accuracy_score(true_labels, pred_labels)

        self.log("test/f1", self.test_f1, on_epoch=True, prog_bar=True)
        self.log("test/ner_acc", self.test_ner_acc, on_epoch=True, prog_bar=True)
        self.log("test/cls_f1", self.test_cls_f1, on_epoch=True, prog_bar=True)
        self.log("test/cls_acc", self.test_cls_acc, on_epoch=True, prog_bar=True)
        print(classification_report(true_tags, pred_tags, mode='strict', scheme=IOB2))


    def on_epoch_end(self):
        self.val_cls_acc = 0
        self.test_cls_acc = 0
        self.val_f1 = 0
        self.test_f1 = 0
        self.val_cls_f1 = 0
        self.test_cls_f1 = 0
        self.val_ner_acc = 0
        self.test_ner_acc = 0


    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.hparams.model.parameters(), lr=self.hparams.lr
        )

