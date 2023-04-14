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
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from seqeval.scheme import IOB2
from sklearn.metrics import precision_score as sc_precision_score, recall_score as sc_recall_score, f1_score as sc_f1_score

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

        self.val_f1 = 0
        self.test_f1 = 0
        self.val_pre = 0
        self.test_pre = 0
        self.val_recall = 0
        self.test_recall = 0

        self.val_cls_f1 = 0
        self.test_cls_f1 = 0
        self.val_cls_pre = 0
        self.test_cls_pre = 0
        self.val_cls_recall = 0
        self.test_cls_recall = 0

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
        self.val_cls_f1 = sc_f1_score(pred_labels, true_labels, average='binary')

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/cls_f1", self.val_cls_f1, on_step=False, on_epoch=True, prog_bar=False)
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
        self.val_pre = precision_score(true_tags, pred_tags, mode='strict', scheme=IOB2)
        self.val_recall = recall_score(true_tags, pred_tags, mode='strict', scheme=IOB2)
        self.val_cls_f1 = sc_f1_score(true_labels, pred_labels,average='binary')
        self.val_cls_pre = sc_precision_score(true_labels, pred_labels, average='binary')
        self.val_cls_recall = sc_recall_score(true_labels, pred_labels, average='binary')

        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val/pre", self.val_pre, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall, on_epoch=True, prog_bar=True)
        self.log("val/cls_f1", self.val_cls_f1, on_epoch=True, prog_bar=True)
        self.log("val/cls_pre", self.val_cls_pre, on_epoch=True, prog_bar=True)
        self.log("val/cls_recall", self.val_cls_recall, on_epoch=True, prog_bar=True)

        if self.val_f1 > self.val_f1_best:
            print(f"The f1 on validation set: {self.val_f1:.3f}, higher than the previous best one {self.val_f1_best:.3f}")
            print(f"Epoch: {self.current_epoch}: Found better model! Save the current model as the best model!")
            self.val_f1_best = self.val_f1
            torch.save(self.model.state_dict(), self.save_path)
        else:
            print(f"Epoch: {self.current_epoch}: No improvement! The current best model is still the best.")
        self.log("val/f1_best", self.val_f1_best, on_epoch=True, prog_bar=True)
        print("Epoch:", self.current_epoch, ", val/f1:", self.val_f1, ", val/precision:", self.val_pre, ", val/recall:", self.val_recall, ", val/cls_f1:", self.val_cls_f1, ", val/cls_pre:", self.val_cls_pre, ", val/cls_recall", self.val_cls_recall, ", val/best_f1:", self.val_f1_best)
        print(classification_report(true_tags, pred_tags, mode='strict', scheme=IOB2))


    def test_step(self, batch: Any, batch_idx: int):
        loss, ner_output, cls_output, batch_tags, batch_labels = self.step(batch)
        pred_tags, true_tags, pred_labels, true_labels = self.data_utils.postprocess(ner_output, cls_output, batch_tags, batch_labels)
        self.test_f1 = f1_score(true_tags, pred_tags, mode='strict', scheme=IOB2)
        self.test_cls_f1 = sc_f1_score(pred_labels, true_labels, average='binary')

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/cls_f1", self.test_cls_f1, on_step=False, on_epoch=True, prog_bar=False)

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
        self.test_pre = precision_score(true_tags, pred_tags, mode='strict', scheme=IOB2)
        self.test_recall = recall_score(true_tags, pred_tags, mode='strict', scheme=IOB2)
        self.test_cls_f1 = sc_f1_score(true_labels, pred_labels,average='binary')
        self.test_cls_pre = sc_precision_score(true_labels, pred_labels, average='binary')
        self.test_cls_recall = sc_recall_score(true_labels, pred_labels, average='binary')

        self.log("test/f1", self.test_f1, on_epoch=True, prog_bar=True)
        self.log("test/pre", self.test_pre, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_recall, on_epoch=True, prog_bar=True)
        self.log("test/cls_f1", self.test_cls_f1, on_epoch=True, prog_bar=True)
        self.log("test/cls_pre", self.test_cls_pre, on_epoch=True, prog_bar=True)
        self.log("test/cls_recall", self.test_cls_recall, on_epoch=True, prog_bar=True)
        print("test/f1:", self.test_f1, ", test/precision:", self.test_pre, ", test/recall:", self.test_recall, ", test/cls_f1:", self.test_cls_f1, ", test/cls_pre:", self.test_cls_pre, ", val/test_recall", self.test_cls_recall)
        print(classification_report(true_tags, pred_tags, mode='strict', scheme=IOB2))


    def on_epoch_end(self):
        self.val_f1 = 0
        self.test_f1 = 0
        self.val_pre = 0
        self.test_pre = 0
        self.val_recall = 0
        self.test_recall = 0

        self.val_cls_f1 = 0
        self.test_cls_f1 = 0
        self.val_cls_pre = 0
        self.test_cls_pre = 0
        self.val_cls_recall = 0
        self.test_cls_recall = 0


    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.hparams.model.parameters(), lr=self.hparams.lr
        )

