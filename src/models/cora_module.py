import os
from typing import Any, List

import torch
import seaborn as sn
import matplotlib.pyplot as plt

from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix
from torchmetrics import F1Score
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.bert_token_classifier import BertTokenClassifier

from src.datamodules.components.cora_label import num_labels, LABEL_NAMES
from src.utils.pad_for_token_level import postprocess


class CoraLitModule(LightningModule):
    def __init__(
        self,
        model: BertTokenClassifier,
        lr: float = 2e-5,
        save_name: str = "scibert-uncased.pt",
        model_dir: str = "models/"
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = model

        # num_classes + 1 to account for the extra class used for padding
        self.val_acc = Accuracy(num_classes=num_labels+1, ignore_index=num_labels)
        self.test_acc = Accuracy(num_classes=num_labels+1, ignore_index=num_labels)
        self.val_micro_f1 = F1Score(num_classes=num_labels+1, ignore_index=num_labels, average="micro")
        self.test_micro_f1 = F1Score(num_classes=num_labels+1, ignore_index=num_labels, average="micro")
        # self.val_macro_f1 = F1Score(num_classes=num_labels+1, ignore_index=num_labels, average="macro")
        self.test_macro_f1 = F1Score(num_classes=num_labels+1, ignore_index=num_labels, average="macro")

        self.conf_matrix = ConfusionMatrix(num_classes=num_labels+1)

        self.val_acc_best = MaxMetric()
        self.val_micro_f1_best = MaxMetric()
        # self.val_macro_f1_best = MaxMetric()
        self.best_f1 = 0
        self.save_path = os.path.join(model_dir,save_name)




    def forward(self, inputs):
        return self.hparams.model(**inputs)

    def on_train_start(self):
        self.val_acc_best.reset()
        # self.val_macro_f1_best.reset()
        self.val_micro_f1_best.reset()

    def step(self, batch: Any):
        inputs, labels = batch, batch["labels"]
        outputs = self.forward(inputs)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        return loss, preds, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, labels = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, labels = self.step(batch)

        true_preds, true_labels = postprocess(preds, labels, LABEL_NAMES)
        true_labels = torch.flatten(true_labels)
        true_preds = torch.flatten(true_preds)

        acc = self.val_acc(true_preds, true_labels)
        micro_f1 = self.val_micro_f1(true_preds, true_labels)
        # macro_f1 = self.val_macro_f1(true_preds, true_labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/micro_f1", micro_f1, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/macro_f1", macro_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": true_preds, "labels": true_labels}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

        micro_f1 = self.val_micro_f1.compute()
        self.val_micro_f1_best.update(micro_f1)
        self.log("val/micro_f1_best", self.val_micro_f1_best.compute(), on_epoch=True, prog_bar=True)
        if micro_f1 > self.best_f1:
            # print(f"The micro_f1 on validation set: {micro_f1:.3f}, higher than the previous best one {self.best_f1:.3f}")
            print(f"Epoch: {self.current_epoch}: Save the current best model.")
            self.best_f1 = micro_f1
            torch.save(self.model.state_dict(), self.save_path)
        # macro_f1 = self.val_macro_f1.compute()
        # self.val_macro_f1_best.update(macro_f1)
        # self.log("val/macro_f1_best", self.val_micro_f1_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, labels = self.step(batch)

        true_preds, true_labels = postprocess(preds, labels, LABEL_NAMES)

        device = self.val_acc.device
        true_labels = torch.flatten(true_labels).to(device)
        true_preds = torch.flatten(true_preds).to(device)

        acc = self.test_acc(true_preds, true_labels)
        micro_f1 = self.test_micro_f1(true_preds, true_labels)
        macro_f1 = self.test_macro_f1(true_preds, true_labels)
        confmat = self.conf_matrix(true_preds, true_labels)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/micro_f1", micro_f1, on_step=False, on_epoch=True)
        self.log("test/macro_f1", macro_f1, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": true_preds, "labels": true_labels}

    def test_epoch_end(self, outputs: List[Any]):
        # wandb.init()
        acc = self.test_acc.compute()
        micro_f1 = self.test_micro_f1.compute()
        macro_f1 = self.test_macro_f1.compute()
        confmat = self.conf_matrix.compute()[:-1, :-1].tolist()

        self.log("test/acc", acc, on_epoch=True, prog_bar=True)
        self.log("test/micro_f1", micro_f1, on_epoch=True, prog_bar=True)
        self.log("test/macro_f1", macro_f1, on_epoch=True, prog_bar=True)

        plt.figure(figsize=(24, 26))
        sn.heatmap(confmat, annot=True, xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, fmt='d')
        # wandb.log({"Confusion Matrix": wandb.Image(plt)})

    def on_epoch_end(self):
        self.val_acc.reset()
        self.test_acc.reset()
        self.val_micro_f1.reset()
        self.test_micro_f1.reset()
        # self.val_macro_f1.reset()
        self.test_macro_f1.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.hparams.model.parameters(), lr=self.hparams.lr
        )
