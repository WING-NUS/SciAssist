import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.text.rouge import ROUGEScore
from transformers import AutoModelForSeq2SeqLM

from src.models.components.bart_tokenizer import bart_tokenizer
from src.utils.pad_for_seq2seq import postprocess


class MupBartLitModule(LightningModule):
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        lr: float = 2e-5,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = model
        self.val_metric = ROUGEScore(use_stemmer=True)
        self.val_best_Rouge1 = MaxMetric()
        self.test_metric = ROUGEScore(use_stemmer=True)
        self.test_best_Rouge1 = MaxMetric()
        self.best_Rouge1 = 0

        self.save_path = os.path.join(self.model.model_dir,self.model.save_name)

    def forward(self, inputs):
        return self.hparams.model(**inputs)

    def step(self, batch: Any):
        inputs, labels = batch, batch["labels"]
        outputs = self.forward(inputs)
        loss = outputs.loss
        return loss, labels

    def training_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch)
        loss = outputs[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.step(batch)
        loss = outputs[0]
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # Get prediction ids sequence
        preds = self.model.generate(input_ids=input_ids,attention_mask=attention_mask)
        # Convert ids to strings
        decoded_preds, decoded_labels = postprocess(preds, labels)

        # Compute Rouge Metrics
        rouge_metric = self.val_metric(preds=decoded_preds, target=decoded_labels)

        result = {key: value * 100 for key, value in rouge_metric.items()}

        # Compute average length of summaries
        prediction_lens = [np.count_nonzero(pred != bart_tokenizer.pad_token_id) for pred in preds.to("cpu")]
        result["gen_len"] = np.mean(prediction_lens)
        result["preds"] = decoded_preds
        result["labels"] = decoded_labels

        #  Log results
        self.log("val/Rouge-1", result["rouge1_fmeasure"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/PRouge-1", result["rouge1_precision"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/RRouge-1", result["rouge1_recall"], on_step=False, on_epoch=True, prog_bar=True)

        self.log("val/Rouge-2", result["rouge2_fmeasure"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/PRouge-2", result["rouge2_precision"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/RRouge-2", result["rouge2_recall"], on_step=False, on_epoch=True, prog_bar=True)

        self.log("val/Rouge-L", result["rougeL_fmeasure"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/PRouge-L", result["rougeL_precision"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/RRouge-L", result["rougeL_recall"], on_step=False, on_epoch=True, prog_bar=True)

        self.log("val/Rouge-Lsum", result["rougeLsum_fmeasure"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/PRouge-Lsum", result["rougeLsum_precision"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/RRouge-Lsum", result["rougeLsum_recall"], on_step=False, on_epoch=True, prog_bar=True)

        self.log("val/gen_len", result["gen_len"], on_step=True, on_epoch=False, prog_bar=True)

        return result

    def validation_epoch_end(self, outputs: List[Any]):
        rouge = self.val_metric.compute()

        if rouge["rouge1_fmeasure"] > self.best_Rouge1:
            self.val_best_Rouge1.update(rouge["rouge1_fmeasure"])
            self.best_Rouge1 = rouge["rouge1_fmeasure"]
            print(f"Epoch: {self.current_epoch}: Save the current best model.")
            torch.save(self.model.state_dict(), self.save_path)

        self.log("val/best_rouge1", self.val_best_Rouge1.compute(), on_epoch=True, prog_bar=True)


    def test_step(self, batch: Any, batch_idx: int):
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        # Get prediction ids sequence
        preds = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        # Convert ids to strings
        decoded_preds, decoded_labels = postprocess(preds, labels)

        # Compute Rouge Metrics
        rouge_metric = self.test_metric(preds=decoded_preds, target=decoded_labels)

        result = {key: value * 100 for key, value in rouge_metric.items()}

        # Compute average length of summaries
        prediction_lens = [np.count_nonzero(pred != bart_tokenizer.pad_token_id) for pred in preds.to("cpu")]
        result["gen_len"] = np.mean(prediction_lens)
        result["preds"] = decoded_preds
        result["labels"] = decoded_labels

        # Log results
        self.log("test/Rouge-1", result["rouge1_fmeasure"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/Rouge-2", result["rouge2_fmeasure"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/Rouge-L", result["rougeL_fmeasure"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/Rouge-Lsum", result["rougeLsum_fmeasure"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/gen_len", result["gen_len"], on_step=True, on_epoch=False, prog_bar=True)

        return result

    def test_epoch_end(self, outputs: List[Any]):
        # wandb.init()
        rouge = self.test_metric.compute()
        self.log("test/Rouge-1", rouge["rouge1_fmeasure"], on_epoch=True, prog_bar=True)
        self.log("test/Rouge-2", rouge["rouge2_fmeasure"], on_epoch=True, prog_bar=True)
        self.log("test/Rouge-L", rouge["rougeL_fmeasure"], on_epoch=True, prog_bar=True)
        self.log("test/Rouge-Lsum", rouge["rougeLsum_fmeasure"], on_epoch=True, prog_bar=True)


        plt.figure(figsize=(24, 26))

        # wandb.log({"Confusion Matrix": wandb.Image(plt)})

    def on_epoch_end(self):
        self.val_metric.reset()
        self.val_best_Rouge1.reset()
        self.test_metric.reset()
        self.test_best_Rouge1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.hparams.model.parameters(), lr=self.hparams.lr
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer],[lr_scheduler]
