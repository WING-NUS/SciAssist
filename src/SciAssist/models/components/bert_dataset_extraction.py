from transformers import *
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
import torch

class BertForDatasetExtraction(nn.Module):
    def __init__(
        self,
        model_checkpoint: str = "allenai/scibert_scivocab_uncased",
        cache_dir: str = ".cache",
        save_name: str = "scibert-ner.pt",
        model_dir: str = "pretrained"
    ):
        super().__init__()
        self.save_name = save_name
        self.model_dir = model_dir
        self.num_ner_tags = 3
        self.num_cls_labels = 2
        self.bert = AutoModel.from_pretrained(model_checkpoint, cache_dir=cache_dir)

        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.crf = CRF(self.num_ner_tags, batch_first=True)
        self.cel = CrossEntropyLoss()
        self.ner_classifier = nn.Linear(self.bert.config.hidden_size, self.num_ner_tags)
        self.cls_classifier = nn.Linear(self.bert.config.hidden_size, self.num_cls_labels)

    def forward(self, input_subwords, input_token_start_indexs, attention_mask=None, ner_tags=None, cls_labels=None):
        outputs = self.bert(input_subwords,
                            attention_mask=attention_mask,
                            token_type_ids=None,
                            position_ids=None,
                            head_mask=None,
                            inputs_embeds=None)
        sequence_output = outputs[0]

        # obtain original token representations from subwords representations (by selecting the first subword)
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)] for layer, starts in zip(sequence_output, input_token_start_indexs)]
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        padded_sequence_output = self.dropout(padded_sequence_output)

        ner_logits = self.ner_classifier(padded_sequence_output)

        cls_logits = self.cls_classifier(sequence_output[:, 0, :])

        outputs = (ner_logits, cls_logits,)

        if ner_tags is not None:
            alpha = 0.3
            mask = ner_tags.gt(-1)

            loss_ner = self.crf(ner_logits, ner_tags, mask) * (-1)
            loss_cls = self.cel(cls_logits.view(-1, 2), cls_labels.view(-1))
            loss = alpha * loss_ner + (1- alpha) * loss_cls

            outputs = (loss,) + outputs

        return outputs

