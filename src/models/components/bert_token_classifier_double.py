import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from src.datamodules.components.cora_label import num_labels
from src.models.components.bert_token_classifier import BertTokenClassifier



class BertTokenClassifierDouble(nn.Module):
    def __init__(
        self,
        model_checkpoint: str,
        output_size: int = num_labels,
        cache_dir: str = ".cache"
    ):
        super().__init__()
        self.bert_classifier: BertTokenClassifier = BertTokenClassifier()
        self.bert_classifier.load_state_dict(torch.load(model_checkpoint))
        self.bert_classifier.eval()
        self.output_size = output_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(in_features=19, out_features=output_size, bias=True)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert_classifier(input_ids, attention_mask=attention_mask)
        outputs = self.dropout(outputs[0])
        logits = self.classifier(outputs)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.output_size), labels.view(-1))
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

