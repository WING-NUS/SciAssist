from torch import nn
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import Seq2SeqLMOutput


class BartForSummarization(nn.Module):
    def __init__(
        self,
        model_checkpoint: str = "facebook/bart-large-cnn",
        cache_dir: str = ".cache",
        save_name: str = "bart-large-cnn-mup.pt",
        model_dir: str = "pretrained"
    ):
        super().__init__()
        self.save_name = save_name
        self.model_dir = model_dir
        self.bart = AutoModelForSeq2SeqLM.from_pretrained(
            model_checkpoint,
            cache_dir=cache_dir,
        )
        print(f"The model checkpoint are cached in '{cache_dir}'.")

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bart(input_ids, attention_mask=attention_mask, labels=labels)

        return Seq2SeqLMOutput(
            loss=outputs.loss,
            logits=outputs.logits
        )

    def generate(self, input_ids=None, attention_mask=None, num_beams=1, num_return_sequences=1):
        diversity_penalty = 0.0
        if num_return_sequences>1:
            diversity_penalty = 1.0
        return self.bart.generate(input_ids=input_ids, attention_mask=attention_mask,
                                  num_beams=num_beams,
                                  num_return_sequences=num_return_sequences,
                                  num_beam_groups=num_return_sequences,
                                  diversity_penalty=diversity_penalty)
