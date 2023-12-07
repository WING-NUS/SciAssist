from torch import nn
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import Seq2SeqLMOutput


class FlanT5ForSummarization(nn.Module):
    def __init__(
        self,
        model_checkpoint: str = "google/flan-t5-base",
        cache_dir: str = ".cache",
        save_name: str = "flan-t5-mup.pt",
        model_dir: str = "pretrained"
    ):
        super().__init__()
        self.save_name = save_name
        self.model_dir = model_dir
        self.flant5 = AutoModelForSeq2SeqLM.from_pretrained(
            model_checkpoint,
            cache_dir=cache_dir,
        )
        print(f"The model checkpoint are cached in '{cache_dir}'.")

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.flant5(input_ids, attention_mask=attention_mask, labels=labels)

        return Seq2SeqLMOutput(
            loss=outputs.loss,
            logits=outputs.logits
        )

    def generate(self, input_ids=None, attention_mask=None, num_beams=1, num_return_sequences=1, top_k=0, max_length=500, do_sample=False):
        diversity_penalty = 0.0
        if num_return_sequences>1:
            diversity_penalty = 1.0
        return self.flant5.generate(input_ids=input_ids, attention_mask=attention_mask,
                                    num_beams=num_beams,
                                    num_return_sequences=num_return_sequences,
                                    diversity_penalty = diversity_penalty,
                                    top_k=top_k,
                                    max_length=max_length,
                                    do_sample=do_sample,)

