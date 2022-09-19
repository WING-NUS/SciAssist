import os
from typing import List, Tuple, Optional

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from SciAssist import BASE_CACHE_DIR, BASE_TEMP_DIR, BASE_OUTPUT_DIR
from SciAssist.models.components.bart_tokenizer import BartTokenizer
from SciAssist.pipelines.Pipeline import Pipeline
from SciAssist.utils.pad_for_seq2seq import tokenize_and_align_labels
from SciAssist.utils.pdf2text import process_pdf_file, get_bodytext


class Summarization(Pipeline):

    def __init__(
            self, model_name: str = "default", device = "gpu",
            cache_dir = BASE_CACHE_DIR,
            output_dir = BASE_OUTPUT_DIR,
            temp_dir = BASE_TEMP_DIR
    ):
        super().__init__(task_name="summarization", model_name=model_name, device=device, cache_dir=cache_dir)
        self.output_dir = output_dir
        self.temp_dir = temp_dir

    def predict(
            self, input: str, type: str = "pdf",
            output_dir = None,
            temp_dir = None,
            num_beams = 1,
            save_results = False,
    ):
        if output_dir is None:
            output_dir = self.output_dir
        if temp_dir is None:
            temp_dir = self.temp_dir

        if type in ["str","string"]:
            return self._summarize_for_string(example=input)
        elif type in ["txt","text"]:
            return self._summarize_for_text(filename=input, output_dir=output_dir, save_results=save_results)
        elif type == "pdf":
            return self._summarize_for_pdf(filename=input, output_dir=output_dir, temp_dir=temp_dir, num_beams=num_beams, save_results=save_results)

    def _to_device(self, batch):
        if self.model_name in ["default", "bart-cnn-on-mup"]:
            return {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }


    def _summarize(
            self,
            examples: List[str],
            num_beams = 1
    ) -> List[str]:
        """
        Summarize each text in the list.
        Args:
            examples(`List[str]`): A list of texts to be summarized
            num_beams(`int`): Number of beams for beam search. 1 means no beam search.

        Returns:
            `List[str]`: A list of the summarization, each item corresponds to a text in the list.
        """

        # Prepare the dataset
        dict_data = {"text": examples}
        dataset = Dataset.from_dict(dict_data)

        # Tokenize for Bart, get input_ids and attention_masks
        tokenized_example = dataset.map(
            lambda x: tokenize_and_align_labels(x),
            batched=True,
            remove_columns=dataset.column_names
        )
        dataloader = DataLoader(
            dataset=tokenized_example,
            batch_size=8,
            collate_fn=DataCollatorForSeq2Seq(self.config["tokenizer"], model=self.config["model"], pad_to_multiple_of=8)
        )

        results = []
        for batch in dataloader:

            batch = self._to_device(batch)

            # Get token ids of summary
            pred = self.model.generate(batch["input_ids"], batch["attention_mask"],num_beams)
            # Convert token ids to text
            decoded_preds = BartTokenizer.batch_decode(pred, skip_special_tokens=True)

            results.extend(decoded_preds)

        return results

    def _summarize_for_string(
            self,
            example: str,
            num_beams = 1
    ) -> Tuple[str, str]:

        """
        Summarize a text in string format.

        Args:
            example (`str`): The string to summarize.
            num_beams (`int`): Number of beams for beam search. 1 means no beam search.
        Returns:
           `Tuple[str, str]`:
                Predicted summarization and source text.

        """
        res = self._summarize([example], num_beams)

        return  res[0], example


    def _summarize_for_text(
            self,
            filename: str,
            num_beams: int = 1,
            output_dir: Optional[str] = BASE_OUTPUT_DIR,
            save_results: Optional[bool] = False,
    ) -> Tuple[str, str]:
        """

        Summarize a document from a text file.

        Args:
            num_beams (`int`): Number of beams for beam search. 1 means no beam search.
            filename (`str`): The path to the input text file.

        Returns:
            `Tuple[str, str]`:
                Predicted summarization and source text.

        """

        with open(filename, "r") as f:
            examples = f.readlines()
        examples = ["".join(examples)]
        res = self._summarize(examples, num_beams)

        # Save predicted results as a text file
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.basename(filename)
            with open(os.path.join(output_dir, f"{output_file[:-4]}_summ.txt"), "w") as output:
                output.write(res[0] + "\n")

        return res[0], examples


    def _summarize_for_pdf(
            self,
            filename: str,
            temp_dir: Optional[str] = BASE_TEMP_DIR,
            output_dir: Optional[str] = BASE_OUTPUT_DIR,
            num_beams: int = 1,
            save_results: Optional[bool] = False
    ) -> Tuple[str, str]:
        """
        Summarize a document from a PDF file.

        Args:
            filename (`str`): The path to the pdf file to summarize.
            temp_dir (`Optional[str]`): The diretorcy to save intermediate file, default to `temp/`.
            output_dir (`Optional[str]`): The diretorcy to save text file, default to `output/`.

        Returns:
            `Tuple[str, str]`:
                Predicted summarization and source text.
        """

        # Convert PDF to JSON with doc2json.
        json_file = process_pdf_file(input_file=filename, temp_dir=temp_dir, output_dir=temp_dir)
        # Extract bodytext from pdf and save them in TEXT format.
        text_file = get_bodytext(json_file=json_file, output_dir=output_dir)
        # Do summarization
        return self._summarize_for_text(text_file, num_beams=num_beams, save_results=save_results)
