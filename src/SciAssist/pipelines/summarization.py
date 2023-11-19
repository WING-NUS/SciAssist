# main developer: Yixi Ding <dingyixi@hotmail.com>
import json
import math
import os
from typing import List, Tuple, Optional, Dict

from SciAssist import BASE_TEMP_DIR, BASE_OUTPUT_DIR
from SciAssist.pipelines.pipeline import Pipeline
from SciAssist.pipelines.testing_pipeline import test
from SciAssist.utils.pdf2text import process_pdf_file, get_bodytext
from SciAssist.utils.windows_pdf2text import windows_get_bodytext
from datasets import Dataset


class Summarization(Pipeline):
    """
    The pipeline for single document summarization.

    Args:
        model_name (`str`, *optional*):
            A string, the *model name* of a pretrained model provided for this task.
        device (`str`, *optional*):
            A string, `cpu` or `gpu`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model should be
            cached if the standard cache should not be used.
        output_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which the predicted results files should be stored.
        temp_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory which holds temporary files such as `.tei.xml`.
        tokenizer (PreTrainedTokenizer, *optional*):
            A specific tokenizer.
        checkpoint (`str` or `os.PathLike`, *optional*):
            A checkpoint for the tokenizer. You can also specify the `checkpoint` while
            using the default tokenizer.
            Can be either:

                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                  Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                  user or organization name, like `facebook/bart-large-cnn`.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                  using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
                - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                  single vocabulary file (like Bert or XLNet), e.g.: `./my_model_directory/vocab.txt`. (Not
                  applicable to all derived classes)

        model_max_length (`int`, *optional*): The max sequence length the model accepts.
        max_source_length (`int`, *optional*): The max length of the input text.
        max_target_length (`int`, *optional*): The max length of the generated summary.
    """

    def __init__(
            self, model_name: str = "default", device="gpu",
            task_name = "controlled-summarization",
            cache_dir=None,
            output_dir=None,
            temp_dir=None,
            tokenizer=None,
            checkpoint="google/flan-t5-base",
            model_max_length=1024,
            max_source_length=1024,
            max_target_length=500,
            os_name=None,
    ):
        super().__init__(task_name=task_name, model_name=model_name, checkpoint=checkpoint,device=device,
                         cache_dir=cache_dir, output_dir=output_dir, temp_dir=temp_dir, )

        self.data_utils = self.data_utils(
            tokenizer=tokenizer,
            model_class=self.config["model"],
            checkpoint=checkpoint,
            model_max_length=model_max_length,
            max_source_length=max_source_length,
            max_target_length=max_target_length
        )
        self.tokenizer = self.data_utils.tokenizer
        self.os_name = os_name if os_name != None else os.name

    def predict(
            self, input: str, type: str = "pdf",
            output_dir=None,
            temp_dir=None,
            num_beams=5,
            num_return_sequences=1,
            save_results=True,
            length = None,
            keywords: List[str] = None,
            top_k=0,
            max_length=500,
            do_sample=False,
    ):
        """

        Args:
            input (`str` or `List[str]` or `os.PathLike`):
            Can be either:

                   - A string, the reference string to be parsed.
                   - A list of strings to be parsed.
                   - A path to a *.txt* file to be summarized.
                   - A path to a *.pdf* file to be summarized, a raw scientific document without processing.
                     The pipeline will automatically extract the body text from the pdf.

            type (`str`, default to `pdf`):
                The type of input, can be either:

                    - `str` or `string`.
                    - `text`or `txt` for a .txt file.
                    - `pdf` for a pdf file. This is the default value.

            output_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which the predicted results files should be stored.
                If not provided, it will use the `output_dir` set for the pipeline.
            temp_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory which holds temporary files such as `.tei.xml`.
                If not provided, it will use the `temp_dir` set for the pipeline.
            num_beams (`int`, *optional*):
                Number of beams for beam search. 1 means no beam search.
                `num_beams` should be divisible by `num_return_sequences` for group beam search.
            num_return_sequences(`int`):
                The number of independently computed returned sequences for each element in the batch.
            save_results (`bool`, default to `True`):
                Whether to save the results in a *.json* file.
                **Note**: This is invalid when `type` is set to `str` or `string`.

        Returns:
            `Dict`: { "summary": [summary1, summary2, ...], "raw_text": raw_text }


        Examples:

             >>> from SciAssist import Summarization
             >>> pipeline = Summarization()
             >>> res = pipeline.predict('N18-3011.pdf', type="pdf", num_beams=4, num_return_sequences=2)
             >>> res["summary"]
             ['The paper proposes a method for extracting structured information from scientific documents into the literature graph. The paper describes the attributes associated with nodes and edges of different types in the graph, and describes how to extract the entities mentioned in paper text. The method is evaluated on three tasks: sequence labeling, entity linking and relation extraction. ',
             'The paper proposes a method for extracting structured information from scientific documents into the literature graph. The paper describes the attributes associated with nodes and edges of different types in the graph, and describes how to extract the entities mentioned in paper text. The method is evaluated on three tasks: sequence labeling, entity linking and relation extraction.  ']

        """

        if output_dir is None:
            output_dir = self.output_dir
        if temp_dir is None:
            temp_dir = self.temp_dir

        if type in ["str", "string"]:
            results = self._summarize_for_string(example=input, num_beams=num_beams,
                                                 num_return_sequences=num_return_sequences,length=length, keywords=keywords,top_k=top_k,max_length=max_length,do_sample=do_sample)
        elif type in ["txt", "text"]:
            results = self._summarize_for_text(filename=input, num_beams=num_beams,
                                               num_return_sequences=num_return_sequences,length=length, keywords=keywords,top_k=top_k,max_length=max_length,do_sample=do_sample)
        elif type == "pdf":
            results = self._summarize_for_pdf(filename=input, output_dir=output_dir, temp_dir=temp_dir,
                                              num_beams=num_beams, num_return_sequences=num_return_sequences,
                                              length=length, keywords=keywords,top_k=top_k,max_length=max_length,do_sample=do_sample)

        # Save predicted results as a text file
        if save_results and type not in ["str", "string"]:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.basename(input)
            with open(os.path.join(output_dir, f"{output_file[:-4]}_summ.json"), "w") as output:
                output.write(json.dumps(results) + "\n")

        return results

    def _to_device(self, batch):
        if self.model_name in ["default", "bart-cnn-on-mup", "flan-t5", "t5"]:
            return {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }

    def _summarize(
            self,
            examples: List[str],
            num_beams=1,
            num_return_sequences=1,
            length=100,
            keywords=None,
            top_k=top_k,
            max_length=max_length,
            do_sample=do_sample,
    ) -> List[str]:
        """
        Summarize each text in the list.
        Args:
            examples(`List[str]`): A list of texts to be summarized
            num_beams(`int`): Number of beams for beam search. 1 means no beam search.
            num_return_sequences(`int`): The number of independently computed returned sequences for each element in the batch.
        Returns:
            `List[str]`: A list of the summarization, each item corresponds to a text in the list.
        """

        # Prepare the dataset
        dict_data = {"text": examples, "length": [length]*len(examples), "keywords": [keywords]*len(examples) }
        dataset = Dataset.from_dict(dict_data)

        # Tokenize for Bart, get input_ids and attention_masks
        dataloader = self.data_utils.get_dataloader(dataset)
        results = []
        for batch in dataloader:
            batch = self._to_device(batch)

            # Get token ids of summary
            pred = self.model.generate(batch["input_ids"], batch["attention_mask"], num_beams, num_return_sequences,top_k=top_k,max_length=max_length,do_sample=do_sample)
            # Convert token ids to text
            decoded_preds = self.tokenizer.batch_decode(pred, skip_special_tokens=True)

            results.extend(decoded_preds)
        return results

    def _summarize_for_string(
            self,
            example: str,
            num_beams=1,
            num_return_sequences=1,
            length=100,
            keywords=None,
            top_k=top_k,
            max_length=max_length,
            do_sample=do_sample,
    ) -> Tuple[str, str]:

        """
        Summarize a text in string format.

        Args:
            example (`str`): The string to summarize.
            num_beams (`int`): Number of beams for beam search. 1 means no beam search.
            num_return_sequences(`int`): The number of independently computed returned sequences for each element in the batch.
        Returns:
           `Tuple[str, str]`:
                Predicted summarization and source text.

        """
        num = 10
        res = self._summarize([example], num_beams, num_return_sequences,length=length, keywords=keywords,top_k=top_k,max_length=max_length,do_sample=do_sample)
        if length is not None:
            num = 5*math.ceil(length/50)
        # if keywords is not None:
        #     example = extract_related_sentences(example,keywords[0],num)
        return {"summary": res, "raw_text": example}

    def _summarize_for_text(
            self,
            filename: str,
            num_beams: int = 1,
            num_return_sequences: int = 1,
            length=100,
            keywords=None,
            top_k=top_k,
            max_length=max_length,
            do_sample=do_sample,
    ) -> Tuple[str, str]:
        """

        Summarize a document from a text file.

        Args:
            num_beams (`int`): Number of beams for beam search. 1 means no beam search.
            filename (`str`): The path to the input text file.
            num_return_sequences(`int`): The number of independently computed returned sequences for each element in the batch.

        Returns:
            `Tuple[str, str]`:
                Predicted summarization and source text.

        """
        num = 10
        if length is not None:
            num = 5 * math.ceil(length / 50)
        with open(filename, "r") as f:
            examples = f.readlines()
        examples = [" ".join(examples)]
        res = self._summarize(examples, num_beams, num_return_sequences,length=length,keywords=keywords,top_k=top_k,max_length=max_length,do_sample=do_sample)
        # if keywords is not None:
        #     examples = [extract_related_sentences(examples[0], keywords[0],num)]
        return {"summary": res, "raw_text": examples[0]}

    def _summarize_for_pdf(
            self,
            filename: str,
            temp_dir: Optional[str] = BASE_TEMP_DIR,
            output_dir: Optional[str] = BASE_OUTPUT_DIR,
            num_beams: int = 1,
            num_return_sequences=1,
            length = 100,
            keywords = None,
            top_k=top_k,
            max_length=max_length,
            do_sample=do_sample,
    ) -> Dict:
        """
        Summarize a document from a PDF file.

        Args:
            filename (`str`): The path to the pdf file to summarize.
            temp_dir (`Optional[str]`): The diretorcy to save intermediate file, default to `temp/`.
            output_dir (`Optional[str]`): The diretorcy to save text file, default to `output/`.
            num_return_sequences(`int`): The number of independently computed returned sequences for each element in the batch.

        Returns:
            `Dict`:
                Predicted summarization and source text.
        """
        if self.os_name == "posix":
            # Convert PDF to JSON with doc2json.
            json_file = process_pdf_file(input_file=filename, temp_dir=temp_dir, output_dir=temp_dir)
            # Extract bodytext from pdf and save them in TEXT format.
            text_file = get_bodytext(json_file=json_file, output_dir=output_dir)
        elif self.os_name == "nt":
            text_file = windows_get_bodytext(path=filename, output_dir=output_dir)

        # Do summarization
        return self._summarize_for_text(text_file, num_beams=num_beams, num_return_sequences=num_return_sequences, length=length, keywords=keywords,top_k=top_k,max_length=max_length,do_sample=do_sample)


    def evaluate(self):

        return test()