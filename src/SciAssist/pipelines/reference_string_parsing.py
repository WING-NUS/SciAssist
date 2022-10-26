# main developer: Yixi Ding <dingyixi@hotmail.com>

import json
import os
from typing import List, Tuple, Optional, Union, Dict

from datasets import Dataset
from transformers import PreTrainedTokenizer

from SciAssist import BASE_OUTPUT_DIR, BASE_TEMP_DIR
from SciAssist.datamodules.components.cora_label import LABEL_NAMES
from SciAssist.datamodules.components.cora_label import label2id
from SciAssist.pipelines.pipeline import Pipeline
from SciAssist.utils.pdf2text import process_pdf_file, get_reference
from SciAssist.utils.windows_pdf2text import windows_get_reference


class ReferenceStringParsing(Pipeline):
    """
    The pipeline for reference string parsing.

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
                  user or organization name, like `allenai/scibert_scivocab_uncased`.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                  using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
                - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                  single vocabulary file (like Bert or XLNet), e.g.: `./my_model_directory/vocab.txt`. (Not
                  applicable to all derived classes)

        model_max_length (`int`, *optional*): The max sequence length the model accepts.
    """

    def __init__(
            self, model_name: Optional[str] = "default", device: Optional[str] = "gpu",
            cache_dir = None,
            output_dir = None,
            temp_dir = None,
            tokenizer: PreTrainedTokenizer = None,
            checkpoint="allenai/scibert_scivocab_uncased",
            model_max_length=512,
            os_name=None,
    ):

        super().__init__(task_name="reference-string-parsing", model_name=model_name, device=device,
                         cache_dir=cache_dir, output_dir=output_dir, temp_dir=temp_dir)

        self.data_utils = self.data_utils(
            tokenizer=tokenizer,
            checkpoint=checkpoint,
            model_max_length=model_max_length
        )
        self.os_name = os_name if os_name != None else os.name

    def predict(
            self, input, type: str = "pdf", dehyphen=False,
            output_dir=None,
            temp_dir=None,
            save_results=True,
    ):

        """

        Args:
            input (`str` or `List[str]` or `os.PathLike`):
                Can be either:

                    - A string, the reference string to be parsed.
                    - A list of strings to be parsed.
                    - A path to a *.txt* file to be parsed. Each line of the source file contains a
                      reference string.
                    - A path to a *.pdf* file to be parsed, a raw scientific document without processing.
                      The pipeline will automatically extract the reference strings from the pdf.

            type (`str`, default to `pdf`):
                The type of input, can be either:

                    - `str` or `string`.
                    - `text`or `txt` for a .txt file.
                    - `pdf` for a pdf file. This is the default value.

            dehyphen (`bool`, default to `False`):
                Whether to remove hyphens in raw text.
            output_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which the predicted results files should be stored.
                If not provided, it will use the `output_dir` set for the pipeline.
            temp_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory which holds temporary files such as `.tei.xml`.
                If not provided, it will use the `temp_dir` set for the pipeline.
            save_results (`bool`, default to `True`):
                Whether to save the results in a *.json* file.
                **Note**: This is invalid when `type` is set to `str` or `string`.

        Returns:
            `List[Dict]`: [{"tagged_text": tagged_text, "tokens": tokens_list ,"tags": tags_list } , ... ]


        Examples:

            >>> from SciAssist import ReferenceStringParsing
            >>> pipeline = ReferenceStringParsing()
            >>> pipeline.predict(
            ...     "Waleed Ammar, Matthew E. Peters, Chandra Bhagavat- ula, and Russell Power. 2017. The ai2 system at semeval-2017 task 10 (scienceie): semi-supervised end-to-end entity and relation extraction. In ACL workshop (SemEval).",
            ...     type="str"
            ... )
            [{'tagged_text': '<author>Waleed</author> <author>Ammar,</author> <author>Matthew</author> <author>E.</author> <author>Peters,</author> <author>Chandra</author> <author>Bhagavat-</author> <author>ula,</author> <author>and</author> <author>Russell</author> <author>Power.</author> <date>2017.</date> <title>The</title> <title>ai2</title> <title>system</title> <title>at</title> <title>semeval-2017</title> <title>task</title> <title>10</title> <title>(scienceie):</title> <title>semi-supervised</title> <title>end-to-end</title> <title>entity</title> <title>and</title> <title>relation</title> <title>extraction.</title> <booktitle>In</booktitle> <booktitle>ACL</booktitle> <booktitle>workshop</booktitle> <booktitle>(SemEval).</booktitle>',
            'tokens': ['Waleed', 'Ammar,', 'Matthew', 'E.', 'Peters,', 'Chandra', 'Bhagavat-', 'ula,', 'and', 'Russell', 'Power.', '2017.', 'The', 'ai2', 'system', 'at', 'semeval-2017', 'task', '10', '(scienceie):', 'semi-supervised', 'end-to-end', 'entity', 'and', 'relation', 'extraction.', 'In', 'ACL', 'workshop', '(SemEval).'],
            'tags': ['author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'date', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'booktitle', 'booktitle', 'booktitle', 'booktitle']}]

        """
        if output_dir is None:
            output_dir = self.output_dir
        if temp_dir is None:
            temp_dir = self.temp_dir

        if type in ["str", "string"]:
            results = self._predict_for_string(example=input, dehyphen=dehyphen)
        elif type in ["txt", "text"]:
            results = self._predict_for_text(filename=input, dehyphen=dehyphen)
        elif type == "pdf":
            results = self._predict_for_pdf(filename=input, output_dir=output_dir, temp_dir=temp_dir, dehyphen=dehyphen)
        # Save predicted results as a text file
        if save_results and type not in ["str", "string"]:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.basename(input)
            with open(os.path.join(output_dir, f"{output_file[:-4]}_rsp.json"), "w") as output:
                for res in results:
                    output.write(json.dumps(res) + "\n")

        return results

    def _dehyphen_for_str(self, text: str):
        text = text.replace("- ", "")
        text = text.replace("-", " ")
        return text

    def _to_device(self, batch):
        if self.model_name in ["default", "scibert-on-cora"]:
            return {
                "input_ids": batch["input_ids"].to(self.device),
                "token_type_ids": batch["token_type_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
                "token_mapping": batch["token_mapping"].to(self.device)
            }

    def _predict(self, examples: List[List[str]]) -> List[Dict]:
        """
        Parse a list of tokens obtained from reference strings.

        Args:
            examples (`List[List[str]]`):
                The inputs for inference, where each item is a list of tokens.

        Returns:
            `List[Dict]`:
                Tagged strings, origin tokens and labels predicted by the model.

        """

        # Prepare the dataset
        dict_data = {"tokens": examples}
        dataset = Dataset.from_dict(dict_data)

        dataloader = self.data_utils.get_dataloader(dataset, label2id=label2id)

        results = []
        true_preds = []
        for batch in dataloader:
            # Predict the labels

            batch = self._to_device(batch)
            outputs = self.model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            # Convert ids to labels and
            # merge the labels according to origin tokens.
            preds = [[LABEL_NAMES[i] for i in pred if i < len(LABEL_NAMES)] for pred in preds]
            true_preds.extend(preds)

        tokens = examples

        # Generate the tagged strings.
        for i in range(len(tokens)):
            tagged_words = []
            for token, label in zip(tokens[i], true_preds[i]):
                tagged_word = f"<{label}>{token}</{label}>"
                tagged_words.append(tagged_word)
            result = " ".join(tagged_words)
            results.append(
                {
                    "tagged_text": result,
                    "tokens": tokens[i],
                    "tags": true_preds[i],
                }
            )
        return results

    def _predict_for_string(self, example: Union[str, List[str]], dehyphen: Optional[bool] = False) -> List[Dict]:
        """
        Parse a reference string.

        Args:
            example (`Union[str, List[str]]`): The string to parse.
            dehyphen (`Optional[bool]`): Whether to remove '-', default to `False`.
        Returns:
           `List[Dict]`:
                Tagged string, origin tokens and labels predicted by the model.

        """

        if isinstance(example, list):
            examples = example
        else:
            examples = [example]

        # remove '-' in text
        if dehyphen == True:
            examples = [self._dehyphen_for_str(example) for example in examples]

        splitted_examples = [example.split() for example in examples]
        results = self._predict(splitted_examples)

        return results

    def _predict_for_text(
            self,
            filename: str,
            dehyphen: Optional[bool] = False,
    ) -> List[Dict]:
        """

        Parse reference strings from a text and save the result as a text file.

        Args:
            filename (`str`): The path to the text file to predict.
            dehyphen (`Optional[bool]`): Whether to remove '-', default to `False`.

        Returns:
            `List[Dict]`:
                Tagged strings, origin tokens and labels predicted by the model.

        """

        with open(filename, "r") as f:
            examples = f.readlines()

        # remove '-' in text
        if dehyphen == True:
            examples = [self._dehyphen_for_str(example) for example in examples]

        splitted_examples = [example.split() for example in examples]
        results = self._predict(splitted_examples)

        return results

    def _predict_for_pdf(
            self,
            filename: str,
            output_dir: Optional[str] = BASE_OUTPUT_DIR,
            temp_dir: Optional[str] = BASE_TEMP_DIR,
            dehyphen: Optional[bool] = False,
    ) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        """
        Parse reference strings from a PDF and save the result as a text file.

        Args:
            filename (`str`): The path to the pdf file to parse.
            output_dir (`Optional[str]`): The directory to save the result file, default to `result/`.
            temp_dir (`Optional[str]`): The diretorcy to save intermediate file, default to `temp/`.
            dehyphen (`Optional[bool]`): Whether to remove '-', default to `False`.

        Returns:
           `List[Dict]`:
                Tagged strings, origin tokens and labels predicted by the model.
        """
        if self.os_name == "posix":
            # Convert PDF to JSON with doc2json.
            json_file = process_pdf_file(input_file=filename, temp_dir=temp_dir, output_dir=temp_dir)
            # Extract reference strings from JSON and save them in TEXT format.
            text_file = get_reference(json_file=json_file, output_dir=output_dir)
        elif self.os_name == "nt":
            text_file = windows_get_reference(path=filename, output_dir=output_dir)


        return self._predict_for_text(text_file, dehyphen=dehyphen)
