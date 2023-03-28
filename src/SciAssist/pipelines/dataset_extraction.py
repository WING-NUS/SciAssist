import json
import os
import re
import torch
import numpy as np
import nltk
from typing import List, Tuple, Optional, Union, Dict
from transformers import AutoTokenizer

from SciAssist import BASE_OUTPUT_DIR, BASE_TEMP_DIR
from SciAssist.pipelines.pipeline import Pipeline
from SciAssist.utils.pdf2text import process_pdf_file, get_bodytext
from SciAssist.utils.windows_pdf2text import windows_get_bodytext


class DatasetExtraction(Pipeline):
    """

    The pipeline for dataset extraction.

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
            tokenizer = None,
            checkpoint = "allenai/scibert_scivocab_uncased",
            model_max_length = 128,
            os_name = None,
    ):

        super().__init__(task_name = "dataset-extraction", model_name = model_name, device = device,
                         cache_dir = cache_dir, output_dir = output_dir, temp_dir = temp_dir)

        # print('load my model')
        # self.model.load_state_dict(torch.load("/home/linxiao/SciAssist-scibert-0223/src/models/scibert_ner/2023-02-27_10-40-00/roberta-base"))
        # self.model.load_state_dict(torch.load("/home/linxiao/SciAssist-scibert-0223/src/models/scibert_ner/2023-03-24_16-57-27/roberta-base"))
        
        # print('ok')

        self.data_utils = self.data_utils(
            tokenizer = tokenizer,
            checkpoint = checkpoint,
            model_max_length = model_max_length
        )

        self.tokenizer = self.data_utils.tokenizer

        self.os_name = os_name if os_name != None else os.name

        nltk.download('punkt')

    def _to_device(self, batch):
        if self.model_name in ["default"]:
            return {
                "input_subwords": batch['input_subwords'].to(self.device),
                "input_token_start_indexs": batch['input_token_start_indexs'].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }

    def extract(
            self, input, type: str = "pdf",
            output_dir = None,
            temp_dir = None,
            save_results = True,
    ):
        """

        Args:
            input (`str` or `List[str]` or `os.PathLike`):
                Can be either:
                    - A list of strings (sentences) to be extracted dataset mentions from.
                    - A path to a *.txt* file to be extracted dataset mentions from.
                    - A path to a *.pdf* file to be extracted dataset mentions from, a raw scientific document without processing. The pipeline will automatically extract the body text from the pdf.

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

            save_results (`bool`, default to `True`):
                Whether to save the results in a *.json* file.
                **Note**: This is invalid when `type` is set to `str` or `string`.

        Returns:
            `Dict`: { "dataset_mentions": [[dataset_mentions1], sentence1], [[dataset_mentions2], sentence2], ...], "text": [sentence1, sentence2, ...]}
            Please note that only positive sentences will be shown in Dict['dataset_mentions'], while Dict['text'] will show all sentences.

        Examples:
             >>> from SciAssist import DatasetExtraction
             >>> pipeline = DatasetExtraction()
             >>> res = pipeline.predict('N18-3011.pdf', type="pdf")
             >>> res["dataset_mentions"]

        """

        if output_dir is None:
            output_dir = self.output_dir
        if temp_dir is None:
            temp_dir = self.temp_dir

        if type in ["str", "string"]:
            results = self._extract_for_string(example=input)
        elif type in ["txt", "text"]:
            results = self._extract_for_text(filename=input)
        elif type == "pdf":
            results = self._extract_for_pdf(filename=input, output_dir=output_dir, temp_dir=temp_dir)

        # Save predicted results as a text file
        if save_results and type not in ["str", "string"]:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.basename(input)
            with open(os.path.join(output_dir, f"{output_file[:-4]}_de.json"), "w") as output:
                output.write(json.dumps(results) + "\n")

        return results


    def _extract(
            self,
            examples: List[str]
    ):
        """

        Extract dataset mentions from each sentence in the list.

        Args:
            examples(`List[str]`): A list of sentences to be extracted dataset mentions.

        Returns:
            `Dict`: { "dataset_mentions": [[dataset_mentions1], sentence1], [[dataset_mentions2], sentence2], ...], "text": [sentence1, sentence2, ...]}

        """

        results = {}

        dataset_mentions = []
        dataset_set = set()
        dataset = {'data':{'text': examples}}
        dataloader = self.data_utils.get_dataloader(dataset)

        pred_output = []
        for batch in dataloader:
            batch = self._to_device(batch)
            batch_output = self.model(**batch)
            ner_output = batch_output[0]
            ner_output = ner_output.detach().cpu().numpy()
            pred_output.extend([[idx for idx in indices] for indices in np.argmax(ner_output, axis=2)]) # pred_output: List[List[tag_idx]] num_sentences * each_sentence_length

        for i, sentence in enumerate(pred_output): # for each sentence
            entity_indexes = self.find_entity_indexes(sentence) # List of entity_index pairs: [[2,2],[5,9],[16,21]]
            if len(entity_indexes) != 0:
                tokens = examples[i].strip().split(' ')
                entities = self.find_entities(entity_indexes, tokens) # List of mentions in this sentence
                dataset_set.update(entities)
                dataset_mentions.append([entities, examples[i]]) # [[List of mentions in sentence1, sentence1], [List of mentions in sentence2, sentence2], ...]

        results['all_dataset_mentions'] = list(dataset_set)
        results['dataset_mentions'] = dataset_mentions
        results['text'] = examples

        return results


    def _extract_for_string(
            self,
            example: Union[str, List[str]]
    ):
        """

        Extract dataset mentions from each sentence in the string or list.

        Args:
            example (`Union[str, List[str]]`): The string to be extracted dataset mentions from.

        Returns:
            `Dict`: { "dataset_mentions": [[dataset_mentions1], sentence1], [[dataset_mentions2], sentence2], ...], "text": [sentence1, sentence2, ...]}

        """

        if isinstance(example, list):
            examples = example
        else:
            examples = [example]

        results = self._extract(examples)

        return results


    def _extract_for_text(
            self,
            filename: str
    ):
        """

        Extract dataset mentions from each sentence of a text file and save the result as a text file.

        Args:
            filename (`str`): The path to the text file to be extracted dataset mentions from.

        Returns:
            `Dict`: { "dataset_mentions": [[dataset_mentions1], sentence1], [[dataset_mentions2], sentence2], ...], "text": [sentence1, sentence2, ...]}

        """

        with open(filename, "r") as f:
            text = f.readlines()
            sentences = []
            for i in text:
                sentences.append(i.strip())
        f.close()

        results = self._extract(sentences)

        return results


    def _extract_for_pdf(
            self,
            filename: str,
            output_dir: Optional[str] = BASE_OUTPUT_DIR,
              temp_dir: Optional[str] = BASE_TEMP_DIR
    ):
        """

        Extract dataset mentions from each sentence of a pdf file and save the result as a text file.

        Args:
            filename (`str`): The path to the pdf file to parse.
            output_dir (`Optional[str]`): The directory to save the result file, default to `result/`.
            temp_dir (`Optional[str]`): The diretorcy to save intermediate file, default to `temp/`.

        Returns:
           `List[Dict]`:
                Tagged strings, origin tokens and labels predicted by the model.

        """
        
        if self.os_name == "posix":
            # Convert PDF to JSON with doc2json.
            json_file = process_pdf_file(input_file=filename, temp_dir=temp_dir, output_dir=temp_dir)
            # Extract bodytext from pdf and save them in TEXT format.
            text_file = get_bodytext(json_file=json_file, output_dir=output_dir)
        elif self.os_name == "nt":
            text_file = windows_get_bodytext(path=filename, output_dir=output_dir)

        with open(text_file, "r") as f:
            text = f.readlines()
            sentences = nltk.sent_tokenize(text[0])
        f.close()

        results = self._extract(sentences)

        return results


    def find_entity_indexes(self, lst):
        # {0: 'B-DATA', 1: 'I-DATA', 2: 'O'}
        indexes = []
        i = 0
        for i in range(len(lst)):
            tmp = []
            if lst[i] == 0:
                tmp = [i,i]
                j = i
                while j < len(lst):
                    if j+1 == len(lst) or lst[j+1] != 1:
                        indexes.append(tmp)
                        break                    
                    else:
                        j += 1
                        tmp = [i,j]

        return indexes


    def find_entities(self, idx_lst, tokens):
        dataset_mentions = []
        for idx_pair in idx_lst:
            if idx_pair[0] == idx_pair[1]:
                dataset_mentions.append(tokens[idx_pair[0]])
            else:
                dataset_mentions.append(' '.join(tokens[idx_pair[0]:idx_pair[1] + 1]))

        return dataset_mentions


