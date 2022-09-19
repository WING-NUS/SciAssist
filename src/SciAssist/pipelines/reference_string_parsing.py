import os
from typing import List, Tuple, Optional

from datasets import Dataset
from torch.utils.data import DataLoader

from SciAssist import BASE_OUTPUT_DIR, BASE_TEMP_DIR, BASE_CACHE_DIR
from SciAssist.datamodules.components.cora_label import LABEL_NAMES
from SciAssist.datamodules.components.cora_label import label2id
from SciAssist.pipelines.Pipeline import Pipeline
from SciAssist.utils.pad_for_token_level import pad, tokenize_and_align_labels
from SciAssist.utils.pdf2text import process_pdf_file, get_reference


class ReferenceStringParsing(Pipeline):

    def __init__(
            self, model_name: str = "default", device = "gpu",
            cache_dir = BASE_CACHE_DIR,
            output_dir = BASE_OUTPUT_DIR,
            temp_dir = BASE_TEMP_DIR
    ):
        super().__init__(task_name="reference-string-parsing", model_name=model_name, device=device, cache_dir=cache_dir)
        self.output_dir = output_dir
        self.temp_dir = temp_dir

    def predict(
            self, input: str, type: str = "pdf", dehyphen = False,
            output_dir = None,
            temp_dir = None,
    ):
        if output_dir is None:
            output_dir = self.output_dir
        if temp_dir is None:
            temp_dir = self.temp_dir

        if type in ["str","string"]:
            return self._predict_for_string(example=input, dehyphen=dehyphen)
        elif type in ["txt","text"]:
            return self._predict_for_text(filename=input, output_dir=output_dir, dehyphen=dehyphen)
        elif type == "pdf":
            print(temp_dir, self.temp_dir)
            return self._predict_for_pdf(filename=input, output_dir=output_dir, temp_dir=temp_dir, dehyphen=dehyphen)


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



    def _predict(self, examples: List[List[str]]) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        """
        Parse a list of tokens obtained from reference strings.

        Args:
            examples (`List[List[str]]`):
                The inputs for inference, where each item is a list of tokens.

        Returns:
            `Tuple[List[str], List[List[str]], List[List[str]]]`:
                Tagged strings, origin tokens and labels predicted by the model.

        """

        #Prepare the dataset
        dict_data = {"tokens": examples}
        dataset = Dataset.from_dict(dict_data)

        #Tokenize for Bert
        tokenized_example = dataset.map(
            lambda x: tokenize_and_align_labels(x, label2id),
            batched=True,
            remove_columns=dataset.column_names
        )
        dataloader = DataLoader(
            dataset=tokenized_example,
            batch_size=8,
            collate_fn=pad
        )
        results = []
        true_preds = []
        for batch in dataloader:
            #Predict the labels

            batch = self._to_device(batch)
            outputs = self.model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            #Convert ids to labels and
            #merge the labels according to origin tokens.
            preds = [[LABEL_NAMES[i] for i in pred if i<len(LABEL_NAMES)] for pred in preds]
            true_preds.extend(preds)

        tokens = examples

        #Generate the tagged strings.
        for i in range(len(tokens)):
            tagged_words = []
            for token, label in zip(tokens[i], true_preds[i]):
                tagged_word = f"<{label}>{token}</{label}>"
                tagged_words.append(tagged_word)
            result = " ".join(tagged_words)
            results.append(result)
        return results, tokens, true_preds


    def _predict_for_string(self, example: str, dehyphen: Optional[bool] = False) -> Tuple[str, List[str], List[str]]:
        """
        Parse a reference string.

        Args:
            example (`str`): The string to parse.
            dehyphen (`Optional[bool]`): Whether to remove '-', default to `False`.
        Returns:
           `Tuple[str, List[str], List[str]]`:
                Tagged string, origin tokens and labels predicted by the model.

        """
        # remove '-' in text
        examples = example.split("\n")
        if dehyphen == True:
            examples = [self._dehyphen_for_str(example) for example in examples]

        splitted_examples = [example.split() for example in examples]
        results, tokens, preds = self._predict(splitted_examples)

        return results, tokens, preds


    def _predict_for_text(
            self,
            filename: str,
            output_dir: Optional[str] = BASE_OUTPUT_DIR,
            dehyphen: Optional[bool] = False
    ) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        """

        Parse reference strings from a text and save the result as a text file.

        Args:
            filename (`str`): The path to the text file to predict.
            output_dir (`Optional[str]`): The directory to save the result file, default to `result/`.
            dehyphen (`Optional[bool]`): Whether to remove '-', default to `False`.

        Returns:
            `Tuple[List[str], List[List[str]], List[List[str]]]`:
                Tagged strings, origin tokens and labels predicted by the model.

        """
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.basename(filename)

        with open(filename, "r") as f:
            examples = f.readlines()

        # remove '-' in text
        if dehyphen == True:
            examples = [self._dehyphen_for_str(example) for example in examples]

        splitted_examples = [example.split() for example in examples]
        results, tokens, preds = self._predict(splitted_examples)
        with open(os.path.join(output_dir, f"{output_file[:-4]}_rs.txt"), "w") as output:
            for res in results:
                output.write(res + "\n")
        return results, tokens, preds


    def _predict_for_pdf(
            self,
            filename: str,
            output_dir: Optional[str] = BASE_OUTPUT_DIR,
            temp_dir: Optional[str] = BASE_TEMP_DIR,
            dehyphen: Optional[bool] = False
    ) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        """
        Parse reference strings from a PDF and save the result as a text file.

        Args:
            filename (`str`): The path to the pdf file to parse.
            output_dir (`Optional[str]`): The directory to save the result file, default to `result/`.
            temp_dir (`Optional[str]`): The diretorcy to save intermediate file, default to `temp/`.
            dehyphen (`Optional[bool]`): Whether to remove '-', default to `False`.

        Returns:
           `Tuple[List[str], List[List[str]], List[List[str]]]`:
                Tagged strings, origin tokens and labels predicted by the model.
        """

        #Convert PDF to JSON with doc2json.
        json_file = process_pdf_file(input_file=filename, temp_dir=temp_dir, output_dir=temp_dir)
        #Extract reference strings from JSON and save them in TEXT format.
        text_file = get_reference(json_file=json_file, output_dir=output_dir)
        return self._predict_for_text(text_file, output_dir=output_dir, dehyphen=dehyphen)
