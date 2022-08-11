import os
import timeit
from typing import List, Tuple, Optional

import numpy
import torch
from torch.utils.data import DataLoader
from collections import Counter
from datasets import Dataset

from src.models.components.bert_token_classifier import BertTokenClassifier
from src.datamodules.components.cora_label import LABEL_NAMES
from src.datamodules.components.cora_label import label2id
from src.utils.pdf2text import process_pdf_file, get_reference
from src.utils.pad_for_token_level import pad, tokenize_and_align_labels

ROOT_DIR = os.getcwd()
BASE_OUTPUT_DIR = os.path.join(ROOT_DIR, "output/result")
BASE_TEMP_DIR = os.path.join(ROOT_DIR,"output/.temp")
BASE_CACHE_DIR = os.path.join(ROOT_DIR, ".cache")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertTokenClassifier(
    model_checkpoint="allenai/scibert_scivocab_uncased",
    output_size=13,
    cache_dir=BASE_CACHE_DIR
)

model.load_state_dict(torch.load("models/default/scibert-uncased.pt"))

model.eval()
if torch.cuda.is_available():
    model.cuda()
    print("CUDA is available.")
else:
    print("Not use CUDA.")


def dehyphen_for_str(text: str):
    text = text.replace("- ", "")
    text = text.replace("-", " ")
    return text


def to_device(batch):

    return {
        "input_ids": batch["input_ids"].to(device),
        "token_type_ids": batch["token_type_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "h_mapping": batch["h_mapping"].to(device)
    }



def predict(examples: List[List[str]]) -> Tuple[List[str], List[List[str]], List[List[str]]]:
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

        batch = to_device(batch)
        outputs = model(**batch)
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


def predict_for_string(example: str, dehyphen: Optional[bool] = False) -> Tuple[str, List[str], List[str]]:
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
        examples = [dehyphen_for_str(example) for example in examples]

    splitted_examples = [example.split() for example in examples]
    results, tokens, preds = predict(splitted_examples)

    return results, tokens, preds


def predict_for_text(
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

    start_time = timeit.default_timer()

    with open(filename, "r") as f:
        examples = f.readlines()

    # remove '-' in text
    if dehyphen == True:
        examples = [dehyphen_for_str(example) for example in examples]

    splitted_examples = [example.split() for example in examples]
    results, tokens, preds = predict(splitted_examples)
    with open(os.path.join(output_dir, f"{output_file[:-4]}_result.txt"), "w") as output:
        for res in results:
            output.write(res + "\n")
    total_time = timeit.default_timer() - start_time
    print("total_time:", total_time)
    return results, tokens, preds


def predict_for_pdf(
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
    return predict_for_text(text_file, output_dir=output_dir, dehyphen=dehyphen)
