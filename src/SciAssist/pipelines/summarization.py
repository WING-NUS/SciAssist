import os
import timeit
from typing import List, Tuple, Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from SciAssist.models.components.bart_summarization import BartForSummarization
from SciAssist.models.components.bart_tokenizer import bart_tokenizer
from SciAssist.utils.pad_for_seq2seq import tokenize_and_align_labels
from SciAssist.utils.pdf2text import process_pdf_file, get_bodytext

ROOT_DIR = os.getcwd()
BASE_OUTPUT_DIR = os.path.join(ROOT_DIR, "output/result")
BASE_TEMP_DIR = os.path.join(ROOT_DIR,"output/.temp")
BASE_CACHE_DIR = os.path.join(ROOT_DIR, ".cache")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BartForSummarization(
    model_checkpoint="facebook/bart-large-cnn",
    cache_dir=BASE_CACHE_DIR
)

model.load_state_dict(torch.load("models/mup/bart-large-cnn-e5.pt"))

model.eval()
if torch.cuda.is_available():
    model.cuda()
    print("CUDA is available.")
else:
    print("Not use CUDA.")


def to_device(batch):

    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }


def summarize(
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
        collate_fn=DataCollatorForSeq2Seq(bart_tokenizer, model=BartForSummarization, pad_to_multiple_of=8)
    )

    results = []
    for batch in dataloader:

        batch = to_device(batch)

        # Get token ids of summary
        pred = model.generate(batch["input_ids"], batch["attention_mask"],num_beams)
        # Convert token ids to text
        decoded_preds = bart_tokenizer.batch_decode(pred, skip_special_tokens=True)

        results.extend(decoded_preds)

    return results

def summarize_for_string(
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
            Source text and predicted summarization.

    """
    res = summarize([example], num_beams)

    return  example, res[0]


def summarize_for_text(
        filename: str,
        num_beams: int = 1
) -> Tuple[str, str]:
    """

    Summarize a document from a text file.

    Args:
        num_beams (`int`): Number of beams for beam search. 1 means no beam search.
        filename (`str`): The path to the input text file.

    Returns:
        `Tuple[str, str]`:
            Source text and predicted summarization.

    """

    start_time = timeit.default_timer()

    with open(filename, "r") as f:
        examples = f.readlines()
    examples = ["".join(examples)]
    res = summarize(examples, num_beams)

    total_time = timeit.default_timer() - start_time
    print("total_time:", total_time)
    return examples, res[0]


def summarize_for_pdf(
        filename: str,
        temp_dir: Optional[str] = BASE_TEMP_DIR,
        output_dir: Optional[str] = BASE_OUTPUT_DIR
) -> Tuple[str, str]:
    """
    Summarize a document from a PDF file.

    Args:
        filename (`str`): The path to the pdf file to summarize.
        temp_dir (`Optional[str]`): The diretorcy to save intermediate file, default to `temp/`.
        output_dir (`Optional[str]`): The diretorcy to save text file, default to `output/`.

    Returns:
        `Tuple[str, str]`:
            Source text and predicted summarization.
    """

    # Convert PDF to JSON with doc2json.
    json_file = process_pdf_file(input_file=filename, temp_dir=temp_dir, output_dir=temp_dir)
    # Extract bodytext from pdf and save them in TEXT format.
    text_file = get_bodytext(json_file=json_file, output_dir=output_dir)
    # Do summarization
    return summarize_for_text(text_file)
