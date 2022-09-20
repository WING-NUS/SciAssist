from typing import Dict

import torch

from SciAssist import BASE_CACHE_DIR
from SciAssist.models.components.bart_summarization import BartForSummarization
from SciAssist.models.components.bert_token_classifier import BertForTokenClassifier
from SciAssist.utils.data_utils import (
    DataUtilsForTokenClassification,
    DataUtilsForSeq2Seq,
)

# Provided models for each task
TASKS = {
    "reference-string-parsing": {
        "scibert-on-cora":  {
            "model": BertForTokenClassifier,
            "model_dict_url": "https://huggingface.co/spaces/wing-nus/SciAssist/resolve/main/scibert-uncased.pt",
            "data_utils": DataUtilsForTokenClassification
        },
        "default": {
            "model": BertForTokenClassifier,
            "model_dict_url": "https://huggingface.co/spaces/wing-nus/SciAssist/resolve/main/scibert-uncased.pt",
            "data_utils": DataUtilsForTokenClassification,
        },
    },

    "summarization": {
        "bart-cnn-on-mup": {
            "model": BartForSummarization,
            "model_dict_url": "https://huggingface.co/spaces/wing-nus/SciAssist/resolve/main/bart-large-cnn-e5.pt",
            "data_utils": DataUtilsForSeq2Seq,
        },
        "default": {
            "model": BartForSummarization,
            "model_dict_url": "https://huggingface.co/spaces/wing-nus/SciAssist/resolve/main/bart-large-cnn-e5.pt",
            "data_utils": DataUtilsForSeq2Seq,
        }
    }

}

def load_model(config: Dict, cache_dir=BASE_CACHE_DIR ):
    '''

    Args:
        config (Dict):
            A dictionary including the model class and the url to download the weights dict.
            For example:
                {
                    "model": BertForTokenClassifier,
                    "model_dict_url": "https://huggingface.co/spaces/wing-nus/SciAssist/resolve/main/scibert-uncased.pt",
                }

    Returns: A loaded model.

    '''

    print("Loading the model...")
    model_class = config["model"]
    model = model_class(cache_dir=cache_dir)
    state_dict = torch.hub.load_state_dict_from_url(config["model_dict_url"], model_dir=cache_dir)
    model.load_state_dict(state_dict)
    model.eval()
    print("Completed.")
    return model