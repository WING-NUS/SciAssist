# main developer: Yixi Ding <dingyixi@hotmail.com>

from typing import Dict

import torch

from SciAssist import BASE_CACHE_DIR
from SciAssist.models.components.bart_summarization import BartForSummarization
from SciAssist.models.components.bert_token_classifier import BertForTokenClassifier
from SciAssist.models.components.flant5_summarization import FlanT5ForSummarization
from SciAssist.models.components.bert_dataset_extraction import BertForDatasetExtraction
from SciAssist.utils.data_utils import (
    DataUtilsForTokenClassification,
    DataUtilsForSeq2Seq,
    DataUtilsForDatasetExtraction
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

    "single-doc-summarization": {
        "bart-cnn-on-mup": {
            "model": BartForSummarization,
            "model_dict_url": "https://huggingface.co/spaces/wing-nus/SciAssist/resolve/main/bart-large-cnn-e5.pt",
            "data_utils": DataUtilsForSeq2Seq,
        },
        "default": {
            "model": BartForSummarization,
            "model_dict_url": "https://huggingface.co/spaces/wing-nus/SciAssist/resolve/main/bart-large-cnn-e5.pt",
            "data_utils": DataUtilsForSeq2Seq,
        },
        "flan-t5": {
            "model": FlanT5ForSummarization,
            "model_dict_url": None,
            "data_utils": DataUtilsForSeq2Seq,
        }
    },

    "dataset-extraction": {
        "default": {
            "model": BertForDatasetExtraction,
            "model_dict_url": None,
            "data_utils": DataUtilsForDatasetExtraction,
        },
    },

    # "controlled-summarization": {
    #     "default": {
    #         "model": FrostForSummarization,
    #         "model_dict_url": None,
    #         "data_utils": DataUtilsForFrost,
    #     }
    # }

}

def load_model(config: Dict, cache_dir=BASE_CACHE_DIR, device="gpu"):
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
    map_location = None if torch.cuda.is_available() and device in ["gpu","cuda"] else torch.device("cpu")
    if config["model_dict_url"]!=None:
        state_dict = torch.hub.load_state_dict_from_url(config["model_dict_url"], model_dir=cache_dir, map_location=map_location)
        model.load_state_dict(state_dict)
    model.eval()

    return model