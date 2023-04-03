# main developer: Yixi Ding <dingyixi@hotmail.com>

from typing import Dict

import torch

from SciAssist import BASE_CACHE_DIR
from SciAssist.models.components.bert_token_classifier import BertForTokenClassifier
from SciAssist.models.components.flant5_summarization import FlanT5ForSummarization
from SciAssist.utils.data_utils import (
    DataUtilsForTokenClassification,
    DataUtilsForSeq2Seq, DataUtilsForFlanT5, DataUtilsForT5,
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
            "model": FlanT5ForSummarization,
            "model_dict_url": "https://huggingface.co/spaces/dyxohjl666/Controlled-summarization/resolve/main/flant5-mup-scisumm.pt",
            "data_utils": DataUtilsForSeq2Seq,
        },
        "default": {
            "model": FlanT5ForSummarization,
            "model_dict_url": "https://huggingface.co/spaces/dyxohjl666/Controlled-summarization/resolve/main/flant5-mup-scisumm.pt",
            "data_utils": DataUtilsForSeq2Seq,
        },
        "t5": {
            "model": FlanT5ForSummarization,
            "model_dict_url": None,
            "data_utils": DataUtilsForT5,
        },
        "flan-t5": {
            "model": FlanT5ForSummarization,
            "model_dict_url": None,
            "data_utils": DataUtilsForFlanT5,
        }
    },

    "controlled-summarization": {
        "default": {
            "model": FlanT5ForSummarization,
            "model_dict_url": "https://huggingface.co/spaces/dyxohjl666/Controlled-summarization/resolve/main/flant5-mup-scisumm.pt",
            # "model_dict_url": None,
            "data_utils": DataUtilsForFlanT5,
        }
    }

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