# main developer: Yixi Ding <dingyixi@hotmail.com>

import torch
from typing import Dict

from SciAssist import BASE_CACHE_DIR
from SciAssist.models.components.bert_dataset_extraction import BertForDatasetExtraction
from SciAssist.models.components.bert_token_classifier import BertForTokenClassifier
from SciAssist.models.components.flant5_summarization import FlanT5ForSummarization
from SciAssist.utils.data_utils import (
    DataUtilsForTokenClassification,
    DataUtilsForFlanT5, DataUtilsForSeq2Seq,
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
        "default": {
            "model": FlanT5ForSummarization,
            "model_dict_url": "https://huggingface.co/spaces/dyxohjl666/Controlled-summarization/resolve/main/flant5-base-mup-scisumm-noctrl.pt",
            "data_utils": DataUtilsForSeq2Seq,
        },
        "flan-t5": {
            "model": FlanT5ForSummarization,
            "model_dict_url": "https://huggingface.co/spaces/dyxohjl666/Controlled-summarization/resolve/main/flant5-base-mup-scisumm-noctrl.pt",
            "data_utils": DataUtilsForFlanT5,
        }
    },

    "controlled-summarization": {
        "default": {
            "model": FlanT5ForSummarization,
            "model_dict_url": "https://huggingface.co/spaces/dyxohjl666/Controlled-summarization/resolve/main/flant5-base-mup-scisumm-repeat5-kws.pt",
            "data_utils": DataUtilsForFlanT5,
        },
    },
    "dataset-extraction": {
        "default": {
            "model": BertForDatasetExtraction,
            "model_dict_url": "https://huggingface.co/spaces/kirinzhu/SciAssist/resolve/main/dataset_extraction.pt",
            "data_utils": DataUtilsForDatasetExtraction,
        },
    },

}

def load_model(config: Dict, checkpoint=None, cache_dir=BASE_CACHE_DIR, device="gpu"):
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
    model = model_class(cache_dir=cache_dir,model_checkpoint=checkpoint)
    map_location=None
    if device == "cpu":
        map_location = torch.device("cpu")
    if config["model_dict_url"]!=None:
        map_location = config["map_location"] if "map_location" in config.keys() else map_location
        state_dict = torch.hub.load_state_dict_from_url(config["model_dict_url"], model_dir=cache_dir, map_location=map_location)
        model.load_state_dict(state_dict)
    else:
        # You can also choose to load your local trained model:
        # model.load_state_dict(torch.load("/home/linxiao/SciAssist-scibert-0223/src/models/scibert_ner/2023-03-25_02-27-17/scibert_dataset_extraction.pt"))
        pass

    model.eval()

    return model