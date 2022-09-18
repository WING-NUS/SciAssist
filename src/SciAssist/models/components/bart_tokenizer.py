from transformers import AutoTokenizer

from SciAssist import BASE_CACHE_DIR

BART_MODEL_CHECKPOINT = "facebook/bart-large-cnn"
MODEL_MAX_LENGTH = 1024
MAX_SOURCE_LENGTH = 1024
MAX_TARGET_LENGTH = 128


bart_tokenizer = AutoTokenizer.from_pretrained(
    BART_MODEL_CHECKPOINT,
    model_max_length = MODEL_MAX_LENGTH,
    cache_dir=BASE_CACHE_DIR,
    use_fast=True
)
