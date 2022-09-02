from transformers import AutoTokenizer

BART_MODEL_CHECKPOINT = "facebook/bart-large-cnn"
MODEL_MAX_LENGTH = 1024
MAX_SOURCE_LENGTH = 1024
MAX_TARGET_LENGTH = 128
TOKENIZER_CACHE_DIR = ".cache"

bart_tokenizer = AutoTokenizer.from_pretrained(
    BART_MODEL_CHECKPOINT,
    model_max_length = MODEL_MAX_LENGTH,
    cache_dir=TOKENIZER_CACHE_DIR,
    use_fast=True
)
