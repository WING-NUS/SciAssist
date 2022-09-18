from transformers import AutoTokenizer

from SciAssist import BASE_CACHE_DIR

BERT_MODEL_CHECKPOINT = "allenai/scibert_scivocab_uncased"
MODEL_MAX_LENGTH = 512

bert_tokenizer = AutoTokenizer.from_pretrained(
    BERT_MODEL_CHECKPOINT,
    model_max_length=MODEL_MAX_LENGTH,
    cache_dir=BASE_CACHE_DIR
)
