from transformers import AutoTokenizer

BERT_MODEL_CHECKPOINT = "allenai/scibert_scivocab_uncased"
MODEL_MAX_LENGTH = 512
TOKENIZER_CACHE_DIR = ".cache"

bert_tokenizer = AutoTokenizer.from_pretrained(
    BERT_MODEL_CHECKPOINT,
    model_max_length=MODEL_MAX_LENGTH,
    cache_dir=TOKENIZER_CACHE_DIR
)
