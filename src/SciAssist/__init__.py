import os

# Directory for cached files, including checkpoints, model dicts, tokenizers involved in pipelines.
# Unlike `cache_dir` set in config files, this is for users.
BASE_CACHE_DIR = os.path.join(os.environ["HOME"],".cache/sciassist")