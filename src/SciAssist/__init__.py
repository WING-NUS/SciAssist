import os

ROOT_DIR = os.getcwd()
BASE_OUTPUT_DIR = os.path.join(ROOT_DIR, "output/result")
BASE_TEMP_DIR = os.path.join(ROOT_DIR,"output/.temp")
# Directory for cached files, including checkpoints, model dicts, tokenizers involved in pipelines.
# Unlike `cache_dir` set in config files, this is for users.
BASE_CACHE_DIR = os.path.join(os.environ["HOME"],".cache/sciassist")

from SciAssist.pipelines.reference_string_parsing import ReferenceStringParsing
from SciAssist.pipelines.summarization import Summarization