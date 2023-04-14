import os
import platform

ROOT_DIR = os.getcwd()
BASE_OUTPUT_DIR = os.path.join(ROOT_DIR, "output/result")
BASE_TEMP_DIR = os.path.join(ROOT_DIR,"output/.temp")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

USER_DIR_MAP = {
    "linux": "HOME",
    "darwin": "HOME",
    "windows": "USERPROFILE"
}
USER_DIR = os.environ[USER_DIR_MAP[platform.system().lower()]]
# Directory for cached files, including checkpoints, model dicts, tokenizers involved in pipelines.
# Unlike `cache_dir` set in config files, this is for users.

BASE_CACHE_DIR = os.path.join(USER_DIR, ".cache/sciassist")


from SciAssist.pipelines.reference_string_parsing import ReferenceStringParsing
from SciAssist.pipelines.summarization import Summarization
from SciAssist.pipelines.dataset_extraction import DatasetExtraction