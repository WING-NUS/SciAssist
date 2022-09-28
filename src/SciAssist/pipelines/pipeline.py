# main developer: Yixi Ding <dingyixi@hotmail.com>

import torch

from SciAssist import BASE_OUTPUT_DIR, BASE_TEMP_DIR, BASE_CACHE_DIR
from SciAssist.pipelines import TASKS, load_model


class Pipeline():

    """

    Args:
        task_name (`str`):
            The task name, which is used to load model configs.
        model_name (`str`, *optional*):
            A string, the *model name* of a pretrained model provided for this task.
        device (`str`, *optional*):
            A string, `cpu` or `gpu`.
        cache_dir (`str` or `os.PathLike`, *optional*, default to "~/.cache/sciassist"):
            Path to a directory in which a downloaded pretrained model should be
            cached if the standard cache should not be used.
        output_dir (`str` or `os.PathLike`, *optional*, default to "output/result" from current work directory):
            Path to a directory in which the predicted results files should be stored.
        temp_dir (`str` or `os.PathLike`, *optional*, default to "output/.temp" from current work directory):
            Path to a directory which holds temporary files such as `.tei.xml`.

    """

    def __init__(self, task_name: str, model_name: str = "default", device="gpu",
                 cache_dir=None, output_dir=None, temp_dir=None):

        self.device = device
        self.cache_dir = cache_dir if cache_dir is not None else BASE_CACHE_DIR
        self.output_dir = output_dir if output_dir is not None else BASE_OUTPUT_DIR
        self.temp_dir = temp_dir if temp_dir is not None else BASE_TEMP_DIR

        self.config = TASKS[task_name][model_name]
        self.model_name = model_name
        self.model = load_model(config=self.config, cache_dir=self.cache_dir)
        if device in ["cuda", "gpu"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.cuda()
        else:
            self.device = torch.device("cpu")

        self.data_utils = self.config["data_utils"]
