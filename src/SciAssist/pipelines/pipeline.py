# main developer: Yixi Ding <dingyixi@hotmail.com>

import torch

from SciAssist import BASE_OUTPUT_DIR, BASE_TEMP_DIR, BASE_CACHE_DIR
from SciAssist.pipelines import TASKS, load_model


class Pipeline():

    def __init__(self, task_name: str, model_name: str = "default", device="gpu",
                 cache_dir=BASE_CACHE_DIR, output_dir=BASE_OUTPUT_DIR, temp_dir=BASE_TEMP_DIR):

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
