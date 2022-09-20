import torch

from SciAssist import BASE_CACHE_DIR
from SciAssist.pipelines import TASKS, load_model


class Pipeline():

    def __init__(self, task_name: str, model_name: str = "default", device = "gpu", cache_dir = BASE_CACHE_DIR):

        self.config = TASKS[task_name][model_name]
        self.model_name = model_name
        self.model = load_model(config=self.config, cache_dir=cache_dir)

        self.device = device
        if device in ["cuda","gpu"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.cuda()
        else:
            self.device = torch.device("cpu")

        self.data_utils = self.config["data_utils"]