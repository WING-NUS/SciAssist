import dotenv
import hydra
from omegaconf import DictConfig
import pyrootutils
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
@hydra.main(version_base="1.2", config_path="configs/", config_name="test.yaml")
def main(config: DictConfig):

    from src import utils
    from src.testing_pipeline import test

    # Applies optional utilities
    utils.extras(config)

    # Evaluate model
    return test(config)


if __name__ == "__main__":
    main()
