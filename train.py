import dotenv
import hydra
from omegaconf import DictConfig
import pyrootutils
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

@hydra.main(version_base="1.2", config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):

    from src import utils
    from src.training_pipeline import train

    # Applies optional utilities
    utils.extras(config)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
