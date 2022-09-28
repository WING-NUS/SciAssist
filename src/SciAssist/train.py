import dotenv
import hydra
import pyrootutils
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

root = pyrootutils.find_root(search_from=__file__, indicator=["setup.py"])
root = root / "src"
root = pyrootutils.set_root(path=root, dotenv=True, pythonpath=True)

@hydra.main(version_base="1.2", config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):

    from SciAssist import utils
    from SciAssist.pipelines.training_pipeline import train

    # Applies optional utilities
    utils.extras(config)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
