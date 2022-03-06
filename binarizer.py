import dotenv
import hydra
from src.datamodules.components.sr_bianarizer import DapsSRBinarizer
from omegaconf import DictConfig

dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="binarizer.yaml")
def main(config: DictConfig):
    binarizer: DapsSRBinarizer = hydra.utils.instantiate(config.binarizer)
    binarizer.process()


if __name__ == "__main__":
    main()
