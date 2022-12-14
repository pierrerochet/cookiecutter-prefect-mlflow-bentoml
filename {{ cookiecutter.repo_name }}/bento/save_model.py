import bentoml
import hydra
import mlflow
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="main")
def save_model(cfg: DictConfig):

    mlflow.set_tracking_uri(cfg.TRACKING_SERVER_URI)

    loaded_model = mlflow.sklearn.load_model(cfg.deployment.MODEL_URI)
    bentoml.sklearn.save_model("pmb-model", loaded_model)


if __name__ == "__main__":
    save_model()
