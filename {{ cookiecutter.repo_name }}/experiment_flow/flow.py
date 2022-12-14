import os

import hydra
import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from omegaconf import DictConfig, OmegaConf
from prefect import flow


from pathlib import Path
from typing import List, Union

import mlflow
import pandas as pd

from . import tasks


@flow(version=os.getenv("GIT_COMMIT_SHA"))
def experiment_flow(cfg: dict):

    # To facilitate manipulation of the configuration
    cfg = OmegaConf.create(cfg)

    mlflow.set_tracking_uri(cfg.TRACKING_SERVER_URI)

    with mlflow.start_run():
        # *** Manage your tasks below ***

        # --- Task 1: Load data -----------
        df = tasks.load_data(cfg.path.data.raw.path)
        mlflow.log_artifact(
            cfg.path.data.raw.path,
            cfg.path.data.dir,
        )

        # --- Task 2: Split data ----------
        df_train, df_test = tasks.split_data(df, cfg.path.data.splitted.path)
        mlflow.log_artifact(
            cfg.path.data.splitted.path,
            cfg.path.data.dir,
        )

        # --- Task 3: Analyze data -----------
        df_train_g = df_train.assign(group=["train"] * df_train.shape[0])
        df_test_g = df_test.assign(group=["test"] * df_test.shape[0])
        tasks.analyze_data(pd.concat([df_train_g, df_test_g]), cfg.path.reports.dir)
        mlflow.log_artifact(cfg.path.reports.dir)

        # --- Task 4: Build model ---------
        model = tasks.build_model(**cfg.training)
        params = model.get_params()
        mlflow.log_params(params)

        # --- Task 5: Train model ---------
        signature = infer_signature(df_train.drop("target", axis=1), df_train["target"])
        model_trained, timeit = tasks.train_model(
            model, df_train, cfg.path.model.dir, signature
        )

        mlflow.log_metric("training_time", timeit)
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
        )

        # --- Task 6: Test model ----------
        df_test = tasks.test_model(
            model_trained,
            df_test,
            cfg.path.data.processed.path,
        )
        mlflow.log_artifact(
            cfg.path.data.processed.path,
            cfg.path.data.dir,
        )

        # --- Task 7: Eval metrics --------
        scores = tasks.eval_metrics(df_test, cfg.path.reports.dir)
        mlflow.log_metrics(scores)
        mlflow.log_artifact(cfg.path.reports.dir)

    return scores


@hydra.main(version_base=None, config_path="../conf", config_name="main")
def main(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    experiment_flow(config_dict)


if __name__ == "__main__":
    main()
