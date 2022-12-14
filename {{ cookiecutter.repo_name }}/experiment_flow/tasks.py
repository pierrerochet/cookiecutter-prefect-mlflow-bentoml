from pathlib import Path
from typing import List, Union

import mlflow
import pandas as pd
from prefect import task
from prefect.tasks import task_input_hash
from sklearn.base import ClassifierMixin

from .utils import log


@task(cache_key_fn=task_input_hash)
def load_data(data_path: str) -> pd.DataFrame:

    # --- I. Put imports here ------------------
    from sklearn.datasets import load_wine

    # --- II. Add your custom code here --------
    df = load_wine(as_frame=True)["frame"]
    df.to_csv(Path(data_path) / "data.csv", index=False)

    return df


@task(cache_key_fn=task_input_hash)
def split_data(
    df: pd.DataFrame,
    data_path: str,
    test_size: Union[float, int, None] = 0.3,
    random_state: int = 42,
    stratify=False,
) -> List[pd.DataFrame]:

    # --- I. Put imports here ------------------
    from sklearn.model_selection import train_test_split

    # --- II. Add your custom code here --------
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"] if stratify else None,
    )

    df_train.to_csv(Path(data_path) / "train.csv", index=False)
    df_test.to_csv(Path(data_path) / "test.csv", index=False)

    return df_train, df_test


@task(cache_key_fn=task_input_hash)
def analyze_data(
    df: pd.DataFrame,
    report_path: str,
):
    # --- I. Put imports here ------------------
    from .chart.data import train_test_chart

    # --- II. Add your custom code here --------
    chart = train_test_chart(df)

    chart.save(Path(report_path) / "data.html")
    return


@task(cache_key_fn=task_input_hash)
def build_model(**params) -> ClassifierMixin:

    # --- I. Put imports here ------------------
    from sklearn.linear_model import LogisticRegression

    # --- II. Add your custom code here --------
    model = LogisticRegression(**params)

    return model


@task(cache_key_fn=task_input_hash)
@log.log_timeit
def train_model(
    model: ClassifierMixin,
    df_train: pd.DataFrame,
    path: str,
    signature,
) -> ClassifierMixin:

    # --- I. Put imports here ------------------
    import os
    import shutil

    # --- II. Add your custom code here --------
    X_train = df_train.drop("target", axis=1)
    y_train = df_train["target"]

    model.fit(X_train, y_train)

    if os.path.exists(path):
        shutil.rmtree(path)

    mlflow.sklearn.save_model(
        model,
        path,
        signature=signature,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )

    return model


@task(cache_key_fn=task_input_hash)
def test_model(
    model: ClassifierMixin,
    df_test: pd.DataFrame,
    processed_path: str,
) -> pd.DataFrame:

    # --- I. Put imports here ------------------
    # import ...

    # --- II. Add your custom code here --------
    X_test = df_test.drop("target", axis=1)
    y_pred = model.predict(X_test)

    df_test["pred"] = y_pred

    df_test.to_csv(Path(processed_path) / "test.csv", index=False)

    return df_test


@task(cache_key_fn=task_input_hash)
def eval_metrics(
    df_test: pd.DataFrame,
    report_path: str,
) -> dict:

    # --- I. Put imports here ------------------
    import json
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    from .chart import metrics

    # --- II. Add your custom code here --------
    y_true = df_test["target"]
    y_pred = df_test["pred"]

    # Save confusion matrix
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm = df_cm.stack()
    df_cm = df_cm.reset_index()
    df_cm.columns = ["true", "pred", "value"]

    chart = metrics.cm_chart(df_cm)
    chart.save(Path(report_path) / "confusion_matrix.html")

    # Save scores
    scores = {
        "f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
    }

    with open(Path(report_path) / "metrics.json", "w", encoding="utf8") as fp:
        json.dump(scores, fp)

    return scores
