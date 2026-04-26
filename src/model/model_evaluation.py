import numpy as np
import pandas as pd
import pickle
import json
import os
import tempfile

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

import mlflow
import mlflow.sklearn
import dagshub
from src.logger import logging


# Below code block is for production use
# -------------------------------------------------------------------------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError(
         "CAPSTONE_TEST environment variable is not set"
     )
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "gaur3786"
repo_name = "CAPSTONE-1"

mlflow.set_tracking_uri(
   f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
 )
# -------------------------------------------------------------------------------------


# local use
# -------------------------------------------------------------------------------------
#mlflow.set_tracking_uri(
#    "https://dagshub.com/gaur3786/CAPSTONE-1.mlflow"
#)

#dagshub.init(
#    repo_owner="gaur3786",
#    repo_name="CAPSTONE-1",
#    mlflow=True,
  
#)
# -------------------------------------------------------------------------------------


def load_model(file_path):
    try:
        with open(
            file_path,
            "rb"
        ) as f:
            model = pickle.load(f)

        logging.info(
            "Model loaded from %s",
            file_path
        )

        return model

    except Exception as e:
        logging.error(
            "Model loading failed: %s",
            e
        )
        raise


def load_data(file_path):
    try:
        df = pd.read_csv(
            file_path
        )

        logging.info(
            "Data loaded from %s",
            file_path
        )

        return df

    except Exception as e:
        logging.error(
            "Data loading failed: %s",
            e
        )
        raise


def evaluate_model(
    clf,
    X_test,
    y_test
):

    y_pred = clf.predict(
        X_test
    )

    y_prob = clf.predict_proba(
        X_test
    )[:,1]


    metrics = {

        "accuracy":
        accuracy_score(
            y_test,
            y_pred
        ),

        "precision":
        precision_score(
            y_test,
            y_pred
        ),

        "recall":
        recall_score(
            y_test,
            y_pred
        ),

        "auc":
        roc_auc_score(
            y_test,
            y_prob
        )
    }

    return metrics


def save_metrics(
    metrics,
    file_path
):

    with open(
        file_path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


def save_model_info(
    run_id,
    model_path,
    file_path
):

    info = {
        "run_id":run_id,
        "model_path":model_path
    }

    with open(
        file_path,
        "w"
    ) as f:

        json.dump(
            info,
            f,
            indent=4
        )


def main():

    mlflow.set_experiment(
        "my-dvc-pipeline"
    )


    with mlflow.start_run() as run:

        try:

            clf = load_model(
                "./models/model.pkl"
            )


            test_data = load_data(
                "./data/processed/test_bow.csv"
            )


            X_test = test_data.iloc[
                :,:-1
            ].values

            y_test = test_data.iloc[
                :,-1
            ].values


            metrics = evaluate_model(
                clf,
                X_test,
                y_test
            )


            save_metrics(
                metrics,
                "reports/metrics.json"
            )


            for k,v in metrics.items():

                mlflow.log_metric(
                    k,
                    v
                )


            if hasattr(
                clf,
                "get_params"
            ):

                for p,v in clf.get_params().items():

                    mlflow.log_param(
                        p,
                        v
                    )


            # -----------------------------------------
            # TEMP DIRECTORY FIX
            with tempfile.TemporaryDirectory() as tmpdir:

                model_dir = os.path.join(
                    tmpdir,
                    "mlflow_model"
                )

                mlflow.sklearn.save_model(
                    sk_model=clf,
                    path=model_dir
                )

                mlflow.log_artifacts(
                    model_dir,
                    artifact_path="model"
                )
            # -----------------------------------------


            save_model_info(
                run.info.run_id,
                "model",
                "reports/experiment_info.json"
            )


            mlflow.log_artifact(
                "reports/metrics.json"
            )


            print(
                "Model evaluation completed"
            )


        except Exception as e:

            logging.error(
                "Failed model evaluation: %s",
                e
            )

            print(e)



if __name__=="__main__":
    main()