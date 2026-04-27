# register model

import json
import mlflow
import os
import warnings
from mlflow.tracking import MlflowClient
from src.logger import logging

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")

if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "gaur3786"
    repo_name = "CAPSTONE-1"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(
        f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
    )
# # -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
# Uncomment for local use if needed
# mlflow.set_tracking_uri(
#    "https://dagshub.com/gaur3786/CAPSTONE-1.mlflow"
#  )

# dagshub.init(
#     repo_owner="gaur3786",
#     repo_name="CAPSTONE-1",
#     mlflow=True
#  )
# -------------------------------------------------------------------------------------



def load_model_info(file_path):

    try:
        with open(
            file_path,
            "r"
        ) as file:

            model_info = json.load(
                file
            )

        logging.info(
            "Loaded model info from %s",
            file_path
        )

        return model_info

    except Exception as e:

        logging.error(
            "Error loading model info: %s",
            e
        )

        raise



def register_model(
    model_name,
    model_info
):

    try:

        model_uri = (
            f"runs:/{model_info['run_id']}/"
            f"{model_info['model_path']}"
        )

        client = MlflowClient()


        try:
            client.create_registered_model(
                model_name
            )

            print(
                f"Created model {model_name}"
            )

        except:
            print(
                "Model already exists"
            )


        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )


        print(
            f"Registered version: "
            f"{model_version.version}"
        )


        try:

            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )

            print(
                "Moved model to Staging"
            )

        except Exception as e:

            logging.warning(
                "Stage transition skipped: %s",
                e
            )


    except Exception as e:

        logging.error(
            "Registration error: %s",
            e
        )

        raise



def main():

    try:

        model_info = load_model_info(
            "reports/experiment_info.json"
        )


        register_model(
            "my_model",
            model_info
        )


    except Exception as e:

        logging.error(
            "Failed model registration: %s",
            e
        )

        print(
            f"Error: {e}"
        )


if __name__=="__main__":
    main()