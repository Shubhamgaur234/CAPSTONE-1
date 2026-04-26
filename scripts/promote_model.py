# promote model

import os
import mlflow


def promote_model():

    # ---------------- Production / GitHub Actions ----------------
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
    # ------------------------------------------------------------

    client = mlflow.MlflowClient()

    model_name = "my_model"

    # get latest registered version
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    if not versions:
        raise Exception("No model versions found")

    latest_version = max(
        [int(v.version) for v in versions]
    )

    # Promote using alias (modern replacement for stages)
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=latest_version
    )

    print(
        f"Model version {latest_version} promoted to Production (champion)"
    )


if __name__ == "__main__":
    promote_model()