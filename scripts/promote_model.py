import os
import mlflow


def promote_model():

    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError(
            "CAPSTONE_TEST environment variable is not set"
        )

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(
        "https://dagshub.com/gaur3786/CAPSTONE-1.mlflow"
    )

    client = mlflow.tracking.MlflowClient()

    model_name = "my_model"

    # latest registered version
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    if not versions:
        raise Exception("No versions found")

    latest_version = str(
        max(int(v.version) for v in versions)
    )

    # archive old production versions
    try:
        prod_versions = client.get_latest_versions(
            model_name,
            stages=["Production"]
        )

        for v in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=v.version,
                stage="Archived"
            )
    except:
        pass

    # promote latest to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production"
    )

    print(
        f"Model version {latest_version} promoted to Production"
    )


if __name__ == "__main__":
    promote_model()