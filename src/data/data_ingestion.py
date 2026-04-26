# data ingestion
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml
from src.logger import logging
from src.connections import s3_connection   # kept for later S3 use


def load_params(params_path: str) -> dict:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logging.debug("Parameters loaded")
        return params
    except Exception as e:
        logging.error("Error loading params: %s", e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            data_url,
            engine="python",
            on_bad_lines="skip"
        )
        logging.info("Data loaded from GitHub")
        return df
    except Exception as e:
        logging.error("CSV parsing failed: %s", e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Pre-processing...")

        final_df = df[
            df["sentiment"].isin(["positive","negative"])
        ].copy()

        final_df["sentiment"] = final_df["sentiment"].replace({
            "positive":1,
            "negative":0
        })

        logging.info("Preprocessing completed")
        return final_df

    except Exception as e:
        logging.error("Preprocess error: %s", e)
        raise


def save_data(train_data, test_data, data_path):
    try:
        raw_data_path = os.path.join(data_path,"raw")
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(
            os.path.join(raw_data_path,"train.csv"),
            index=False
        )

        test_data.to_csv(
            os.path.join(raw_data_path,"test.csv"),
            index=False
        )

        logging.info("Train and test saved")

    except Exception as e:
        logging.error("Save error: %s", e)
        raise


def main():
    try:
        params = load_params("params.yaml")
        test_size = params["data_ingestion"]["test_size"]
        #test_size = 0.2

        # -------- CURRENT SOURCE: GITHUB --------
        df = load_data(
            "https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv"
        )

        # -------- KEEP S3 CODE FOR LATER --------
        # s3 = s3_connection.s3_operations(
        #     "bucket-name",
        #     "accesskey",
        #     "secretkey"
        # )
        # df = s3.fetch_file_from_s3("data.csv")

        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=42
        )

        save_data(train_data, test_data, "./data")

        print("Data ingestion completed successfully")

    except Exception as e:
        logging.error(
            "Failed to complete data ingestion: %s", e
        )
        print(e)


if __name__ == "__main__":
    main()