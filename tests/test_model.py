import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # ---------------- Production / GitHub Actions ----------------
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST secret missing")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "gaur3786"
        repo_name = "CAPSTONE-1"

        mlflow.set_tracking_uri(
            f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
        )

        # -------------------------------------------------------------

        cls.new_model_name = "my_model"

        # load latest model directly (avoid stage issue)
        cls.new_model_uri = f"models:/{cls.new_model_name}/latest"

        cls.new_model = mlflow.pyfunc.load_model(
            cls.new_model_uri
        )

        cls.vectorizer = pickle.load(
            open("models/vectorizer.pkl","rb")
        )

        cls.holdout_data = pd.read_csv(
            "data/processed/test_bow.csv"
        )


    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)


    def test_model_signature(self):

        text = "hi how are you"

        input_data = self.vectorizer.transform([text])

        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[
                str(i)
                for i in range(input_data.shape[1])
            ]
        )

        pred = self.new_model.predict(input_df)

        self.assertEqual(
            input_df.shape[1],
            len(self.vectorizer.get_feature_names_out())
        )

        self.assertEqual(
            len(pred),
            input_df.shape[0]
        )


    def test_model_performance(self):

        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred = self.new_model.predict(X_holdout)

        accuracy = accuracy_score(
            y_holdout, y_pred
        )
        precision = precision_score(
            y_holdout, y_pred
        )
        recall = recall_score(
            y_holdout, y_pred
        )
        f1 = f1_score(
            y_holdout, y_pred
        )

        self.assertGreaterEqual(accuracy,0.40)
        self.assertGreaterEqual(precision,0.40)
        self.assertGreaterEqual(recall,0.40)
        self.assertGreaterEqual(f1,0.40)


if __name__ == "__main__":
    unittest.main()