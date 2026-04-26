from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text):
    text = ''.join(
        [char for char in text if not char.isdigit()]
    )
    return text


def lower_case(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)


def removing_punctuations(text):
    text = re.sub(
        '[%s]' % re.escape(string.punctuation),
        ' ',
        text
    )

    text = text.replace('؛', "")

    text = re.sub(
        '\s+',
        ' ',
        text
    ).strip()

    return text


def removing_urls(text):
    url_pattern = re.compile(
        r'https?://\S+|www\.\S+'
    )
    return url_pattern.sub(r'', text)


def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text


# =========================================================
# LOCAL + PRODUCTION DAGSHUB SAFE CONFIG
# =========================================================

repo_owner = "gaur3786"
repo_name = "CAPSTONE-1"

dagshub_token = os.getenv("CAPSTONE_TEST")

# GitHub Actions / Production
if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri(
    f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
)

# Local only (skip in CI)
if not os.getenv("GITHUB_ACTIONS"):
    try:
        dagshub.init(
            repo_owner=repo_owner,
            repo_name=repo_name,
            mlflow=True
        )
        print("Local DagsHub initialized")
    except Exception:
        print("Skipping dagshub.init")

# =========================================================


app = Flask(__name__)


# ---------------- Prometheus Metrics ----------------

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total number of requests",
    ["method", "endpoint"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Latency of requests",
    ["endpoint"],
    registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Prediction counts",
    ["prediction"],
    registry=registry
)


# ---------------- Load Registered Model ----------------

model_name = "my_model"


def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    if not versions:
        raise Exception(
            "No registered model versions found"
        )

    latest_version = max(
        int(v.version) for v in versions
    )

    return latest_version


model_version = get_latest_model_version(
    model_name
)

model_uri = (
    f"models:/{model_name}/{model_version}"
)

print(
    f"Fetching model from {model_uri}"
)

model = mlflow.pyfunc.load_model(
    model_uri
)


# Load vectorizer
vectorizer = pickle.load(
    open(
        "models/vectorizer.pkl",
        "rb"
    )
)


# ---------------- Routes ----------------

@app.route("/")
def home():

    REQUEST_COUNT.labels(
        method="GET",
        endpoint="/"
    ).inc()

    start_time = time.time()

    response = render_template(
        "index.html",
        result=None
    )

    REQUEST_LATENCY.labels(
        endpoint="/"
    ).observe(
        time.time() - start_time
    )

    return response


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    REQUEST_COUNT.labels(
        method="POST",
        endpoint="/predict"
    ).inc()

    start_time = time.time()

    text = request.form["text"]

    text = normalize_text(text)

    features = vectorizer.transform(
        [text]
    )

    features_df = pd.DataFrame(
        features.toarray(),
        columns=[
            str(i)
            for i in range(
                features.shape[1]
            )
        ]
    )

    result = model.predict(
        features_df
    )

    prediction = result[0]

    PREDICTION_COUNT.labels(
        prediction=str(prediction)
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint="/predict"
    ).observe(
        time.time() - start_time
    )

    return render_template(
        "index.html",
        result=prediction
    )


@app.route(
    "/metrics",
    methods=["GET"]
)
def metrics():

    return (
        generate_latest(registry),
        200,
        {
            "Content-Type":
            CONTENT_TYPE_LATEST
        }
    )


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )