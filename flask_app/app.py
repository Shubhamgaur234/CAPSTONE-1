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
    text = text.replace(
        '؛',
        ""
    )
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
    return url_pattern.sub(
        r'',
        text
    )


def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text


# local use
# ----------------------------------------------------
mlflow.set_tracking_uri(
    'https://dagshub.com/gaur3786/CAPSTONE-1.mlflow'
)

dagshub.init(
    repo_owner='gaur3786',
    repo_name='CAPSTONE-1',
    mlflow=True
)
# ----------------------------------------------------


# Flask app
app = Flask(__name__)


# Prometheus metrics
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total requests",
    ["method","endpoint"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Latency",
    ["endpoint"],
    registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Prediction counts",
    ["prediction"],
    registry=registry
)


# ----------------------------------------------------
# FIXED MODEL VERSION LOADER
model_name = "my_model"


def get_latest_model_version(
    model_name
):
    client = mlflow.MlflowClient()

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    latest_version = max(
        int(v.version)
        for v in versions
    )

    return latest_version


model_version = get_latest_model_version(
    model_name
)

model_uri = (
    f"models:/{model_name}/"
    f"{model_version}"
)

print(
    f"Fetching model from: {model_uri}"
)

model = mlflow.pyfunc.load_model(
    model_uri
)

vectorizer = pickle.load(
    open(
        'models/vectorizer.pkl',
        'rb'
    )
)
# ----------------------------------------------------


@app.route("/")
def home():

    REQUEST_COUNT.labels(
        method="GET",
        endpoint="/"
    ).inc()

    start = time.time()

    response = render_template(
        "index.html",
        result=None
    )

    REQUEST_LATENCY.labels(
        endpoint="/"
    ).observe(
        time.time()-start
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

    start=time.time()

    text = request.form["text"]

    text = normalize_text(
        text
    )

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
        prediction=str(
            prediction
        )
    ).inc()


    REQUEST_LATENCY.labels(
        endpoint="/predict"
    ).observe(
        time.time()-start
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
        generate_latest(
            registry
        ),
        200,
        {
          "Content-Type":
          CONTENT_TYPE_LATEST
        }
    )


if __name__=="__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )