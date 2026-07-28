# app.py
import io
import os
import re
import string
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Allows matplotlib to work without a display

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pickle
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

for resource in ["stopwords", "wordnet"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)


def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text


def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def normalize_text(text):
    """Run the full cleaning pipeline on a single comment."""
    if not isinstance(text, str):
        text = str(text or "")
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text


app = Flask(__name__)
CORS(app)

# app.py lives in flask-app/, .env lives in the project root one level up
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

# Pass DagsHub credentials through to MLflow if they're set in .env
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN") or os.getenv("MLFLOW_TRACKING_PASSWORD")
if DAGSHUB_TOKEN:
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "Aayush10671")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    print(f"DagsHub credentials found in .env (token length: {len(DAGSHUB_TOKEN)}) -- proceeding with authenticated MLflow access.")
else:
    print(
        "WARNING: No DAGSHUB_TOKEN / MLFLOW_TRACKING_PASSWORD found in .env -- "
        "downloading the model artifact from DagsHub will likely hang or fail with "
        "an auth error. Add your DagsHub token to .env as DAGSHUB_TOKEN."
    )

YOUTUBE_API_KEY = os.getenv("yt_API_KEY", "").strip().strip('"').strip("'")

# --- MLflow / DagsHub setup: model is loaded directly by run_id, not from the registry ---
mlflow.set_tracking_uri("https://dagshub.com/Aayush10671/yt-comment-sentiment-analysis.mlflow")

RUN_ID = "f5ac001b8acd47c8ae8d35654f3fc642"
MODEL_ID = "m-54c6a8ccaf6f44b1bf2140aea954d3f8"  # from the Model ID field in DagsHub's Models view

print(f"Loading model from models:/{MODEL_ID} ... this can take a moment.")
model = mlflow.pyfunc.load_model(f"models:/{MODEL_ID}")
print("Model loaded successfully.")
vectorizer = pickle.load(open(project_root / "models" / "tfidf_vectorizer.pkl", "rb"))


def filter_valid_comments(df, col="comment"):
    """
    Drop rows that would make the vectorizer/model throw:
    - missing (NaN) comments
    - comments that are empty after stripping whitespace
    Also ensures every remaining value is a plain string.
    """
    df = df.dropna(subset=[col])
    df = df[df[col].astype(str).str.strip() != ""]
    df[col] = df[col].astype(str)
    return df


def make_predictions(raw_comments):
    """Clean + vectorize + predict for a list of already-filtered comment strings."""
    cleaned = [normalize_text(c) for c in raw_comments]
    features_df = pd.DataFrame(
        vectorizer.transform(cleaned).toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    return [str(p) for p in model.predict(features_df)]


@app.route("/")
def home():
    return "Welcome to Sentiment Analysis API"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    comments = data.get("comments", [])

    if not isinstance(comments, list) or not comments:
        return jsonify({"error": "No comments provided"}), 400

    df = pd.DataFrame({"comment": comments})
    df = filter_valid_comments(df, "comment")

    if df.empty:
        return jsonify({"error": "No valid comments provided"}), 400

    predictions = make_predictions(df["comment"].tolist())

    response = [
        {"comment": comment, "sentiment": prediction}
        for comment, prediction in zip(df["comment"], predictions)
    ]
    return jsonify(response)


@app.route("/fetch_comments", methods=["POST"])
def fetch_comments():
    data = request.get_json(silent=True) or {}
    video_id = data.get("video_id", "")

    if not video_id:
        return jsonify({"error": "No video ID provided"}), 400
    if not YOUTUBE_API_KEY:
        return jsonify({"error": "YouTube API key is not configured"}), 400

    comments = []
    token = ""

    while len(comments) < 200:
        url = (
            "https://www.googleapis.com/youtube/v3/commentThreads"
            f"?part=snippet&videoId={video_id}&maxResults=100&pageToken={token}&key={YOUTUBE_API_KEY}"
        )
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return jsonify({"error": f"Failed to fetch comments: {exc}"}), 502

        items = data.get("items", [])
        if not items:
            break

        for item in items:
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            comments.append({
                "text": snippet.get("textOriginal", ""),
                "timestamp": snippet.get("publishedAt", "")
            })

        token = data.get("nextPageToken", "")
        if not token:
            break

    return jsonify(comments)


@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    data = request.get_json(silent=True) or {}
    comments_data = data.get("comments", [])

    if not isinstance(comments_data, list) or not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    df = pd.DataFrame(comments_data)  # expects "text" and "timestamp" keys
    df = df.rename(columns={"text": "comment"})
    df = filter_valid_comments(df, "comment")

    if df.empty:
        return jsonify({"error": "No valid comments provided"}), 400

    predictions = make_predictions(df["comment"].tolist())

    response = [
        {"comment": comment, "sentiment": prediction, "timestamp": timestamp}
        for comment, prediction, timestamp in zip(df["comment"], predictions, df["timestamp"])
    ]
    return jsonify(response)


@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    data = request.get_json()
    sentiment_counts = data.get("sentiment_counts")

    if not sentiment_counts:
        return jsonify({"error": "No data found"}), 400

    positive = int(sentiment_counts.get("1", 0))
    neutral = int(sentiment_counts.get("0", 0))
    negative = int(sentiment_counts.get("-1", 0))

    plt.figure(figsize=(6, 6))
    plt.pie(
        [positive, neutral, negative],
        labels=["Positive", "Neutral", "Negative"],
        colors=["green", "gray", "red"],
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Sentiment Distribution")

    img = io.BytesIO()
    plt.savefig(img, format="PNG")
    img.seek(0)
    plt.close()

    return send_file(img, mimetype="image/png")


@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    data = request.get_json()
    sentiment_data = data.get("sentiment_data")

    if not sentiment_data:
        return jsonify({"error": "No data found"}), 400

    df = pd.DataFrame(sentiment_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df["sentiment"] = df["sentiment"].astype(int)

    monthly_counts = df.resample("ME")["sentiment"].value_counts().unstack(fill_value=0)
    monthly_percentages = (monthly_counts.T / monthly_counts.sum(axis=1)).T * 100

    plt.figure(figsize=(10, 5))
    if 1 in monthly_percentages.columns:
        plt.plot(monthly_percentages.index, monthly_percentages[1],
                  color="green", marker="o", linewidth=2, label="Positive")
    if 0 in monthly_percentages.columns:
        plt.plot(monthly_percentages.index, monthly_percentages[0],
                  color="gray", marker="o", linewidth=2, label="Neutral")
    if -1 in monthly_percentages.columns:
        plt.plot(monthly_percentages.index, monthly_percentages[-1],
                  color="red", marker="o", linewidth=2, label="Negative")

    plt.title("Monthly Sentiment Trend")
    plt.xlabel("Month")
    plt.ylabel("Percentage (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="PNG")
    img.seek(0)
    plt.close()

    return send_file(img, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)