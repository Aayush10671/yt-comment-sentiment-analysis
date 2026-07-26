import os
import pickle
import logging
import yaml
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_params(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def get_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )


def main():

    root = get_root()

    params = load_params(os.path.join(root, "params.yaml"))

    df = pd.read_csv(
        os.path.join(root, "data/interim/train_processed.csv")
    )

    # Remove any remaining NaN values
    df = df.dropna(subset=["clean_comment"])

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_comment"],
        df["category"],
        test_size=params["data_ingestion"]["test_size"],
        random_state=42,
        stratify=df["category"]
    )

    vectorizer = TfidfVectorizer(
        max_features=params["feature_engineering"]["max_features"],
        ngram_range=tuple(params["feature_engineering"]["ngram_range"])
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # SMOTE only on training data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Create folders
    os.makedirs(os.path.join(root, "artifacts"), exist_ok=True)
    os.makedirs(os.path.join(root, "data/processed"), exist_ok=True)

    # Save vectorizer
    with open(os.path.join(root, "models", "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    engineered_path = os.path.join(root, "data", "engineered")
    os.makedirs(engineered_path, exist_ok=True)

    # Convert sparse matrices to DataFrames
    X_train_df = pd.DataFrame.sparse.from_spmatrix(X_train)
    X_test_df = pd.DataFrame.sparse.from_spmatrix(X_test)

    y_train_df = pd.DataFrame({"category": y_train})
    y_test_df = pd.DataFrame({"category": y_test})

    # Save engineered features
    X_train_df.to_csv(
        os.path.join(engineered_path, "X_train.csv"),
        index=False
    )

    X_test_df.to_csv(
        os.path.join(engineered_path, "X_test.csv"),
        index=False
    )

    y_train_df.to_csv(
        os.path.join(engineered_path, "y_train.csv"),
        index=False
    )

    y_test_df.to_csv(
        os.path.join(engineered_path, "y_test.csv"),
        index=False
    )

    logger.info("Engineered features saved successfully.")


if __name__ == "__main__":
    main()