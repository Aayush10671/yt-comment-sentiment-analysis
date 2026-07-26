import os
import pickle
import logging
import yaml
import pandas as pd
import lightgbm as lgb

from sklearn.feature_extraction.text import TfidfVectorizer

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

    train_data = pd.read_csv(
        os.path.join(root, "data", "interim", "train_processed.csv")
    )

    # Remove any NaN values
    train_data = train_data.dropna(subset=["clean_comment"])

    max_features = params["feature_engineering"]["max_features"]
    ngram_range = tuple(params["feature_engineering"]["ngram_range"])

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )

    X_train = vectorizer.fit_transform(
        train_data["clean_comment"]
    )

    y_train = train_data["category"]

    model = lgb.LGBMClassifier(
        learning_rate=params["model_building"]["learning_rate"],
        max_depth=params["model_building"]["max_depth"],
        n_estimators=params["model_building"]["n_estimators"],
        objective="multiclass",
        num_class=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    models_path = os.path.join(root, "models")
    os.makedirs(models_path, exist_ok=True)

    with open(os.path.join(models_path, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    with open(os.path.join(models_path, "lgbm_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    logger.info("Model and vectorizer saved successfully.")


if __name__ == "__main__":
    main()