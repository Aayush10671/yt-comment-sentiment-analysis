import os
import pickle
import logging
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

    try:
        mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
        mlflow.set_experiment(params["mlflow"]["experiment_name"])
    except Exception as e:
        logger.warning(f"MLflow setup failed: {e}. Continuing without MLflow logging.")

    interim_path = os.path.join(root, "data", "interim")
    models_path = os.path.join(root, "models")

    # Load test data
    test_data = pd.read_csv(
        os.path.join(interim_path, "test_processed.csv")
    )

    # Remove any NaN values
    test_data = test_data.dropna(subset=["clean_comment"])

    X_test_text = test_data["clean_comment"]
    y_test = test_data["category"]

    with open(os.path.join(models_path, "lgbm_model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(models_path, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    # Transform test data
    X_test = vectorizer.transform(X_test_text)

    try:
        with mlflow.start_run() as run:

            mlflow.set_tag("model", "LightGBM")
            mlflow.set_tag("vectorizer", "TF-IDF")

            mlflow.log_params(params["model_building"])

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True
            )

            for label, metrics in report.items():

                if isinstance(metrics, dict):

                    for metric_name, metric_value in metrics.items():

                        mlflow.log_metric(
                            f"{label}_{metric_name}",
                            metric_value
                        )

            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")

            plt.savefig("confusion_matrix.png")
            plt.close()

            mlflow.log_artifact("confusion_matrix.png")

            mlflow.sklearn.log_model(
                model,
                name="lgbm_model"
            )

            mlflow.log_artifact(
                os.path.join(models_path, "tfidf_vectorizer.pkl")
            )

            logger.info("Model evaluation completed successfully.")

            model_info = {
            "run_id": run.info.run_id,
            "model_path": "lgbm_model"
        }
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}. Saving model info without MLflow.")
        model_info = {
            "run_id": "local",
            "model_path": "lgbm_model"
        }

    with open(
        os.path.join(root, "experiment_info.json"),
        "w"
    ) as f:
        json.dump(model_info, f, indent=4)


if __name__ == "__main__":
    main()