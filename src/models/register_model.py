import os
import json
import yaml
import logging
import mlflow


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
        mlflow.set_tracking_uri(
            params["mlflow"]["tracking_uri"]
        )

        with open(os.path.join(root, "experiment_info.json"), "r") as f:
            model_info = json.load(f)

        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        model_name = "yt_chrome_plugin_model"

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        client = mlflow.tracking.MlflowClient()

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.info(
            f"Model version {model_version.version} registered successfully."
        )

    except Exception as e:
        logger.warning(f"MLflow model registration failed: {e}. Skipping MLflow registration.")
        
        with open(os.path.join(root, "experiment_info.json"), "r") as f:
            model_info = json.load(f)
        
        logger.info("Model information saved locally. MLflow registration skipped due to connection error.")


if __name__ == "__main__":
    main()