import os
import pickle
import yaml
import logging
import lightgbm as lgb

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

    processed_path = os.path.join(root, "data/processed")

    with open(os.path.join(processed_path, "X_train.csv"), "rb") as f:
        X_train = pickle.load(f)

    with open(os.path.join(processed_path, "y_train.csv"), "rb") as f:
        y_train = pickle.load(f)

    model = lgb.LGBMClassifier(
        learning_rate=params["model_building"]["learning_rate"],
        max_depth=params["model_building"]["max_depth"],
        n_estimators=params["model_building"]["n_estimators"],
        objective="multiclass",
        num_class=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    with open(os.path.join(root, "models", "lgbm_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    logger.info("Model training completed successfully.")


if __name__ == "__main__":
    main()