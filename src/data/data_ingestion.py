import os
import logging
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_params(params_path):
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    logger.info("Parameters loaded.")
    return params


def load_data(data_path):
    df = pd.read_csv(data_path)
    logger.info("Data loaded.")
    return df


def preprocess_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df = df[df["clean_comment"].str.strip() != ""]

    logger.info("Data preprocessing completed.")
    return df


def save_data(train_df, test_df, data_path):
    raw_path = os.path.join(data_path, "raw")
    os.makedirs(raw_path, exist_ok=True)

    train_df.to_csv(os.path.join(raw_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(raw_path, "test.csv"), index=False)

    logger.info("Train and test datasets saved.")


def main():

    params_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../params.yaml"
    )

    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../data"
    )

    params = load_params(params_path)

    df = load_data(
        "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
    )

    df = preprocess_data(df)

    train_df, test_df = train_test_split(
        df,
        test_size=params["data_ingestion"]["test_size"],
        random_state=42
    )

    save_data(train_df, test_df, data_path)

    logger.info("Data ingestion completed successfully.")


if __name__ == "__main__":
    main()