# src/data/data_preprocessing.py
import os
import re
import logging
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Download required resources
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize once
stop_words = set(stopwords.words("english")) - {
    "not", "no", "but", "however", "yet"
}

lemmatizer = WordNetLemmatizer()

def preprocess_comment(comment):

    comment = comment.lower()
    comment = comment.strip()

    comment = re.sub(r"\n", " ", comment)
    comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

    comment = " ".join(
        word for word in comment.split()
        if word not in stop_words
    )

    comment = " ".join(
        lemmatizer.lemmatize(word)
        for word in comment.split()
    )

    return comment


def normalize_text(df):

    # Remove missing values
    df = df.dropna(subset=["clean_comment"])

    # Remove empty comments
    df = df[df["clean_comment"].astype(str).str.strip() != ""]

    # Ensure string datatype
    df["clean_comment"] = df["clean_comment"].astype(str)

    # Apply preprocessing
    df["clean_comment"] = df["clean_comment"].apply(preprocess_comment)

    logger.info("Text preprocessing completed.")

    return df


def save_data(train_df, test_df, data_path):

    interim_path = os.path.join(data_path, "interim")
    os.makedirs(interim_path, exist_ok=True)

    train_df.to_csv(
        os.path.join(interim_path, "train_processed.csv"),
        index=False
    )

    test_df.to_csv(
        os.path.join(interim_path, "test_processed.csv"),
        index=False
    )

    logger.info("Processed datasets saved.")


def main():

    logger.info("Loading datasets...")

    train_df = pd.read_csv("./data/raw/train.csv")
    test_df = pd.read_csv("./data/raw/test.csv")

    train_df = normalize_text(train_df)
    test_df = normalize_text(test_df)

    save_data(
        train_df,
        test_df,
        "./data"
    )

    logger.info("Data preprocessing completed successfully.")


if __name__ == "__main__":
    main()