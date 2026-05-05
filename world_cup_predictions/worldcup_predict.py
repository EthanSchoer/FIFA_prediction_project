import argparse
import csv
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


SAMPLE_CSV = "results.csv"


def load_match_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")
    df = df.copy()
    # Drop rows with missing scores
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["draw"] = (df["home_score"] == df["away_score"]).astype(int)
    df["away_win"] = (df["away_score"] > df["home_score"]).astype(int)
    return df


def train_logistic_model(df: pd.DataFrame) -> Pipeline:
    df_model = df.copy()
    df_model["home_goal_diff"] = df_model["home_score"] - df_model["away_score"]
    df_model["result"] = df_model["home_win"]
    features = df_model[["home_team", "away_team"]]
    labels = df_model["result"]

    pipeline = Pipeline(
        steps=[
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore"),
            ),
            (
                "logistic",
                LogisticRegression(max_iter=500),
            ),
        ]
    )
    pipeline.fit(features, labels)
    return pipeline


def predict_match(model: Pipeline, home_team: str, away_team: str) -> float:
    proba = model.predict_proba([[home_team, away_team]])[0][1]
    return proba


def main() -> None:
    parser = argparse.ArgumentParser(description="World Cup match prediction CLI")
    parser.add_argument(
        "--matches",
        type=str,
        default=SAMPLE_CSV,
        help="Path to a matches CSV. Columns: date, home_team, away_team, home_score, away_score",
    )
    parser.add_argument(
        "--predict",
        nargs=2,
        metavar=("HOME_TEAM", "AWAY_TEAM"),
        help="Predict the probability that the home team wins the match",
    )
    parser.add_argument(
        "--split-data",
        action="store_true",
        help="Split the default dataset into train and test CSV files",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="train.csv",
        help="Path for the training data CSV",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="test.csv",
        help="Path for the test data CSV",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing (0.0 to 1.0)",
    )
    args = parser.parse_args()

    if args.split_data:
        df = load_match_data(SAMPLE_CSV)
        train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)
        train_df.to_csv(args.train_file, index=False)
        test_df.to_csv(args.test_file, index=False)
        print(f"Data split into {args.train_file} (train) and {args.test_file} (test)")
        return

    if not Path(args.matches).exists():
        raise FileNotFoundError(f"Match file not found: {args.matches}")

    df = load_match_data(args.matches)

    model = train_logistic_model(df)
    print("Logistic regression model trained on provided match data.")
    if args.predict:
        home_team, away_team = args.predict
        home_prob = predict_match(model, home_team, away_team)
        away_prob = 1 - home_prob
        print(f"Prediction for {home_team} vs {away_team}:")
        print(f"{home_team}: {home_prob:.2%}")
        print(f"{away_team}: {away_prob:.2%}")


if __name__ == "__main__":
    main()
