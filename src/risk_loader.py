import pandas as pd


def load_risk_table():

    df = pd.read_csv("data/processed/label_table.csv")

    # USE correct column
    df["risk_score"] = df["gt_change_ratio"] * 100

    df = df.sort_values(
        "risk_score",
        ascending=False
    )

    return df