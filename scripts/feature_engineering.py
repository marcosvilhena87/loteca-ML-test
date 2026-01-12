import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "P(1)",
    "P(X)",
    "P(2)",
    "is_neutro",
    "last5_diff",
    "last5_sum",
    "pmax_market",
    "pspread_market",
    "entropy_market",
    "fav_1",
    "fav_X",
    "fav_2",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Home_Last5_Home" not in df.columns and "last-5-h2h-mandante" in df.columns:
        df["Home_Last5_Home"] = df["last-5-h2h-mandante"]
    if "Away_Last5_Away" not in df.columns and "last-5-h2h-visitante" in df.columns:
        df["Away_Last5_Away"] = df["last-5-h2h-visitante"]

    required_columns = ["P(1)", "P(X)", "P(2)", "Mando", "Home_Last5_Home", "Away_Last5_Away"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "As seguintes colunas são necessárias para gerar as features: "
            f"{missing_columns}"
        )

    df["is_neutro"] = (df["Mando"].str.lower() == "neutro").astype(int)
    df["last5_diff"] = df["Home_Last5_Home"] - df["Away_Last5_Away"]
    df["last5_sum"] = df["Home_Last5_Home"] + df["Away_Last5_Away"]

    df["pmax_market"] = df[["P(1)", "P(X)", "P(2)"]].max(axis=1)
    df["pspread_market"] = df["pmax_market"] - df[["P(1)", "P(X)", "P(2)"]].min(axis=1)

    epsilon = 1e-10
    probs = df[["P(1)", "P(X)", "P(2)"]].to_numpy()
    adjusted_probs = probs + epsilon
    df["entropy_market"] = -np.sum(adjusted_probs * np.log(adjusted_probs), axis=1)

    fav_labels = np.select(
        [
            df["P(1)"] >= df[["P(X)", "P(2)"]].max(axis=1),
            df["P(X)"] >= df[["P(1)", "P(2)"]].max(axis=1),
        ],
        ["1", "X"],
        default="2",
    )
    fav_dummies = pd.get_dummies(fav_labels, prefix="fav")
    for column in ["fav_1", "fav_X", "fav_2"]:
        if column not in fav_dummies.columns:
            fav_dummies[column] = 0
    df = pd.concat([df, fav_dummies[["fav_1", "fav_X", "fav_2"]]], axis=1)

    return df
