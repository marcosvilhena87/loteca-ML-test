import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class PreprocessStats:
    total_rows: int
    valid_rows: int
    invalid_missing_odds: int
    invalid_low_odds: int
    invalid_target: int


def _parse_decimal(value: str) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if "," in text and "." in text:
        text = text.replace(".", "")
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _detect_result_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    candidates = [("[1]", "[x]", "[2]"), ("1", "X", "2")]
    for cols in candidates:
        if all(col in df.columns for col in cols):
            return cols
    raise ValueError("Resultado columns not found in dataset")


def load_dataset(path: str, upcoming: bool = False) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")


def preprocess_dataset(path: str, upcoming: bool = False) -> Tuple[pd.DataFrame, List[str], PreprocessStats]:
    df = load_dataset(path, upcoming=upcoming)
    total_rows = len(df)
    odds_cols = ["Odds_1", "Odds_X", "Odds_2"]

    for col in odds_cols:
        if col not in df.columns:
            raise ValueError(f"Missing odds column: {col}")
        df[col] = df[col].apply(_parse_decimal)

    invalid_missing_odds = int(df[odds_cols].isna().any(axis=1).sum())
    invalid_low_odds = int((df[odds_cols] <= 1.0).any(axis=1).sum())

    if not upcoming:
        col_1, col_x, col_2 = _detect_result_columns(df)
        for col in [col_1, col_x, col_2]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        target_sum = df[[col_1, col_x, col_2]].sum(axis=1)
        invalid_target = int((target_sum != 1).sum())
    else:
        invalid_target = 0

    valid_mask = (~df[odds_cols].isna().any(axis=1)) & (~(df[odds_cols] <= 1.0).any(axis=1))
    if not upcoming:
        valid_mask &= target_sum == 1

    df = df.loc[valid_mask].copy()

    feature_df = add_features(df)
    feature_cols = [
        "log_odds_1",
        "log_odds_x",
        "log_odds_2",
        "spread_12",
        "spread_1x",
        "spread_x2",
        "entropy",
        "overround",
        "favorite_gap",
    ]

    stats = PreprocessStats(
        total_rows=total_rows,
        valid_rows=len(df),
        invalid_missing_odds=invalid_missing_odds,
        invalid_low_odds=invalid_low_odds,
        invalid_target=invalid_target,
    )

    return feature_df, feature_cols, stats


def _logit(p: float) -> float:
    eps = 1e-6
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    odds_cols = ["Odds_1", "Odds_X", "Odds_2"]
    p_raw = df[odds_cols].rdiv(1.0)
    overround = p_raw.sum(axis=1)
    p_norm = p_raw.div(overround, axis=0)

    df = df.copy()
    df["p1_market"] = p_norm["Odds_1"]
    df["px_market"] = p_norm["Odds_X"]
    df["p2_market"] = p_norm["Odds_2"]
    df["overround"] = overround
    df["p1"] = df["p1_market"]
    df["px"] = df["px_market"]
    df["p2"] = df["p2_market"]

    df["log_odds_1"] = df["p1_market"].apply(_logit)
    df["log_odds_x"] = df["px_market"].apply(_logit)
    df["log_odds_2"] = df["p2_market"].apply(_logit)

    df["spread_12"] = (df["p1_market"] - df["p2_market"]).abs()
    df["spread_1x"] = (df["p1_market"] - df["px_market"]).abs()
    df["spread_x2"] = (df["px_market"] - df["p2_market"]).abs()

    df["entropy"] = -(
        df["p1_market"] * (df["p1_market"].clip(1e-9)).apply(math.log)
        + df["px_market"] * (df["px_market"].clip(1e-9)).apply(math.log)
        + df["p2_market"] * (df["p2_market"].clip(1e-9)).apply(math.log)
    )

    df["favorite_gap"] = df[["p1_market", "px_market", "p2_market"]].apply(
        lambda row: row.sort_values(ascending=False).iloc[0]
        - row.sort_values(ascending=False).iloc[1],
        axis=1,
    )

    if "Concurso" in df.columns:
        df["Concurso"] = pd.to_numeric(df["Concurso"], errors="coerce")
    if "Jogo" in df.columns:
        df["Jogo"] = pd.to_numeric(df["Jogo"], errors="coerce")

    return df


def add_target_column(df: pd.DataFrame) -> pd.DataFrame:
    col_1, col_x, col_2 = _detect_result_columns(df)
    target = pd.Series(index=df.index, dtype=str)
    target[df[col_1] == 1] = "1"
    target[df[col_x] == 1] = "X"
    target[df[col_2] == 1] = "2"
    df = df.copy()
    df["target"] = target
    return df
