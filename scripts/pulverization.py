import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _clean_rateio_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df


def load_rateio(rateio_path: str) -> pd.DataFrame:
    """Load the payout history and normalize column types."""
    logging.info("Carregando dados de rateio em %s", rateio_path)
    rateio_df = pd.read_csv(rateio_path, delimiter=";", decimal=".", encoding="utf-8-sig")
    rateio_df = _clean_rateio_columns(rateio_df)

    rateio_df["Concurso"] = pd.to_numeric(rateio_df["Concurso"], errors="coerce").astype("Int64")
    rateio_df["Ganhadores 14 Acertos"] = (
        pd.to_numeric(rateio_df["Ganhadores 14 Acertos"], errors="coerce").fillna(0).astype(int)
    )
    rateio_df["Rateio 14 Acertos"] = pd.to_numeric(rateio_df["Rateio 14 Acertos"], errors="coerce").fillna(0.0)

    rateio_df["Acumulado 14 Acertos"] = (
        rateio_df["Acumulado 14 Acertos"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"sim": True, "não": False, "nao": False})
        .fillna(False)
    )

    logging.info("Rateio carregado com %s concursos", len(rateio_df))
    return rateio_df


def calculate_concurso_features(match_df: pd.DataFrame) -> pd.Series:
    """Aggregate bookmaker probabilities into contest-level features."""
    probabilities = match_df[["P(1)", "P(X)", "P(2)"]].to_numpy(dtype=float)
    p_max = probabilities.max(axis=1)

    sorted_probs = np.sort(probabilities, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2]
    gap = top1 - top2
    gap_std = np.std(gap)

    epsilon = 1e-12
    adjusted = np.clip(probabilities, epsilon, 1.0)
    entropy = -np.sum(adjusted * np.log(adjusted), axis=1)

    draw_probs = match_df["P(X)"].to_numpy(dtype=float)

    features = {
        "mean_pmax": np.mean(p_max),
        "median_pmax": float(np.median(p_max)),
        "count_pmax_ge_0.70": int(np.sum(p_max >= 0.70)),
        "count_pmax_ge_0.60": int(np.sum(p_max >= 0.60)),
        "mean_gap": np.mean(gap),
        "min_gap": float(np.min(gap)),
        "std_gap": gap_std,
        "mean_entropy": np.mean(entropy),
        "sum_entropy": np.sum(entropy),
        "count_gap_le_0.12": int(np.sum(gap <= 0.12)),
        "count_draw_high": int(np.sum(draw_probs >= 0.33)),
    }

    return pd.Series(features)


def build_concurso_dataset(matches_df: pd.DataFrame, rateio_df: pd.DataFrame) -> pd.DataFrame:
    """Combine match-level probabilities with payout labels per contest."""
    required_cols = {"Concurso", "P(1)", "P(X)", "P(2)"}
    missing_cols = required_cols - set(matches_df.columns)
    if missing_cols:
        raise KeyError(f"Colunas ausentes para agregação de concurso: {sorted(missing_cols)}")

    logging.info("Agregando features por concurso...")
    grouped = (
        matches_df.groupby("Concurso", group_keys=False)[["P(1)", "P(X)", "P(2)"]]
        .apply(calculate_concurso_features)
        .reset_index()
    )

    logging.info("Fazendo merge com dados de rateio...")
    merged = grouped.merge(rateio_df, on="Concurso", how="inner")
    merged = merged.dropna(subset=["Ganhadores 14 Acertos", "Rateio 14 Acertos"])
    logging.info("Dataset de concurso criado com %s linhas", len(merged))
    return merged


def _fit_and_store(model, features: pd.DataFrame, target: pd.Series, feature_names: List[str], path: str) -> None:
    model.fit(features, target)
    payload: Dict[str, object] = {"model": model, "feature_names": feature_names}
    dump(payload, path)
    logging.info("Modelo salvo em %s", path)


def train_pulverization_models(processed_matches_path: str, rateio_path: str, model_dir: str) -> None:
    """Train regressors to predict contest pulverization and payout."""
    os.makedirs(model_dir, exist_ok=True)

    matches_df = pd.read_csv(processed_matches_path, delimiter=";", decimal=".")
    rateio_df = load_rateio(rateio_path)

    concurso_df = build_concurso_dataset(matches_df, rateio_df)
    feature_cols = [
        "mean_pmax",
        "median_pmax",
        "count_pmax_ge_0.70",
        "count_pmax_ge_0.60",
        "mean_gap",
        "min_gap",
        "std_gap",
        "mean_entropy",
        "sum_entropy",
        "count_gap_le_0.12",
        "count_draw_high",
    ]

    X = concurso_df[feature_cols]

    ganhadores_target = np.log1p(concurso_df["Ganhadores 14 Acertos"])
    ganhadores_model = Ridge(random_state=42)
    _fit_and_store(
        ganhadores_model,
        X,
        ganhadores_target,
        feature_cols,
        os.path.join(model_dir, "pulverization_ganhadores.joblib"),
    )

    rateio_target = np.log(concurso_df["Rateio 14 Acertos"] + 1)
    rateio_model = GradientBoostingRegressor(random_state=42)
    _fit_and_store(
        rateio_model,
        X,
        rateio_target,
        feature_cols,
        os.path.join(model_dir, "pulverization_rateio.joblib"),
    )

    logging.info("Modelos de pulverização treinados com sucesso.")
