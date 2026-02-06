import math
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def predict_probabilities(df: pd.DataFrame, model, feature_cols: List[str]) -> pd.DataFrame:
    probs = model.predict_proba(df[feature_cols].to_numpy())
    df = df.copy()
    df["p1_model"] = probs[:, 0]
    df["px_model"] = probs[:, 1]
    df["p2_model"] = probs[:, 2]
    return df


def _entropy_row(p1: float, px: float, p2: float) -> float:
    eps = 1e-9
    return -(
        p1 * math.log(max(p1, eps))
        + px * math.log(max(px, eps))
        + p2 * math.log(max(p2, eps))
    )


def score_duplos(
    df: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
) -> pd.DataFrame:
    df = df.copy()
    probs = df[["p1", "px", "p2"]].to_numpy()
    order = np.argsort(-probs, axis=1)
    top1_idx = order[:, 0]
    top2_idx = order[:, 1]
    labels = np.array(["1", "X", "2"])

    df["top1"] = labels[top1_idx]
    df["top2"] = labels[top2_idx]
    df["p_top1"] = probs[np.arange(len(df)), top1_idx]
    df["p_top2"] = probs[np.arange(len(df)), top2_idx]
    df["margem"] = df["p_top1"] - df["p_top2"]
    df["entropy_pred"] = [
        _entropy_row(p1, px, p2) for p1, px, p2 in probs
    ]

    overround_penalty = (df["overround"] - 1).clip(lower=0)
    df["score_duplo"] = (
        df["p_top2"]
        * (1 + alpha * df["entropy_pred"])
        * (1 - beta * overround_penalty)
        * (1 + gamma * (1 - df["margem"]))
    )

    return df


def _format_double(top1: str, top2: str) -> str:
    ordered = sorted([top1, top2], key=lambda x: {"1": 0, "X": 1, "2": 2}[x])
    return "".join(ordered)


def select_ticket(
    df: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
    max_favorite: float = 0.75,
    diversity_entropy_min: float = 0.05,
    diversity_margin_min: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = score_duplos(df, alpha=alpha, beta=beta, gamma=gamma)
    logger = logging.getLogger("loteca")

    candidates = df[df["p_top1"] < max_favorite].copy()
    if len(candidates) < 2:
        candidates = df.copy()

    candidates = candidates.sort_values(
        ["score_duplo", "entropy_pred", "p_top2", "margem", "Jogo"],
        ascending=[False, False, False, True, True],
    )
    top_duplos = []
    for _, row in candidates.head(6).iterrows():
        top_duplos.append(
            {
                "Jogo": int(row["Jogo"]),
                "top1": row["top1"],
                "top2": row["top2"],
                "p_top1": float(row["p_top1"]),
                "p_top2": float(row["p_top2"]),
                "margem": float(row["margem"]),
                "entropy_pred": float(row["entropy_pred"]),
                "overround": float(row["overround"]),
                "score_duplo": float(row["score_duplo"]),
            }
        )
    selected = []
    if not candidates.empty:
        selected.append(candidates.index[0])
        for idx in candidates.index[1:]:
            if len(selected) >= 2:
                break
            entropy_diff = abs(df.loc[idx, "entropy_pred"] - df.loc[selected[0], "entropy_pred"])
            margin_diff = abs(df.loc[idx, "margem"] - df.loc[selected[0], "margem"])
            if entropy_diff > diversity_entropy_min or margin_diff > diversity_margin_min:
                selected.append(idx)
            else:
                logger.info(
                    "Reject Jogo %s: entropy_diff=%.4f, margin_diff=%.4f (thresholds %.2f/%.2f)",
                    int(df.loc[idx, "Jogo"]),
                    entropy_diff,
                    margin_diff,
                    diversity_entropy_min,
                    diversity_margin_min,
                )
        if len(selected) < 2:
            for idx in candidates.index:
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= 2:
                    break

    df["tipo"] = "SECO"
    df.loc[selected, "tipo"] = "DUPLO"
    df["palpite"] = df.apply(
        lambda row: _format_double(row["top1"], row["top2"])
        if row["tipo"] == "DUPLO"
        else row["top1"],
        axis=1,
    )

    summary = {
        "duplos": int((df["tipo"] == "DUPLO").sum()),
        "secos": int((df["tipo"] == "SECO").sum()),
        "coverage_increment": float(df.loc[df["tipo"] == "DUPLO", "p_top2"].sum()),
        "entropy_duplos": float(df.loc[df["tipo"] == "DUPLO", "entropy_pred"].mean()),
        "entropy_total": float(df["entropy_pred"].mean()),
        "top_duplos": top_duplos,
    }

    return df, summary
