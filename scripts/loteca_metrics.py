import logging
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CardMetrics:
    survival: Mapping[int, float]
    duplo_coverage: float
    expected_hits: float
    penalty: float


def _compute_entropy_and_gap(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    epsilon = 1e-12
    adjusted = probabilities + epsilon
    entropies = -np.sum(adjusted * np.log(adjusted), axis=1)
    sorted_probs = np.sort(adjusted, axis=1)[:, ::-1]
    gaps = sorted_probs[:, 0] - sorted_probs[:, 1]
    return entropies, gaps


def _select_duplo_indices(probabilities: np.ndarray, alpha: float, duplo_count: int) -> np.ndarray:
    entropies, gaps = _compute_entropy_and_gap(probabilities)
    scores = entropies + alpha * (1 - gaps)
    count = min(duplo_count, len(scores))
    return np.argsort(scores)[::-1][:count]


def _build_card_frame(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    class_order: Sequence[str],
    alpha: float,
    duplo_count: int,
) -> pd.DataFrame:
    classes = list(class_order)
    top_two_indices = np.argsort(probabilities, axis=1)[:, ::-1]
    chosen_top = top_two_indices[:, 0]

    records = []
    df_reset = df.reset_index(drop=True)
    for _, contest_indices in df_reset.groupby("Concurso").indices.items():
        contest_indices = list(contest_indices)
        contest_probs = probabilities[contest_indices]
        duplo_idxs = {
            contest_indices[i]
            for i in _select_duplo_indices(contest_probs, alpha=alpha, duplo_count=duplo_count)
        }

        for idx in contest_indices:
            seco_choice = classes[chosen_top[idx]]
            duplo = idx in duplo_idxs
            if duplo:
                duplo_choices = [classes[top_two_indices[idx, 0]], classes[top_two_indices[idx, 1]]]
            else:
                duplo_choices = []
            records.append(
                {
                    "Concurso": df_reset.iloc[idx]["Concurso"],
                    "Resultado": df_reset.iloc[idx]["Resultado"],
                    "Seco": seco_choice,
                    "Duplo": duplo,
                    "Duplo_opcoes": duplo_choices,
                    "Probabilidades": probabilities[idx],
                }
            )
    return pd.DataFrame(records)


def _contest_hits(card_df: pd.DataFrame, class_to_idx: Mapping[str, int]) -> pd.DataFrame:
    def _row_hit(row: pd.Series) -> tuple[bool, bool, float]:
        seco_hit = row["Resultado"] == row["Seco"]
        duplo_hit = row["Resultado"] in row["Duplo_opcoes"] if row["Duplo"] else False
        actual_idx = class_to_idx.get(row["Resultado"], None)
        prob_actual = row["Probabilidades"][actual_idx] if actual_idx is not None else 0.0
        return seco_hit or duplo_hit, seco_hit, prob_actual

    hit_data = card_df.apply(_row_hit, axis=1, result_type="expand")
    card_df = card_df.copy()
    card_df[["Hit", "Seco_hit", "Prob_real"]] = hit_data
    return card_df


def _compute_penalty(card_df: pd.DataFrame) -> pd.Series:
    epsilon = 1e-12
    return -np.log(card_df["Prob_real"] + epsilon)


def _expected_hits_per_contest(card_df: pd.DataFrame, class_to_idx: Mapping[str, int]) -> pd.Series:
    def _row_expectation(row: pd.Series) -> float:
        probs = row["Probabilidades"]
        if row["Duplo"]:
            return sum(probs[class_to_idx[c]] for c in row["Duplo_opcoes"])
        return probs[class_to_idx[row["Seco"]]]

    expectations = card_df.apply(_row_expectation, axis=1)
    return expectations.groupby(card_df["Concurso"]).sum()


def _coverage_ratio(card_df: pd.DataFrame) -> float:
    total_errors = (~card_df["Seco_hit"]).sum()
    if total_errors == 0:
        return 0.0
    covered_errors = ((~card_df["Seco_hit"]) & card_df["Duplo"] & card_df["Hit"]).sum()
    return covered_errors / total_errors


def evaluate_card(
    df: pd.DataFrame,
    prob_columns: Iterable[str],
    class_order: Sequence[str],
    alpha: float,
    duplo_count: int = 5,
    survival_thresholds: Sequence[int] = (14, 13, 12),
) -> CardMetrics:
    probabilities = df[list(prob_columns)].to_numpy(dtype=float)
    card_df = _build_card_frame(df.reset_index(drop=True), probabilities, class_order, alpha, duplo_count)

    class_to_idx = {c: i for i, c in enumerate(class_order)}
    card_df = _contest_hits(card_df, class_to_idx)

    hits_per_contest = card_df.groupby("Concurso")["Hit"].sum()
    survival = {}
    total_contests = len(hits_per_contest)
    for threshold in survival_thresholds:
        survived = (hits_per_contest >= threshold).sum()
        survival[threshold] = (survived / total_contests) * 100 if total_contests else 0.0

    coverage = _coverage_ratio(card_df)
    expected_hits = _expected_hits_per_contest(card_df, class_to_idx).mean()
    penalty = _compute_penalty(card_df).groupby(card_df["Concurso"]).sum().mean()

    return CardMetrics(survival=survival, duplo_coverage=coverage, expected_hits=expected_hits, penalty=penalty)


def summarize_alpha_grid(
    df: pd.DataFrame,
    prob_columns: Iterable[str],
    class_order: Sequence[str],
    alphas: Sequence[float],
    duplo_count: int = 5,
) -> pd.DataFrame:
    records: List[dict] = []
    for alpha in alphas:
        metrics = evaluate_card(df, prob_columns, class_order, alpha, duplo_count)
        record = {
            "alpha": alpha,
            "pct_14": metrics.survival.get(14, 0.0),
            "pct_13": metrics.survival.get(13, 0.0),
            "pct_12": metrics.survival.get(12, 0.0),
            "duplo_coverage": metrics.duplo_coverage,
            "expected_hits": metrics.expected_hits,
            "penalty": metrics.penalty,
        }
        records.append(record)
    return pd.DataFrame(records)
