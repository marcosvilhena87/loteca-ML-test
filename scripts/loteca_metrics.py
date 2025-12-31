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
    hits_by_contest: pd.Series
    p14_by_contest: pd.Series
    p14_medio: float
    p13_by_contest: pd.Series
    p13_medio: float


def _exact_hit_distribution(probabilities: Sequence[float]) -> List[float]:
    max_hits = len(probabilities)
    dp = [1.0] + [0.0] * max_hits

    for p in probabilities:
        for k in range(max_hits, 0, -1):
            dp[k] = dp[k] * (1 - p) + dp[k - 1] * p
        dp[0] = dp[0] * (1 - p)

    return dp


def compute_hit_probabilities(probabilities: Sequence[float]) -> dict[int, float]:
    distribution = _exact_hit_distribution(probabilities)
    return {
        target: distribution[target] if len(distribution) > target else 0.0
        for target in (13, 14)
    }


def _select_duplo_indices(probabilities: np.ndarray, alpha: float, duplo_count: int) -> np.ndarray:
    epsilon = 1e-12
    adjusted = probabilities + epsilon
    sorted_probs = np.sort(adjusted, axis=1)[:, ::-1]
    p1 = sorted_probs[:, 0]
    p2 = sorted_probs[:, 1]

    marginal_gain = np.log((p1 + (1 + alpha) * p2) / p1)
    count = min(duplo_count, len(marginal_gain))
    return np.argsort(marginal_gain)[::-1][:count]


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
    def _row_hit(row: pd.Series) -> tuple[bool, bool, float, float]:
        seco_hit = row["Resultado"] == row["Seco"]
        duplo_hit = row["Resultado"] in row["Duplo_opcoes"] if row["Duplo"] else False
        actual_idx = class_to_idx.get(row["Resultado"], None)
        prob_actual = row["Probabilidades"][actual_idx] if actual_idx is not None else 0.0

        if row["Duplo"]:
            prob_cover = sum(row["Probabilidades"][class_to_idx[c]] for c in row["Duplo_opcoes"])
        else:
            prob_cover = row["Probabilidades"][class_to_idx[row["Seco"]]]

        return seco_hit or duplo_hit, seco_hit, prob_actual, prob_cover

    hit_data = card_df.apply(_row_hit, axis=1, result_type="expand")
    card_df = card_df.copy()
    card_df[["Hit", "Seco_hit", "Prob_real", "Prob_cover"]] = hit_data
    return card_df


def _compute_penalty(card_df: pd.DataFrame) -> pd.Series:
    epsilon = 1e-12
    return -np.log(card_df["Prob_real"] + epsilon)


def _expected_hits_per_contest(card_df: pd.DataFrame) -> pd.Series:
    expectations = card_df["Prob_cover"]
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
    survival_thresholds: Sequence[int] = (14, 13),
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
    expected_hits = _expected_hits_per_contest(card_df).mean()
    penalty = _compute_penalty(card_df).groupby(card_df["Concurso"]).sum().mean()
    contest_probabilities = card_df.groupby("Concurso")["Prob_cover"].apply(list)
    distributions = contest_probabilities.apply(_exact_hit_distribution)
    p14_by_contest = distributions.apply(lambda dist: dist[14] if len(dist) > 14 else 0.0)
    p13_by_contest = distributions.apply(lambda dist: dist[13] if len(dist) > 13 else 0.0)
    p14_medio = p14_by_contest.mean()
    p13_medio = p13_by_contest.mean()

    return CardMetrics(
        survival=survival,
        duplo_coverage=coverage,
        expected_hits=expected_hits,
        penalty=penalty,
        hits_by_contest=hits_per_contest,
        p14_by_contest=p14_by_contest,
        p14_medio=p14_medio,
        p13_by_contest=p13_by_contest,
        p13_medio=p13_medio,
    )


def summarize_alpha_grid(
    df: pd.DataFrame,
    prob_columns: Iterable[str],
    class_order: Sequence[str],
    alphas: Sequence[float],
    duplo_count: int = 5,
    rateio_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rateio_14 = rateio_13 = None
    if rateio_df is not None and not rateio_df.empty:
        # Garantir que não existam rótulos duplicados de concurso antes de alinhar
        # os valores de rateio com as métricas calculadas por concurso. Caso
        # contrário, o reindex do pandas lança erro quando há duplicatas.
        rateio_unique = rateio_df.drop_duplicates(subset="Concurso")
        rateio_indexed = rateio_unique.set_index("Concurso")
        rateio_14 = rateio_indexed.get("Rateio_14")
        rateio_13 = rateio_indexed.get("Rateio_13")

    def _ev_mean(prob_series: pd.Series, rateio_series: pd.Series | None) -> float:
        if rateio_series is None:
            return float("nan")
        aligned_rateio = rateio_series.reindex(prob_series.index).fillna(0)
        return float((prob_series * aligned_rateio).mean())

    records: List[dict] = []
    for alpha in alphas:
        metrics = evaluate_card(df, prob_columns, class_order, alpha, duplo_count)
        record = {
            "alpha": alpha,
            "pct_14": metrics.survival.get(14, 0.0),
            "pct_13": metrics.survival.get(13, 0.0),
            "duplo_coverage": metrics.duplo_coverage,
            "expected_hits": metrics.expected_hits,
            "penalty": metrics.penalty,
            "ev14_medio": _ev_mean(metrics.p14_by_contest, rateio_14),
            "ev13_medio": _ev_mean(metrics.p13_by_contest, rateio_13),
            "ev_total": _ev_mean(metrics.p14_by_contest, rateio_14)
            + _ev_mean(metrics.p13_by_contest, rateio_13),
            "p14_medio": metrics.p14_medio,
            "p13_medio": metrics.p13_medio,
        }
        records.append(record)
    return pd.DataFrame(records)
