import math
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def predict_probabilities(
    df: pd.DataFrame,
    model,
    feature_cols: List[str],
    mix_weight: float,
    mix_clip: float,
) -> pd.DataFrame:
    probs = model.predict_proba(df[feature_cols].to_numpy())
    df = df.copy()
    df["p1_model"] = probs[:, 0]
    df["px_model"] = probs[:, 1]
    df["p2_model"] = probs[:, 2]
    for label in ["1", "x", "2"]:
        market_col = f"p{label}_market"
        model_col = f"p{label}_model"
        blended = (1 - mix_weight) * df[market_col] + mix_weight * df[model_col]
        delta = (blended - df[market_col]).clip(lower=-mix_clip, upper=mix_clip)
        df[f"p{label}_delta"] = delta
        df[f"p{label}"] = df[market_col] + delta
    total = df[["p1", "px", "p2"]].sum(axis=1)
    df["p1"] = df["p1"] / total
    df["px"] = df["px"] / total
    df["p2"] = df["p2"] / total
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
    delta: float,
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
    df["d_duplo"] = df["p_top1"] + df["p_top2"]
    df["p3_duplo"] = 1 - df["d_duplo"]
    df["popularidade_duplo"] = df["p_top1"].apply(lambda val: max(val, 1e-9))

    overround_penalty = (df["overround"] - 1).clip(lower=0)
    df["score_duplo"] = (
        alpha * df["d_duplo"]
        + beta * df["entropy_pred"]
        + gamma * df["p3_duplo"]
        - delta * np.log(df["popularidade_duplo"])
        - overround_penalty
    )

    return df


def _format_double(top1: str, top2: str) -> str:
    ordered = sorted([top1, top2], key=lambda x: {"1": 0, "X": 1, "2": 2}[x])
    return "".join(ordered)


def _score_ticket(
    metrics: Dict[str, float],
    lambda_p14: float,
    mu_pop: float,
    d_target_weight: float,
    double12_penalty_weight: float,
    favorite_duplo_penalty_weight: float,
    favorite_heavy_penalty_weight: float,
) -> float:
    ratio_score = (metrics["log_p13"] - metrics["log_p14"]) * lambda_p14
    score = (
        ratio_score
        + mu_pop * metrics["pop_rarity"]
        - d_target_weight * metrics["d_target_penalty"]
        - double12_penalty_weight * metrics["double12_penalty"]
        - favorite_duplo_penalty_weight * metrics["duplo_favorite_logsum"]
        - favorite_heavy_penalty_weight * metrics["favorite_heavy_penalty"]
    )
    return float(score)


def select_ticket(
    df: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    lambda_p14: float,
    mu_pop: float,
    d_target: float,
    d_target_weight: float,
    contrarian_max: int,
    contrarian_margin_max: float,
    contrarian_gain_min: float,
    contrarian_fav_bonus: float,
    favorite_threshold: float,
    favorite_alt_min: float,
    double12_px_threshold: float,
    double12_penalty_weight: float,
    favorite_duplo_penalty_weight: float,
    favorite_heavy_penalty_weight: float,
    max_favorite: float = 0.75,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = score_duplos(df, alpha=alpha, beta=beta, gamma=gamma, delta=delta)
    logger = logging.getLogger("loteca")
    ratio_eps = 0.01
    ratio_eps_used = ratio_eps

    candidates = df[df["p_top1"] < max_favorite].copy()
    if len(candidates) < 2:
        candidates = df.copy()

    candidates = candidates.sort_values(
        ["score_duplo", "entropy_pred", "p_top2", "margem", "Jogo"],
        ascending=[False, False, False, True, True],
    )
    if not candidates.empty:
        top_n = min(24, len(candidates))
        top_score_idx = candidates.head(top_n).index
        low_d_idx = (
            candidates.sort_values(
                ["d_duplo", "entropy_pred", "p_top2", "score_duplo", "Jogo"],
                ascending=[True, False, False, False, True],
            )
            .head(top_n)
            .index
        )
        candidate_pool = candidates.loc[top_score_idx.union(low_d_idx)]
    else:
        candidate_pool = candidates.copy()
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
    selected: List[int] = []
    pair_scores: List[Dict[str, float]] = []
    if len(candidates) >= 2:
        def score_pairs(candidate_indices: List[int], eps: float) -> List[Dict[str, float]]:
            scored: List[Dict[str, float]] = []
            for i, idx_a in enumerate(candidate_indices[:-1]):
                for idx_b in candidate_indices[i + 1 :]:
                    duplo_indices = [idx_a, idx_b]
                    ticket_df, metrics = _build_ticket(
                        df,
                        duplo_indices=duplo_indices,
                        contrarian_max=contrarian_max,
                        contrarian_margin_max=contrarian_margin_max,
                        contrarian_gain_min=contrarian_gain_min,
                        contrarian_fav_bonus=contrarian_fav_bonus,
                        favorite_threshold=favorite_threshold,
                        favorite_alt_min=favorite_alt_min,
                        d_target=d_target,
                        double12_px_threshold=double12_px_threshold,
                        double12_penalty_weight=double12_penalty_weight,
                        favorite_duplo_penalty_weight=favorite_duplo_penalty_weight,
                        favorite_heavy_penalty_weight=favorite_heavy_penalty_weight,
                        lambda_p14=lambda_p14,
                        mu_pop=mu_pop,
                        d_target_weight=d_target_weight,
                    )
                    ratio = metrics["p13"] / max(metrics["p14"], 1e-12)
                    if ratio < 1.0 + eps:
                        continue
                    pop_rarity = metrics["pop_rarity"]
                    d_target_penalty = metrics["d_target_penalty"]
                    score_ticket = _score_ticket(
                        metrics,
                        lambda_p14=lambda_p14,
                        mu_pop=mu_pop,
                        d_target_weight=d_target_weight,
                        double12_penalty_weight=double12_penalty_weight,
                        favorite_duplo_penalty_weight=favorite_duplo_penalty_weight,
                        favorite_heavy_penalty_weight=favorite_heavy_penalty_weight,
                    )
                    scored.append(
                        {
                            "idx_a": int(idx_a),
                            "idx_b": int(idx_b),
                            "jogo_a": int(df.loc[idx_a, "Jogo"]),
                            "jogo_b": int(df.loc[idx_b, "Jogo"]),
                            "d1": float(df.loc[idx_a, "d_duplo"]),
                            "d2": float(df.loc[idx_b, "d_duplo"]),
                            "g13_component": float(metrics["g13_component"]),
                            "p13": float(metrics["p13"]),
                            "p14": float(metrics["p14"]),
                            "pop_rarity": float(pop_rarity),
                            "d_target_penalty": float(d_target_penalty),
                            "score_ticket": float(score_ticket),
                        }
                    )
            return scored

        candidate_indices = candidate_pool.index.to_list()
        pair_scores = score_pairs(candidate_indices, ratio_eps)
        if not pair_scores and len(candidate_pool) < len(candidates):
            pair_scores = score_pairs(candidates.index.to_list(), ratio_eps)
        if not pair_scores and ratio_eps > 0:
            ratio_eps_used = 0.0
            pair_scores = score_pairs(candidates.index.to_list(), ratio_eps_used)
        if pair_scores:
            pair_scores_sorted = sorted(pair_scores, key=lambda row: row["score_ticket"], reverse=True)
            best = pair_scores_sorted[0]
            selected = [best["idx_a"], best["idx_b"]]
            top_pairs = pair_scores_sorted[:10]
        else:
            top_pairs = []
    else:
        top_pairs = []

    if selected:
        ticket_df, metrics = _build_ticket(
            df,
            duplo_indices=selected,
            contrarian_max=contrarian_max,
            contrarian_margin_max=contrarian_margin_max,
            contrarian_gain_min=contrarian_gain_min,
            contrarian_fav_bonus=contrarian_fav_bonus,
            favorite_threshold=favorite_threshold,
            favorite_alt_min=favorite_alt_min,
            d_target=d_target,
            double12_px_threshold=double12_px_threshold,
            double12_penalty_weight=double12_penalty_weight,
            favorite_duplo_penalty_weight=favorite_duplo_penalty_weight,
            favorite_heavy_penalty_weight=favorite_heavy_penalty_weight,
            lambda_p14=lambda_p14,
            mu_pop=mu_pop,
            d_target_weight=d_target_weight,
        )
    else:
        ticket_df = df.copy()
        metrics = _build_ticket_metrics(
            ticket_df,
            duplo_indices=[],
            d_target=d_target,
            double12_px_threshold=double12_px_threshold,
        )

    summary = {
        "duplos": int((ticket_df["tipo"] == "DUPLO").sum()),
        "secos": int((ticket_df["tipo"] == "SECO").sum()),
        "coverage_increment": float(ticket_df.loc[ticket_df["tipo"] == "DUPLO", "p_top2"].sum()),
        "entropy_duplos": float(ticket_df.loc[ticket_df["tipo"] == "DUPLO", "entropy_pred"].mean()),
        "entropy_total": float(ticket_df["entropy_pred"].mean()),
        "top_duplos": top_duplos,
        "top_pairs": top_pairs,
        "p13_approx": float(metrics["p13"]),
        "p14_approx": float(metrics["p14"]),
        "p13_p14_ratio": float(metrics["p13"] / max(metrics["p14"], 1e-12)),
        "pop_rarity": float(metrics["pop_rarity"]),
        "pop_score_raw": float(metrics["pop_score_raw"]),
        "contrarian_count": int(metrics["contrarian_count"]),
        "favorite_heavy_count": int(metrics["favorite_heavy_count"]),
        "double12_penalty": float(metrics["double12_penalty"]),
        "ratio_filter_eps": float(ratio_eps_used),
    }

    return ticket_df, summary


def _build_ticket(
    df: pd.DataFrame,
    duplo_indices: List[int],
    contrarian_max: int,
    contrarian_margin_max: float,
    contrarian_gain_min: float,
    contrarian_fav_bonus: float,
    favorite_threshold: float,
    favorite_alt_min: float,
    d_target: float,
    double12_px_threshold: float,
    double12_penalty_weight: float,
    favorite_duplo_penalty_weight: float,
    favorite_heavy_penalty_weight: float,
    lambda_p14: float,
    mu_pop: float,
    d_target_weight: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    ticket_df = df.copy()
    ticket_df["tipo"] = "SECO"
    ticket_df["is_contrarian"] = False
    if duplo_indices:
        ticket_df.loc[duplo_indices, "tipo"] = "DUPLO"

    secos_mask = ticket_df["tipo"] == "SECO"
    ticket_df.loc[secos_mask, "seco_choice"] = ticket_df.loc[secos_mask, "top1"]
    ticket_df.loc[secos_mask, "p_seco_choice"] = ticket_df.loc[secos_mask, "p_top1"]

    contrarian_candidates = ticket_df[secos_mask].copy()
    contrarian_candidates["contrarian"] = (
        (contrarian_candidates["margem"] <= contrarian_margin_max)
        | (
            (contrarian_candidates["p_top1"] >= favorite_threshold)
            & (contrarian_candidates["p_top2"] >= favorite_alt_min)
        )
    )
    contrarian_candidates = contrarian_candidates[contrarian_candidates["contrarian"]]
    if not contrarian_candidates.empty and contrarian_max > 0:
        contrarian_candidates["cost_log_p13"] = (
            np.log(contrarian_candidates["p_top1"].clip(1e-9))
            - np.log(contrarian_candidates["p_top2"].clip(1e-9))
        )
        contrarian_candidates["pop_gain"] = contrarian_candidates["cost_log_p13"]
        contrarian_candidates = contrarian_candidates[
            contrarian_candidates["pop_gain"] >= contrarian_gain_min
        ]
        contrarian_candidates["favorite_bonus"] = np.where(
            contrarian_candidates["p_top1"] >= favorite_threshold,
            contrarian_fav_bonus,
            0.0,
        )
        contrarian_candidates["contrarian_score"] = (
            contrarian_candidates["cost_log_p13"] - contrarian_candidates["favorite_bonus"]
        )
        contrarian_candidates = contrarian_candidates.sort_values(
            ["contrarian_score", "margem"], ascending=[True, True]
        )
        if not contrarian_candidates.empty:
            base_metrics = _build_ticket_metrics(
                ticket_df,
                duplo_indices=duplo_indices,
                d_target=d_target,
                double12_px_threshold=double12_px_threshold,
            )
            base_score = _score_ticket(
                base_metrics,
                lambda_p14=lambda_p14,
                mu_pop=mu_pop,
                d_target_weight=d_target_weight,
                double12_penalty_weight=double12_penalty_weight,
                favorite_duplo_penalty_weight=favorite_duplo_penalty_weight,
                favorite_heavy_penalty_weight=favorite_heavy_penalty_weight,
            )
            chosen_indices: List[int] = []
            for idx in contrarian_candidates.index:
                if len(chosen_indices) >= contrarian_max:
                    break
                trial_ticket = ticket_df.copy()
                trial_ticket.loc[idx, "seco_choice"] = trial_ticket.loc[idx, "top2"]
                trial_ticket.loc[idx, "p_seco_choice"] = trial_ticket.loc[idx, "p_top2"]
                trial_metrics = _build_ticket_metrics(
                    trial_ticket,
                    duplo_indices=duplo_indices,
                    d_target=d_target,
                    double12_px_threshold=double12_px_threshold,
                )
                trial_score = _score_ticket(
                    trial_metrics,
                    lambda_p14=lambda_p14,
                    mu_pop=mu_pop,
                    d_target_weight=d_target_weight,
                    double12_penalty_weight=double12_penalty_weight,
                    favorite_duplo_penalty_weight=favorite_duplo_penalty_weight,
                    favorite_heavy_penalty_weight=favorite_heavy_penalty_weight,
                )
                if trial_score > base_score:
                    ticket_df = trial_ticket
                    base_score = trial_score
                    chosen_indices.append(idx)
            if chosen_indices:
                ticket_df.loc[chosen_indices, "is_contrarian"] = True

    ticket_df["palpite"] = ticket_df.apply(
        lambda row: _format_double(row["top1"], row["top2"])
        if row["tipo"] == "DUPLO"
        else row["seco_choice"],
        axis=1,
    )
    metrics = _build_ticket_metrics(
        ticket_df,
        duplo_indices=duplo_indices,
        d_target=d_target,
        double12_px_threshold=double12_px_threshold,
    )
    return ticket_df, metrics


def _build_ticket_metrics(
    ticket_df: pd.DataFrame,
    duplo_indices: List[int],
    d_target: float,
    double12_px_threshold: float,
) -> Dict[str, float]:
    secos = ticket_df[ticket_df["tipo"] == "SECO"]
    duplos = ticket_df.loc[duplo_indices] if duplo_indices else ticket_df[ticket_df["tipo"] == "DUPLO"]

    log_p_secos = np.log(secos["p_seco_choice"].clip(1e-12)).sum() if not secos.empty else 0.0
    d_values = duplos["d_duplo"].to_list()
    if len(d_values) == 2:
        d1, d2 = d_values
        g13_component = d1 * (1 - d2) + d2 * (1 - d1)
        p14_component = d1 * d2
    elif len(d_values) == 1:
        d1 = d_values[0]
        g13_component = d1
        p14_component = d1
    else:
        g13_component = 1.0
        p14_component = 1.0

    p13 = math.exp(log_p_secos) * g13_component
    p14 = math.exp(log_p_secos) * p14_component

    contrarian_count = int((secos["seco_choice"] == secos["top2"]).sum())
    favorite_heavy_count = int((secos["p_top1"] >= 0.62).sum())
    log_p_secos_pop = float(np.log(secos["p_seco_choice"].clip(1e-12)).sum())
    log_p_duplos_pop = float(np.log(duplos["d_duplo"].clip(1e-12)).sum()) if not duplos.empty else 0.0
    pop_score_raw = log_p_secos_pop + log_p_duplos_pop
    pop_rarity = -pop_score_raw
    duplo_favorite_logsum = (
        float(np.log(duplos["p_top1"].clip(1e-12)).sum()) if not duplos.empty else 0.0
    )
    if not duplos.empty:
        duplo_formats = duplos.apply(
            lambda row: _format_double(row["top1"], row["top2"]), axis=1
        )
        double12_mask = (duplo_formats == "12") & (duplos["px"] > double12_px_threshold)
        double12_penalty = float((duplos.loc[double12_mask, "px"] - double12_px_threshold).sum())
    else:
        double12_penalty = 0.0
    d_target_penalty = 0.0
    if d_values:
        high_k = 2.0
        for d_val in d_values:
            if d_val > d_target:
                d_target_penalty += (d_val - d_target) ** 2 * high_k
    favorite_heavy_penalty = float(max(favorite_heavy_count - 2, 0))

    return {
        "p13": float(p13),
        "p14": float(p14),
        "g13_component": float(g13_component),
        "log_p13": float(math.log(max(p13, 1e-12))),
        "log_p14": float(math.log(max(p14, 1e-12))),
        "pop_score_raw": float(pop_score_raw),
        "pop_rarity": float(pop_rarity),
        "duplo_favorite_logsum": float(duplo_favorite_logsum),
        "double12_penalty": float(double12_penalty),
        "d_target_penalty": float(d_target_penalty),
        "contrarian_count": contrarian_count,
        "favorite_heavy_count": favorite_heavy_count,
        "favorite_heavy_penalty": float(favorite_heavy_penalty),
    }
