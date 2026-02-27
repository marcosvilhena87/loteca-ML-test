#!/usr/bin/env python3
"""Generate Loteca palpite with 12 secos, 2 duplos, 0 triplos."""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

LOGGER = logging.getLogger(__name__)
OUTCOMES = ("1", "X", "2")
DOUBLE_TOKENS = {"1X", "12", "X2"}


def setup_logging(debug_path: Path) -> None:
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(debug_path, mode="a", encoding="utf-8"), logging.StreamHandler()],
    )


def parse_decimal(value: str) -> float:
    return float(value.strip().replace(",", "."))


def read_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        rows = list(reader)
    return sorted(rows, key=lambda r: int(r["Jogo"]))


def normalized_probs(row: dict) -> Dict[str, float]:
    implied = {
        "1": 1.0 / parse_decimal(row["Odds_1"]),
        "X": 1.0 / parse_decimal(row["Odds_X"]),
        "2": 1.0 / parse_decimal(row["Odds_2"]),
    }
    total = sum(implied.values())
    return {k: v / total for k, v in implied.items()}


def ordered_symbols(prob_map: Dict[str, float]) -> List[str]:
    return sorted(prob_map.keys(), key=lambda symbol: prob_map[symbol], reverse=True)


def double_token(a: str, b: str) -> str:
    pair = "".join(sorted((a, b), key=lambda s: "1X2".index(s)))
    if pair == "1X":
        return "1X"
    if pair == "12":
        return "12"
    if pair == "X2":
        return "X2"
    raise ValueError(f"Par inválido para duplo: {a}, {b}")


def renormalize(prob_map: Dict[str, float]) -> Dict[str, float]:
    total = sum(prob_map.values())
    if total <= 0:
        return {k: 1.0 / len(prob_map) for k in prob_map}
    return {k: v / total for k, v in prob_map.items()}


def apply_bayesian_position_adjustment(market_probs: Dict[str, float], pos_profile: Dict[str, float]) -> Dict[str, float]:
    ordered_market = ordered_symbols(market_probs)
    rank_key = {
        ordered_market[0]: "p_top1_hit",
        ordered_market[1]: "p_top2_hit",
        ordered_market[2]: "p_top3_hit",
    }
    adjusted = {
        outcome: market_probs[outcome] * float(pos_profile.get(rank_key[outcome], 1.0)) for outcome in OUTCOMES
    }
    return renormalize(adjusted)


def monte_carlo_distribution(per_match: List[dict], picks_by_game: Dict[int, str], n_simulations: int = 20000) -> Dict[str, float]:
    bins = {f"P({k})": 0 for k in range(11, 15)}
    rng = random.Random(2025)

    for _ in range(n_simulations):
        hits = 0
        for game in per_match:
            draw = rng.random()
            cumulative = 0.0
            sampled = "2"
            for outcome in OUTCOMES:
                cumulative += game[f"prob_{outcome}"]
                if draw <= cumulative:
                    sampled = outcome
                    break

            pick_token = picks_by_game[game["jogo"]]
            if sampled in pick_token:
                hits += 1

        for k in range(11, 15):
            if hits >= k:
                bins[f"P({k})"] += 1

    return {k: v / n_simulations for k, v in bins.items()}


def symbol_excess_penalty(token: str, counts: Dict[str, int], target_each: float = 14 / 3) -> float:
    if token in DOUBLE_TOKENS:
        return 0.0

    future = counts.copy()
    future[token] += 1
    before = sum(max(0.0, count - target_each) for count in counts.values())
    after = sum(max(0.0, count - target_each) for count in future.values())
    return max(0.0, after - before)


def run_penalty(jogo: int, token: str, picks_by_game: Dict[int, str], window: int = 3) -> float:
    if token in DOUBLE_TOKENS:
        return 0.0

    penalty = 0.0
    for prev in range(max(1, jogo - window), jogo):
        prev_token = picks_by_game.get(prev)
        if prev_token == token:
            penalty += 0.5
    return penalty


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    next_path = root / "data" / "proximo_concurso.csv"
    model_path = root / "models" / "model.json"
    out_path = root / "output" / "predictions.csv"
    debug_path = root / "output" / "debug.log"

    setup_logging(debug_path)
    if not model_path.exists():
        raise FileNotFoundError("Execute scripts/train_model.py antes de prever.")

    model = json.loads(model_path.read_text(encoding="utf-8"))
    rows = read_rows(next_path)
    if len(rows) != 14:
        raise ValueError(f"Concurso de entrada possui {len(rows)} jogos (esperado exatamente 14).")

    expected = model.get("expected_top_hits", {})
    target_top2_exposure = max(0, round(float(expected.get("top2", 0.0)) - 2))
    target_top3_exposure = max(0, round(float(expected.get("top3", 0.0))))

    per_match = []
    for row in rows:
        pos = str(int(row["Jogo"]))
        market_probs = normalized_probs(row)
        pos_profile = model["position_profiles"].get(pos, {})
        probs = apply_bayesian_position_adjustment(market_probs, pos_profile)
        ordered = ordered_symbols(probs)
        top1, top2, top3 = ordered
        margin = probs[top1] - probs[top2]

        p_top1_hist = float(pos_profile.get("p_top1_hit", 0.5))
        p_top2_hist = float(pos_profile.get("p_top2_hit", 0.3))
        p_top3_hist = float(pos_profile.get("p_top3_hit", 0.2))

        w_pos = 1 + p_top2_hist - p_top1_hist
        marginal_double_gain = probs[top2] * max(0.2, w_pos)
        double_score = marginal_double_gain * (1 - margin)
        zebra2_score = (1 - margin) * probs[top2] * (1 - p_top1_hist)
        zebra3_score = (1 - margin) * probs[top3] * (1 - p_top1_hist) * (0.75 + p_top3_hist)

        per_match.append(
            {
                "concurso": row["Concurso"],
                "jogo": int(row["Jogo"]),
                "mandante": row["Mandante"],
                "visitante": row["Visitante"],
                "prob_1": probs["1"],
                "prob_X": probs["X"],
                "prob_2": probs["2"],
                "top1": top1,
                "top2": top2,
                "top3": top3,
                "margin": margin,
                "p_top1_hist": p_top1_hist,
                "p_top2_hist": p_top2_hist,
                "p_top3_hist": p_top3_hist,
                "double_score": double_score,
                "marginal_double_gain": marginal_double_gain,
                "zebra2_score": zebra2_score,
                "zebra3_score": zebra3_score,
            }
        )

    picks_by_game = {g["jogo"]: g["top1"] for g in per_match}

    # Seleção de duplos com ganho marginal bayesiano + dispersão posicional.
    used_games = set()
    double_candidates = sorted(per_match, key=lambda g: g["double_score"], reverse=True)
    selected_doubles = []
    for candidate in double_candidates:
        if candidate["jogo"] in used_games:
            continue
        if any(abs(candidate["jogo"] - d["jogo"]) <= 1 for d in selected_doubles):
            continue
        selected_doubles.append(candidate)
        used_games.add(candidate["jogo"])
        if len(selected_doubles) == 2:
            break

    if len(selected_doubles) < 2:
        for candidate in double_candidates:
            if candidate["jogo"] in used_games:
                continue
            selected_doubles.append(candidate)
            used_games.add(candidate["jogo"])
            if len(selected_doubles) == 2:
                break

    for game in selected_doubles:
        picks_by_game[game["jogo"]] = double_token(game["top1"], game["top2"])

    # Exposição estrutural controlada para top2/top3 via secos não-top1.
    symbol_counts = {"1": 0, "X": 0, "2": 0}
    for token in picks_by_game.values():
        if token in symbol_counts:
            symbol_counts[token] += 1

    available_for_zebra = [
        g for g in per_match if picks_by_game[g["jogo"]] not in DOUBLE_TOKENS
    ]

    zebra2_candidates = sorted(available_for_zebra, key=lambda g: g["zebra2_score"], reverse=True)
    zebra3_candidates = sorted(available_for_zebra, key=lambda g: g["zebra3_score"], reverse=True)

    zebra2_done = 0
    changed_games = set()
    zebra2_ranked = []
    for game in zebra2_candidates:
        jogo = game["jogo"]
        token = game["top2"]
        cost = 0.2 * (game["prob_" + game["top1"]] - game["prob_" + token])
        cost += 0.10 * symbol_excess_penalty(token, symbol_counts)
        cost += 0.10 * run_penalty(jogo, token, picks_by_game)
        utility = game["zebra2_score"] - cost
        zebra2_ranked.append((utility, game))

    for utility, game in sorted(zebra2_ranked, key=lambda x: x[0], reverse=True):
        if zebra2_done >= target_top2_exposure:
            break
        jogo = game["jogo"]
        token = game["top2"]
        if jogo in changed_games:
            continue
        if picks_by_game[jogo] in DOUBLE_TOKENS:
            continue

        old = picks_by_game[jogo]
        picks_by_game[jogo] = token
        if old in symbol_counts:
            symbol_counts[old] -= 1
        symbol_counts[token] += 1
        changed_games.add(jogo)
        zebra2_done += 1
        LOGGER.info("Zebra top2 aplicada no jogo %s (utilidade=%.4f).", jogo, utility)

    zebra3_done = 0
    zebra3_ranked = []
    for game in zebra3_candidates:
        jogo = game["jogo"]
        token = game["top3"]
        if picks_by_game[jogo] in DOUBLE_TOKENS:
            continue
        cost = 0.2 * (game["prob_" + game["top1"]] - game["prob_" + token])
        cost += 0.10 * symbol_excess_penalty(token, symbol_counts)
        cost += 0.10 * run_penalty(jogo, token, picks_by_game)
        utility = game["zebra3_score"] - cost
        zebra3_ranked.append((utility, game))

    for utility, game in sorted(zebra3_ranked, key=lambda x: x[0], reverse=True):
        if zebra3_done >= target_top3_exposure:
            break
        jogo = game["jogo"]
        token = game["top3"]
        if jogo in changed_games:
            continue
        if picks_by_game[jogo] in DOUBLE_TOKENS:
            continue

        old = picks_by_game[jogo]
        picks_by_game[jogo] = token
        if old in symbol_counts:
            symbol_counts[old] -= 1
        symbol_counts[token] += 1
        changed_games.add(jogo)
        zebra3_done += 1
        LOGGER.info("Zebra top3 aplicada no jogo %s (utilidade=%.4f).", jogo, utility)

    double_games = sorted([g["jogo"] for g in selected_doubles])
    LOGGER.info("Duplos selecionados por ganho marginal (12 secos + 2 duplos + 0 triplos): %s", double_games)

    r14_categorical = []
    r14_prob = []
    prediction_rows = []

    for game in sorted(per_match, key=lambda g: g["jogo"]):
        guess = picks_by_game[game["jogo"]]
        pick_type = "duplo" if guess in DOUBLE_TOKENS else "seco"

        r14_categorical.append(guess)
        r14_prob.append(f"{game['top1']}>{game['top2']}>{game['top3']}")

        LOGGER.info(
            "Jogo %02d %s x %s | margin=%.3f probs(1=%.3f X=%.3f 2=%.3f) top=%s/%s/%s | hist(top1=%.3f top2=%.3f top3=%.3f) | score_duplo=%.4f ganho_duplo=%.4f | tipo=%s palpite=%s",
            game["jogo"],
            game["mandante"],
            game["visitante"],
            game["margin"],
            game["prob_1"],
            game["prob_X"],
            game["prob_2"],
            game["top1"],
            game["top2"],
            game["top3"],
            game["p_top1_hist"],
            game["p_top2_hist"],
            game["p_top3_hist"],
            game["double_score"],
            game["marginal_double_gain"],
            pick_type,
            guess,
        )

        prediction_rows.append(
            {
                "Concurso": game["concurso"],
                "Jogo": game["jogo"],
                "Mandante": game["mandante"],
                "Visitante": game["visitante"],
                "Top1": game["top1"],
                "Top2": game["top2"],
                "Top3": game["top3"],
                "Palpite": guess,
                "Tipo": pick_type,
            }
        )

    total_picks = sum(2 if token in DOUBLE_TOKENS else 3 if token == "1X2" else 1 for token in r14_categorical)
    secos = sum(1 for token in r14_categorical if token in {"1", "X", "2"})
    duplos = sum(1 for token in r14_categorical if token in DOUBLE_TOKENS)
    triplos = sum(1 for token in r14_categorical if token == "1X2")

    top2_exposed = sum(1 for g in per_match if picks_by_game[g["jogo"]] == g["top2"] or picks_by_game[g["jogo"]] in DOUBLE_TOKENS)
    top3_exposed = sum(1 for g in per_match if picks_by_game[g["jogo"]] == g["top3"])

    mc = monte_carlo_distribution(per_match, picks_by_game)

    LOGGER.info("Arquitetura final: %s secos, %s duplos, %s triplos, %s picks totais", secos, duplos, triplos, total_picks)
    LOGGER.info("R14 categórico palpite: %s", r14_categorical)
    LOGGER.info("R14 probabilístico (top1/top2/top3): %s", r14_prob)
    LOGGER.info("Esperança histórica top hits por concurso: %s", expected)
    LOGGER.info(
        "Exposição estrutural: top2_expostos=%s (meta~%s), top3_secos=%s (meta~%s)",
        top2_exposed,
        2 + target_top2_exposure,
        top3_exposed,
        target_top3_exposure,
    )
    LOGGER.info("Dispersão símbolos em secos: %s", symbol_counts)
    LOGGER.info("Monte Carlo (20k sims): %s", mc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["Concurso", "Jogo", "Mandante", "Visitante", "Top1", "Top2", "Top3", "Palpite", "Tipo"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(prediction_rows)

    LOGGER.info("Predições gravadas em %s", out_path)


if __name__ == "__main__":
    main()
