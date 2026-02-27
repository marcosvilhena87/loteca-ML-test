#!/usr/bin/env python3
"""Generate Loteca palpite with strategy search and Monte Carlo evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List

LOGGER = logging.getLogger(__name__)
OUTCOMES = ("1", "X", "2")
DOUBLE_TOKENS = {"1X", "12", "X2"}

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))



def setup_logging(debug_path: Path) -> None:
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(debug_path, mode="a", encoding="utf-8"), logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera palpites Loteca com busca de estratégias irmãs.")
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=int(os.getenv("LOTECA_N_SIMULATIONS", "200000")),
        help="Número de simulações do Monte Carlo (também lê LOTECA_N_SIMULATIONS).",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=int(os.getenv("LOTECA_N_CANDIDATES", "80")),
        help="Quantidade de cartões candidatos por estratégia (30-200 recomendado).",
    )
    parser.add_argument(
        "--quick-simulations",
        type=int,
        default=int(os.getenv("LOTECA_QUICK_SIMULATIONS", "5000")),
        help="Simulações rápidas para pré-seleção dos cartões candidatos.",
    )
    parser.add_argument(
        "--mc-seed",
        type=int,
        default=int(os.getenv("LOTECA_MC_SEED", "2025")),
        help="Seed base do Monte Carlo para comparabilidade entre cartões.",
    )
    return parser.parse_args()


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


def apply_bayesian_position_adjustment(
    market_probs: Dict[str, float], pos_profile: Dict[str, float], margin: float, blend_k: float = 12.0
) -> Dict[str, float]:
    prior = renormalize(
        {
            "1": float(pos_profile.get("p_outcome_1", 1 / 3)),
            "X": float(pos_profile.get("p_outcome_X", 1 / 3)),
            "2": float(pos_profile.get("p_outcome_2", 1 / 3)),
        }
    )
    weight_market = sigmoid(blend_k * margin)
    return {
        symbol: (weight_market * market_probs[symbol]) + ((1.0 - weight_market) * prior[symbol])
        for symbol in OUTCOMES
    }


def card_metrics(per_match: List[dict], picks_by_game: Dict[int, str]) -> Dict[str, int]:
    top2_single_exposed = sum(1 for g in per_match if picks_by_game[g["jogo"]] == g["top2"])
    top2_covered_by_double = sum(
        1 for g in per_match if picks_by_game[g["jogo"]] in DOUBLE_TOKENS and g["top2"] in picks_by_game[g["jogo"]]
    )
    top3_single_exposed = sum(1 for g in per_match if picks_by_game[g["jogo"]] == g["top3"])
    dry_counts = {symbol: sum(1 for token in picks_by_game.values() if token == symbol) for symbol in OUTCOMES}
    return {
        "top2_single_exposed": top2_single_exposed,
        "top2_covered_by_double": top2_covered_by_double,
        "top2_exposed": top2_single_exposed + top2_covered_by_double,
        "top3_exposed": top3_single_exposed,
        "min_1_secos": dry_counts["1"],
        "min_X_secos": dry_counts["X"],
        "min_2_secos": dry_counts["2"],
    }


def hard_constraints_ok(per_match: List[dict], picks_by_game: Dict[int, str], target_top2_exposure: int, target_top3_exposure: int) -> bool:
    metrics = card_metrics(per_match, picks_by_game)
    if metrics["min_X_secos"] < 1 or metrics["min_2_secos"] < 1:
        return False
    if metrics["top2_exposed"] < target_top2_exposure:
        return False
    return metrics["top3_exposed"] >= max(0, target_top3_exposure - 1)


def monte_carlo_distribution(
    per_match: List[dict], picks_by_game: Dict[int, str], n_simulations: int, seed: int, prob_prefix: str = "prob"
) -> Dict[str, float]:
    bins = {f"P({k})": 0 for k in range(11, 15)}
    rng = random.Random(seed)
    hits_sum = 0.0
    hits_sq_sum = 0.0

    for _ in range(n_simulations):
        hits = 0
        for game in per_match:
            draw = rng.random()
            cumulative = 0.0
            sampled = "2"
            for outcome in OUTCOMES:
                cumulative += game[f"{prob_prefix}_{outcome}"]
                if draw <= cumulative:
                    sampled = outcome
                    break

            pick_token = picks_by_game[game["jogo"]]
            if sampled in pick_token:
                hits += 1

        hits_sum += hits
        hits_sq_sum += hits * hits
        for k in range(11, 15):
            if hits >= k:
                bins[f"P({k})"] += 1

    mean_hits = hits_sum / n_simulations
    var_hits = max(0.0, (hits_sq_sum / n_simulations) - (mean_hits * mean_hits))
    std_hits = var_hits**0.5

    result = {k: v / n_simulations for k, v in bins.items()}
    result["mean_hits"] = mean_hits
    result["std_hits"] = std_hits
    return result


def expected_hits(per_match: List[dict], picks_by_game: Dict[int, str], prob_prefix: str = "prob") -> float:
    exp = 0.0
    for game in per_match:
        token = picks_by_game[game["jogo"]]
        exp += sum(game[f"{prob_prefix}_{outcome}"] for outcome in OUTCOMES if outcome in token)
    return exp


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
    run_len = 1
    for prev in range(jogo - 1, 0, -1):
        prev_token = picks_by_game.get(prev)
        if prev_token != token:
            break
        run_len += 1

    # Penalidade explosiva para sequências longas do mesmo símbolo.
    if run_len >= 3:
        penalty += 0.8 * (2 ** (run_len - 3))

    for prev in range(max(1, jogo - window), jogo):
        prev_token = picks_by_game.get(prev)
        if prev_token == token:
            penalty += 0.5
    return penalty


def card_hash(picks_by_game: Dict[int, str]) -> str:
    ordered = "|".join(f"{j}:{picks_by_game[j]}" for j in sorted(picks_by_game))
    return hashlib.sha1(ordered.encode("utf-8")).hexdigest()[:12]


def generate_card(
    per_match: List[dict],
    target_top2_exposure: int,
    target_top3_exposure: int,
    max_top2_zebras: int,
    max_top3_zebras: int,
    lambda_top2: float,
    lambda_top3: float,
    rng: random.Random,
    double_top_k: int,
    zebra_top_m: int,
    perturb_prob: float,
    low_margin_threshold: float,
    top3_experimental_max: int,
) -> dict:
    picks_by_game = {g["jogo"]: g["top1"] for g in per_match}

    low_margin_games = [g for g in per_match if g["margin"] <= low_margin_threshold]
    for game in low_margin_games:
        if rng.random() < perturb_prob:
            picks_by_game[game["jogo"]] = game["top2"]

    top3_experimental_done = 0
    for game in sorted(per_match, key=lambda g: g["zebra3_score"], reverse=True):
        if top3_experimental_done >= top3_experimental_max:
            break
        if picks_by_game[game["jogo"]] != game["top1"]:
            continue

        experimental_penalty = 0.35 * (game["prob_" + game["top1"]] - game["prob_" + game["top3"]])
        experimental_utility = game["zebra3_score"] - experimental_penalty
        if experimental_utility > -0.01:
            picks_by_game[game["jogo"]] = game["top3"]
            top3_experimental_done += 1

    used_games = set()
    double_candidates = sorted(per_match, key=lambda g: g["double_score"], reverse=True)
    double_pool = double_candidates[: max(2, double_top_k)]
    selected_doubles = []
    while len(selected_doubles) < 2 and double_pool:
        candidate = rng.choice(double_pool)
        double_pool.remove(candidate)
        if candidate["jogo"] in used_games:
            continue
        if any(abs(candidate["jogo"] - d["jogo"]) <= 1 for d in selected_doubles):
            continue
        selected_doubles.append(candidate)
        used_games.add(candidate["jogo"])

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

    symbol_counts = {"1": 0, "X": 0, "2": 0}
    for token in picks_by_game.values():
        if token in symbol_counts:
            symbol_counts[token] += 1

    changed_games = set()
    top2_soft_done = 0
    top3_soft_done = 0

    for zlabel in ("top2", "top3"):
        zebra_done = 0
        zebra_cap = max_top2_zebras if zlabel == "top2" else max_top3_zebras
        candidates = sorted(per_match, key=lambda g: g[f"zebra{2 if zlabel == 'top2' else 3}_score"], reverse=True)
        candidates = candidates[: max(1, zebra_top_m)]
        rng.shuffle(candidates)
        for game in candidates:
            if zebra_done >= zebra_cap:
                break
            jogo = game["jogo"]
            if jogo in changed_games or picks_by_game[jogo] in DOUBLE_TOKENS:
                continue

            token = game[zlabel]
            score_key = "zebra2_score" if zlabel == "top2" else "zebra3_score"
            cost = 0.2 * (game["prob_" + game["top1"]] - game["prob_" + token])
            cost += 0.10 * symbol_excess_penalty(token, symbol_counts)
            cost += 0.10 * run_penalty(jogo, token, picks_by_game)
            utility = game[score_key] - cost

            # Soft constraint: bônus marginal até a meta, sem forçar aplicação.
            bonus = 0.0
            if zlabel == "top2" and top2_soft_done < target_top2_exposure:
                bonus = lambda_top2 * (target_top2_exposure - top2_soft_done)
            if zlabel == "top3" and top3_soft_done < target_top3_exposure:
                bonus = lambda_top3 * (target_top3_exposure - top3_soft_done)

            if utility <= 0:
                continue
            if utility + bonus <= 0:
                continue

            old = picks_by_game[jogo]
            picks_by_game[jogo] = token
            if old in symbol_counts:
                symbol_counts[old] -= 1
            symbol_counts[token] += 1
            changed_games.add(jogo)
            zebra_done += 1
            if zlabel == "top2":
                top2_soft_done += 1
            else:
                top3_soft_done += 1

    exposure_metrics = card_metrics(per_match, picks_by_game)

    return {
        "picks_by_game": picks_by_game,
        "card_hash": card_hash(picks_by_game),
        "double_games": sorted([g["jogo"] for g in selected_doubles]),
        "symbol_counts": symbol_counts,
        **exposure_metrics,
    }


def mutate_card(per_match: List[dict], base_picks: Dict[int, str], rng: random.Random) -> Dict[int, str]:
    picks = base_picks.copy()
    game = rng.choice(per_match)
    jogo = game["jogo"]
    move = rng.choice(("top1_to_top2", "flip_double", "reduce_run"))

    if move == "top1_to_top2" and picks[jogo] == game["top1"]:
        picks[jogo] = game["top2"]
    elif move == "flip_double" and picks[jogo] in DOUBLE_TOKENS:
        picks[jogo] = game["top1"]
    elif move == "reduce_run":
        prev = picks.get(jogo - 1)
        if prev is not None and picks[jogo] == prev:
            picks[jogo] = game["top2"] if game["top2"] != prev else game["top3"]
    return picks


def local_search(
    per_match: List[dict],
    card: dict,
    target_top2_exposure: int,
    target_top3_exposure: int,
    rng: random.Random,
    n_steps: int = 30,
) -> dict:
    best_picks = card["picks_by_game"].copy()
    best_eh = expected_hits(per_match, best_picks, prob_prefix="prob")
    for _ in range(n_steps):
        trial = mutate_card(per_match, best_picks, rng)
        if not hard_constraints_ok(per_match, trial, target_top2_exposure, target_top3_exposure):
            continue
        trial_eh = expected_hits(per_match, trial, prob_prefix="prob")
        if trial_eh > best_eh:
            best_picks, best_eh = trial, trial_eh

    improved = card.copy()
    improved["picks_by_game"] = best_picks
    improved["card_hash"] = card_hash(best_picks)
    improved.update(card_metrics(per_match, best_picks))
    return improved


def pareto_candidates(candidates: List[tuple], prob_12_floor: float = 0.20) -> List[tuple]:
    feasible = [c for c in candidates if c[3]["P(12)"] >= prob_12_floor]
    if not feasible:
        feasible = candidates
    frontier = []
    for cand in feasible:
        obj_a = cand[3]["P(13)"] + cand[3]["P(14)"]
        obj_b = cand[3]["P(11)"] + cand[3]["P(12)"]
        dominated = False
        for other in feasible:
            if other is cand:
                continue
            other_a = other[3]["P(13)"] + other[3]["P(14)"]
            other_b = other[3]["P(11)"] + other[3]["P(12)"]
            if other_a >= obj_a and other_b >= obj_b and (other_a > obj_a or other_b > obj_b):
                dominated = True
                break
        if not dominated:
            frontier.append(cand)
    return frontier


def score_from_distribution(mc: Dict[str, float], eh: float) -> float:
    return mc["P(12)"] + 2.0 * mc["P(13)"] + 5.0 * mc["P(14)"] + 0.01 * eh


def combined_score(
    mc_market: Dict[str, float],
    eh_market: float,
    mc_adjusted: Dict[str, float],
    eh_adjusted: float,
    market_weight: float = 0.7,
) -> float:
    market_component = score_from_distribution(mc_market, eh_market)
    adjusted_component = score_from_distribution(mc_adjusted, eh_adjusted)
    return market_weight * market_component + (1.0 - market_weight) * adjusted_component


def main() -> None:
    args = parse_args()
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
        market_order = ordered_symbols(market_probs)
        market_margin = market_probs[market_order[0]] - market_probs[market_order[1]]
        pos_profile = model["position_profiles"].get(pos, {})
        probs = apply_bayesian_position_adjustment(market_probs, pos_profile, margin=market_margin)
        ordered = ordered_symbols(probs)
        top1, top2, top3 = ordered
        margin = probs[top1] - probs[top2]

        p_top1_hist = float(pos_profile.get("p_top1_hit", 0.5))
        p_top2_hist = float(pos_profile.get("p_top2_hit", 0.3))
        p_top3_hist = float(pos_profile.get("p_top3_hit", 0.2))

        delta_eh_double = probs[top1] + probs[top2] - probs[top1]
        structural_cost = 0.15 * margin
        marginal_double_gain = delta_eh_double
        double_score = max(0.0, marginal_double_gain - structural_cost)
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
                "market_prob_1": market_probs["1"],
                "market_prob_X": market_probs["X"],
                "market_prob_2": market_probs["2"],
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

    strategies = [
        {"name": "A_base_soft", "max2": target_top2_exposure, "max3": target_top3_exposure, "l2": 0.010, "l3": 0.008},
        {"name": "B_pos_only", "max2": max(1, target_top2_exposure), "max3": max(0, target_top3_exposure - 1), "l2": 0.006, "l3": 0.004},
        {"name": "C_cap_3_2", "max2": 3, "max3": 2, "l2": 0.009, "l3": 0.007},
    ]

    evaluated = []
    for idx, strategy in enumerate(strategies):
        strategy_seed = args.mc_seed + idx * 1000
        strategy_rng = random.Random(strategy_seed)
        candidate_count = max(1, args.n_candidates)
        quick_n = max(100, min(args.quick_simulations, args.n_simulations))
        quick_eval = []
        seen_hashes = set()

        for _ in range(candidate_count):
            card = generate_card(
                per_match,
                target_top2_exposure,
                target_top3_exposure,
                min(strategy["max2"], strategy_rng.randint(0, 3)),
                min(strategy["max3"], strategy_rng.randint(0, 3)),
                strategy["l2"],
                strategy["l3"],
                rng=strategy_rng,
                double_top_k=5,
                zebra_top_m=strategy_rng.randint(5, 10),
                perturb_prob=0.15,
                low_margin_threshold=0.08,
                top3_experimental_max=strategy_rng.randint(0, 1),
            )
            card = local_search(
                per_match,
                card,
                target_top2_exposure=2 + target_top2_exposure,
                target_top3_exposure=target_top3_exposure,
                rng=strategy_rng,
                n_steps=30,
            )
            if not hard_constraints_ok(per_match, card["picks_by_game"], 2 + target_top2_exposure, target_top3_exposure):
                continue
            if card["card_hash"] in seen_hashes:
                continue
            seen_hashes.add(card["card_hash"])
            eh_market = expected_hits(per_match, card["picks_by_game"], prob_prefix="market_prob")
            eh_adjusted = expected_hits(per_match, card["picks_by_game"], prob_prefix="prob")
            mc_quick_market = monte_carlo_distribution(
                per_match,
                card["picks_by_game"],
                n_simulations=quick_n,
                seed=args.mc_seed,
                prob_prefix="market_prob",
            )
            mc_quick_adjusted = monte_carlo_distribution(
                per_match,
                card["picks_by_game"],
                n_simulations=quick_n,
                seed=args.mc_seed,
                prob_prefix="prob",
            )
            quick_score = combined_score(mc_quick_market, eh_market, mc_quick_adjusted, eh_adjusted)
            quick_eval.append((quick_score, card, eh_market, eh_adjusted, mc_quick_adjusted, mc_quick_market))

        if not quick_eval:
            LOGGER.warning("Estratégia %s não gerou cartões válidos sob as restrições atuais.", strategy["name"])
            continue

        quick_eval.sort(key=lambda x: x[0], reverse=True)
        finalists = quick_eval[: min(8, len(quick_eval))]

        best_full = None
        for _, card, eh_market, eh_adjusted, _, _ in finalists:
            mc_full_market = monte_carlo_distribution(
                per_match,
                card["picks_by_game"],
                n_simulations=args.n_simulations,
                seed=args.mc_seed,
                prob_prefix="market_prob",
            )
            mc_full_adjusted = monte_carlo_distribution(
                per_match,
                card["picks_by_game"],
                n_simulations=args.n_simulations,
                seed=args.mc_seed,
                prob_prefix="prob",
            )
            final_score = combined_score(mc_full_market, eh_market, mc_full_adjusted, eh_adjusted)
            candidate = (final_score, strategy, card, mc_full_market, mc_full_adjusted, eh_market, eh_adjusted)
            if best_full is None or candidate[0] > best_full[0]:
                best_full = candidate

        assert best_full is not None
        evaluated.append(best_full)
        LOGGER.info(
            "Estratégia %s | best_score=%.6f | expected_hits_market=%.4f | expected_hits_adjusted=%.4f | MC_market=%s | card_hash=%s | candidatos_unicos=%s",
            strategy["name"],
            best_full[0],
            best_full[5],
            best_full[6],
            best_full[3],
            best_full[2]["card_hash"],
            len(seen_hashes),
        )

    if not evaluated:
        raise RuntimeError("Nenhuma estratégia gerou cartões válidos com as restrições definidas.")

    global_frontier = pareto_candidates(evaluated)
    _, best_strategy, best_card, mc_market, mc_adjusted, exp_hits_market, exp_hits_adjusted = sorted(
        global_frontier, key=lambda x: x[0], reverse=True
    )[0]
    picks_by_game = best_card["picks_by_game"]

    LOGGER.info("Estratégia vencedora: %s", best_strategy["name"])
    LOGGER.info("Cartão vencedor hash: %s", best_card["card_hash"])
    LOGGER.info("Duplos selecionados: %s", best_card["double_games"])

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
                "P12": mc_market["P(12)"],
                "P13": mc_market["P(13)"],
                "P14": mc_market["P(14)"],
                "MeanHits": mc_market["mean_hits"],
                "StdHits": mc_market["std_hits"],
                "Top2Secos": best_card["top2_single_exposed"],
                "Top2Duplos": best_card["top2_covered_by_double"],
                "Top3Secos": best_card["top3_exposed"],
            }
        )

    total_picks = sum(2 if token in DOUBLE_TOKENS else 3 if token == "1X2" else 1 for token in r14_categorical)
    secos = sum(1 for token in r14_categorical if token in {"1", "X", "2"})
    duplos = sum(1 for token in r14_categorical if token in DOUBLE_TOKENS)
    triplos = sum(1 for token in r14_categorical if token == "1X2")

    LOGGER.info("Arquitetura final: %s secos, %s duplos, %s triplos, %s picks totais", secos, duplos, triplos, total_picks)
    LOGGER.info("R14 categórico palpite: %s", r14_categorical)
    LOGGER.info("R14 probabilístico (top1/top2/top3): %s", r14_prob)
    LOGGER.info("Esperança histórica top hits por concurso: %s", expected)
    LOGGER.info(
        "Exposição estrutural: top2_secos=%s, top2_duplos=%s, top2_total=%s (meta~%s), top3_secos=%s (meta~%s)",
        best_card["top2_single_exposed"],
        best_card["top2_covered_by_double"],
        best_card["top2_exposed"],
        2 + target_top2_exposure,
        best_card["top3_exposed"],
        target_top3_exposure,
    )
    LOGGER.info("Dispersão símbolos em secos: %s", best_card["symbol_counts"])
    LOGGER.info("Expected hits do cartão (market): %.4f", exp_hits_market)
    LOGGER.info("Expected hits do cartão (adjusted): %.4f", exp_hits_adjusted)
    LOGGER.info("Monte Carlo market (%s sims): %s", args.n_simulations, mc_market)
    LOGGER.info("Monte Carlo adjusted (%s sims): %s", args.n_simulations, mc_adjusted)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "Concurso",
            "Jogo",
            "Mandante",
            "Visitante",
            "Top1",
            "Top2",
            "Top3",
            "Palpite",
            "Tipo",
            "P12",
            "P13",
            "P14",
            "MeanHits",
            "StdHits",
            "Top2Secos",
            "Top2Duplos",
            "Top3Secos",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(prediction_rows)

    LOGGER.info("Predições gravadas em %s", out_path)


if __name__ == "__main__":
    main()
