import argparse
import csv
import itertools
import logging
import math
import random
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

from scripts.common import dump_json, load_csv, load_json, rank_structure_stats, rank_symbols, setup_logging

LOGGER = logging.getLogger(__name__)


def rank_symbol_map(row):
    ranked = rank_symbols(row)
    return {1: ranked[0], 2: ranked[1], 3: ranked[2]}


def bucket_id(value, n_buckets=10):
    idx = int(value * n_buckets)
    return min(idx, n_buckets - 1)


def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def clamp_prob(p, eps=1e-6):
    return min(1.0 - eps, max(eps, p))


def band_of(model, max_prob):
    for name, (low, high) in model["meta"]["confidence_bands"].items():
        if low <= max_prob < high:
            return name
    return "low"


def apply_calibrator(calibrator, x):
    t = calibrator.get("type", "identity")
    if t == "identity":
        return x
    if t == "constant":
        return calibrator["value"]
    if t == "bucket":
        b = bucket_id(x, calibrator["n_buckets"])
        return calibrator["values"][b]
    if t == "isotonic":
        idx = bisect_right(calibrator["thresholds"], x)
        idx = min(idx, len(calibrator["values"]) - 1)
        return calibrator["values"][idx]
    if t == "platt":
        x2 = math.log(clamp_prob(x) / (1.0 - clamp_prob(x)))
        z = calibrator["weights"][0] + calibrator["weights"][1] * x2
        return sigmoid(z)
    if t == "beta":
        z = (
            calibrator["weights"][0]
            + calibrator["weights"][1] * math.log(clamp_prob(x))
            + calibrator["weights"][2] * math.log(1.0 - clamp_prob(x))
        )
        return sigmoid(z)
    return x


def adjust_prob(model, symbol, raw_prob, max_prob):
    band = band_of(model, max_prob)
    calibrator = model["calibration"][symbol][band]
    calibrated = apply_calibrator(calibrator, raw_prob)
    return model["meta"]["raw_weight"] * raw_prob + model["meta"]["calibration_weight"] * calibrated


def evaluate_soft_penalty(games, selected_rank_sets, soft_config):
    penalty = 0.0
    detail = {}
    targets = soft_config["targets"]
    metric_weights = soft_config["metric_weights"]

    for rank in (1, 2, 3):
        ordered_indices = sorted(range(len(games)), key=lambda i: -games[i][f"p(top{rank})"])
        indicator = [1 if rank in selected_rank_sets[i] else 0 for i in ordered_indices]
        stats = rank_structure_stats(indicator)
        target = targets[f"top{rank}"]

        rank_penalty = 0.0
        metric_breakdown = {}
        for metric, weight in metric_weights[f"top{rank}"].items():
            delta = abs(stats.get(metric, 0.0) - target.get(metric, 0.0))
            metric_penalty = weight * delta
            rank_penalty += metric_penalty
            metric_breakdown[metric] = {
                "weight": weight,
                "delta": delta,
                "penalty": metric_penalty,
            }

        penalty += rank_penalty
        detail[f"top{rank}"] = {
            "actual": stats,
            "target": target,
            "metrics": metric_breakdown,
            "penalty": rank_penalty,
        }
    return penalty, detail


def solve_count_system():
    solutions = []
    for s1 in range(12):
        for s2 in range(12 - s1):
            s3 = 11 - s1 - s2
            for d12 in range(4):
                for d13 in range(4 - d12):
                    d23 = 3 - d12 - d13
                    if s1 + d12 + d13 != 9:
                        continue
                    if s2 + d12 + d23 != 5:
                        continue
                    if s3 + d13 + d23 != 3:
                        continue
                    solutions.append({
                        "S1": s1,
                        "S2": s2,
                        "S3": s3,
                        "D12": d12,
                        "D13": d13,
                        "D23": d23,
                    })
    return solutions


def option_set(option):
    return {
        "S1": {1},
        "S2": {2},
        "S3": {3},
        "D12": {1, 2},
        "D13": {1, 3},
        "D23": {2, 3},
    }[option]


def assign_best_for_distribution(games, counts):
    option_names = ["S1", "S2", "S3", "D12", "D13", "D23"]

    @lru_cache(maxsize=None)
    def dp(i, c1, c2, c3, c12, c13, c23):
        if i == len(games):
            if (c1, c2, c3, c12, c13, c23) == (0, 0, 0, 0, 0, 0):
                return 0.0, []
            return -10**9, []

        remaining = [c1, c2, c3, c12, c13, c23]
        best_score = -10**9
        best_plan = None
        for idx, opt in enumerate(option_names):
            if remaining[idx] <= 0:
                continue
            rs = option_set(opt)
            game = games[i]
            option_score = sum(game["adj_rank_prob"][r] for r in rs)
            new_remaining = remaining.copy()
            new_remaining[idx] -= 1
            future_score, future_plan = dp(i + 1, *new_remaining)
            score = option_score + future_score
            if score > best_score:
                best_score = score
                best_plan = [opt] + future_plan
        return best_score, best_plan

    score, plan = dp(
        0,
        counts["S1"],
        counts["S2"],
        counts["S3"],
        counts["D12"],
        counts["D13"],
        counts["D23"],
    )
    return score, plan


def improve_soft(games, assignment, soft_targets, max_iter=1000, n_starts=8):
    option_to_set = {name: option_set(name) for name in ["S1", "S2", "S3", "D12", "D13", "D23"]}

    def objective(local_assignment):
        rank_sets = [option_to_set[opt] for opt in local_assignment]
        expected = sum(sum(games[i]["adj_rank_prob"][r] for r in rank_sets[i]) for i in range(len(games)))
        soft_penalty, detail = evaluate_soft_penalty(games, rank_sets, soft_targets)
        total = expected - 0.12 * soft_penalty
        return total, expected, soft_penalty, detail

    def local_search(seed_assignment):
        best = seed_assignment[:]
        best_obj, best_expected, best_penalty, best_detail = objective(best)
        n = len(best)
        for _ in range(max_iter):
            improved = False
            for i, j in itertools.combinations(range(n), 2):
                if best[i] == best[j]:
                    continue
                candidate = best[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                cand_obj, cand_expected, cand_penalty, cand_detail = objective(candidate)
                if cand_obj > best_obj + 1e-12:
                    best = candidate
                    best_obj, best_expected, best_penalty, best_detail = (
                        cand_obj,
                        cand_expected,
                        cand_penalty,
                        cand_detail,
                    )
                    improved = True
                    break
            if not improved:
                break
        return best, best_expected, best_penalty, best_detail, best_obj

    best_global = None
    for start in range(n_starts):
        seed = assignment[:]
        if start > 0:
            for _ in range(start * 3):
                i, j = random.sample(range(len(seed)), 2)
                seed[i], seed[j] = seed[j], seed[i]
        candidate = local_search(seed)
        if best_global is None or candidate[4] > best_global[4]:
            best_global = candidate

    return best_global


def ranks_to_bet(rank_set, rank_map):
    symbols = {rank_map[r] for r in rank_set}
    if len(symbols) == 1:
        return next(iter(symbols))
    if len(symbols) == 2:
        if symbols == {"1", "X"}:
            return "1X"
        if symbols == {"1", "2"}:
            return "12"
        if symbols == {"X", "2"}:
            return "X2"
    return "1X2"


def write_predictions(path, games, assignment):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Concurso",
        "Jogo",
        "Mandante",
        "Visitante",
        "Palpite",
        "Tipo",
        "p_escolha",
    ]
    option_to_set = {name: option_set(name) for name in ["S1", "S2", "S3", "D12", "D13", "D23"]}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        for game, opt in zip(games, assignment):
            rank_set = option_to_set[opt]
            palpite = ranks_to_bet(rank_set, game["rank_map"])
            p_escolha = sum(game["adj_rank_prob"][r] for r in rank_set)
            writer.writerow(
                {
                    "Concurso": game["Concurso"],
                    "Jogo": game["Jogo"],
                    "Mandante": game["Mandante"],
                    "Visitante": game["Visitante"],
                    "Palpite": palpite,
                    "Tipo": "seco" if len(rank_set) == 1 else ("duplo" if len(rank_set) == 2 else "triplo"),
                    "p_escolha": f"{p_escolha:.6f}".replace(".", ","),
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/proximo_concurso.csv")
    parser.add_argument("--model", default="models/model.json")
    parser.add_argument("--output", default="output/predictions.csv")
    parser.add_argument("--debug-output", default="output/debug_report.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    random.seed(42)
    games = load_csv(args.input)
    model = load_json(args.model)

    for game in games:
        game["rank_map"] = rank_symbol_map(game)
        inv_rank = {v: k for k, v in game["rank_map"].items()}
        raw = {"1": game["p(1)"], "X": game["p(x)"], "2": game["p(2)"]}
        max_prob = max(raw.values())
        adj = {symbol: adjust_prob(model, symbol, raw[symbol], max_prob) for symbol in raw}
        game["adj_rank_prob"] = {rank: adj[symbol] for symbol, rank in inv_rank.items()}

    distributions = solve_count_system()
    LOGGER.info("Distribuições viáveis encontradas: %s", len(distributions))

    best = None
    for dist in distributions:
        base_score, base_assignment = assign_best_for_distribution(games, dist)
        improved_assignment, expected, soft_penalty, soft_detail, objective = improve_soft(
            games,
            base_assignment,
            model["soft_targets"],
        )
        candidate = {
            "distribution": dist,
            "base_score": base_score,
            "expected_score": expected,
            "soft_penalty": soft_penalty,
            "objective": objective,
            "assignment": improved_assignment,
            "soft_detail": soft_detail,
        }
        if best is None or candidate["objective"] > best["objective"]:
            best = candidate

    write_predictions(args.output, games, best["assignment"])

    assignment_counts = {"secos": 0, "duplos": 0, "triplos": 0, "top1": 0, "top2": 0, "top3": 0}
    option_to_set = {name: option_set(name) for name in ["S1", "S2", "S3", "D12", "D13", "D23"]}
    for opt in best["assignment"]:
        rs = option_to_set[opt]
        if len(rs) == 1:
            assignment_counts["secos"] += 1
        elif len(rs) == 2:
            assignment_counts["duplos"] += 1
        else:
            assignment_counts["triplos"] += 1
        for rank in rs:
            assignment_counts[f"top{rank}"] += 1

    debug_payload = {
        "selected_distribution": best["distribution"],
        "objective": best["objective"],
        "expected_score": best["expected_score"],
        "soft_penalty": best["soft_penalty"],
        "soft_detail": best["soft_detail"],
        "hard_constraints_check": assignment_counts,
        "calibration_diagnostics": model.get("calibration_diagnostics", {}),
    }
    dump_json(args.debug_output, debug_payload)
    LOGGER.info("Predições salvas em %s", args.output)
    LOGGER.info("Debug salvo em %s", args.debug_output)


if __name__ == "__main__":
    main()
