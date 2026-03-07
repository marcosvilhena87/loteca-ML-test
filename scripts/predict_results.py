import argparse
import csv
import itertools
import logging
from functools import lru_cache
from pathlib import Path

from scripts.common import (
    avg_selected_position,
    dump_json,
    load_csv,
    load_json,
    rank_symbols,
    run_stats,
    setup_logging,
)

LOGGER = logging.getLogger(__name__)


def rank_symbol_map(row):
    ranked = rank_symbols(row)
    return {1: ranked[0], 2: ranked[1], 3: ranked[2]}


def adjust_prob(model, symbol, raw_prob):
    n_buckets = model["meta"]["n_buckets"]
    bucket = min(int(raw_prob * n_buckets), n_buckets - 1)
    calibration = model["calibration"][symbol][str(bucket)]
    return model["meta"]["raw_weight"] * raw_prob + model["meta"]["calibration_weight"] * calibration


def evaluate_soft_penalty(games, selected_rank_sets, soft_targets):
    penalty = 0.0
    detail = {}
    for rank in (1, 2, 3):
        ordered_indices = sorted(range(len(games)), key=lambda i: -games[i][f"p(top{rank})"])
        indicator = [1 if rank in selected_rank_sets[i] else 0 for i in ordered_indices]
        stats = run_stats(indicator)
        stats["avg_position"] = avg_selected_position(indicator)
        target = soft_targets[f"top{rank}"]
        target_avg_position = target.get("avg_position", 0.0)
        rank_penalty = (stats["avg_run_length"] - target["avg_run_length"]) ** 2 + (
            stats["runs_count"] - target["runs_count"]
        ) ** 2 + (
            stats["avg_position"] - target_avg_position
        ) ** 2
        penalty += rank_penalty
        detail[f"top{rank}"] = {
            "actual": stats,
            "target": {**target, "avg_position": target_avg_position},
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


def improve_soft(games, assignment, soft_targets, max_iter=1500):
    option_to_set = {name: option_set(name) for name in ["S1", "S2", "S3", "D12", "D13", "D23"]}

    def objective(local_assignment):
        rank_sets = [option_to_set[opt] for opt in local_assignment]
        expected = sum(sum(games[i]["adj_rank_prob"][r] for r in rank_sets[i]) for i in range(len(games)))
        soft_penalty, detail = evaluate_soft_penalty(games, rank_sets, soft_targets)
        total = expected - 0.17 * soft_penalty
        return total, expected, soft_penalty, detail

    best = assignment[:]
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
    games = load_csv(args.input)
    model = load_json(args.model)

    for game in games:
        game["rank_map"] = rank_symbol_map(game)
        inv_rank = {v: k for k, v in game["rank_map"].items()}
        raw = {"1": game["p(1)"], "X": game["p(x)"], "2": game["p(2)"]}
        adj = {symbol: adjust_prob(model, symbol, raw[symbol]) for symbol in raw}
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
    }
    dump_json(args.debug_output, debug_payload)
    LOGGER.info("Predições salvas em %s", args.output)
    LOGGER.info("Debug salvo em %s", args.debug_output)


if __name__ == "__main__":
    main()
