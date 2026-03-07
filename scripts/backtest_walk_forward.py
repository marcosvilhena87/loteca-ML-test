import argparse
import logging
from statistics import mean

from scripts.common import dump_json, group_by_concurso, load_csv, load_json, rank_symbols, setup_logging
from scripts.predict_results import (
    adjust_prob,
    assign_best_for_distribution,
    improve_soft,
    normalized_entropy,
    option_set,
    solve_count_system,
)
from scripts.preprocess_data import build_hit_rates, build_soft_targets
from scripts.train_model import BANDS, build_calibration

LOGGER = logging.getLogger(__name__)


def concurso_sort_key(value):
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def actual_symbol(row):
    for s in ("1", "X", "2"):
        if int(row[s]) == 1:
            return s
    return "X"


def rank_symbol_map(row):
    ranked = rank_symbols(row)
    return {1: ranked[0], 2: ranked[1], 3: ranked[2]}


def prepare_games(rows, model):
    games = []
    for row in rows:
        game = dict(row)
        game["rank_map"] = rank_symbol_map(game)
        inv_rank = {v: k for k, v in game["rank_map"].items()}
        raw = {"1": game["p(1)"], "X": game["p(x)"], "2": game["p(2)"]}
        max_prob = max(raw.values())
        adj = {symbol: adjust_prob(model, symbol, raw[symbol], max_prob) for symbol in raw}
        game["adj_rank_prob"] = {rank: adj[symbol] for symbol, rank in inv_rank.items()}
        ranked_adj = sorted(adj.values(), reverse=True)
        gap12 = max(0.0, ranked_adj[0] - ranked_adj[1])
        entropy = normalized_entropy(list(adj.values()))
        uncertainty = 0.5 * ((1.0 - min(1.0, gap12)) + entropy)
        game["uncertainty"] = {
            "gap_top1_top2": gap12,
            "entropy": entropy,
            "certainty": 1.0 - uncertainty,
            "top3_position": 0,
        }
        games.append(game)

    top3_sorted = sorted(range(len(games)), key=lambda i: -games[i]["p(top3)"])
    for pos, idx in enumerate(top3_sorted, start=1):
        games[idx]["uncertainty"]["top3_position"] = pos
    return games


def infer_assignment(games, model_meta, soft_targets):
    distributions = solve_count_system()
    best = None
    for dist in distributions:
        base_score, base_assignment = assign_best_for_distribution(
            games,
            dist,
            model_meta.get("uncertainty_heuristics", {}),
        )
        improved = improve_soft(games, base_assignment, soft_targets)
        candidate = {
            "distribution": dist,
            "base_score": base_score,
            "objective": improved["objective"],
            "assignment": improved["assignment"],
            "expected": improved["expected"],
            "soft_penalty": improved["soft_penalty"],
        }
        if best is None or candidate["objective"] > best["objective"]:
            best = candidate
    return best


def count_hits(games, assignment):
    hits = 0
    for game, opt in zip(games, assignment):
        selected_ranks = option_set(opt)
        selected_symbols = {game["rank_map"][r] for r in selected_ranks}
        if actual_symbol(game) in selected_symbols:
            hits += 1
    return hits


def baseline_top1_hits(games):
    hits = 0
    for game in games:
        if actual_symbol(game) == game["rank_map"][1]:
            hits += 1
    return hits


def run_backtest(rows, min_train_contests, base_model_meta):
    grouped = group_by_concurso(rows)
    concursos = sorted(grouped.keys(), key=concurso_sort_key)
    if len(concursos) <= min_train_contests:
        raise ValueError("Não há concursos suficientes para walk-forward com min_train_contests atual")

    results = []
    for idx in range(min_train_contests, len(concursos)):
        train_concursos = concursos[:idx]
        test_concurso = concursos[idx]
        train_rows = [r for c in train_concursos for r in grouped[c]]
        test_rows = grouped[test_concurso]

        preprocessed = {
            "soft_targets": build_soft_targets(train_rows),
            "base_hit_rates": build_hit_rates(train_rows),
        }
        calibration, calib_diag = build_calibration(train_rows)
        model_meta = {
            "raw_weight": base_model_meta.get("raw_weight", 0.60),
            "calibration_weight": base_model_meta.get("calibration_weight", 0.40),
            "confidence_bands": base_model_meta.get("confidence_bands", BANDS),
            "uncertainty_heuristics": base_model_meta.get(
                "uncertainty_heuristics",
                {
                    "enabled": True,
                    "top3_early_penalty": 0.09,
                    "top3_early_window": 6,
                    "top1_certainty_bonus": 0.04,
                },
            ),
        }
        fold_model = {
            "meta": model_meta,
            "calibration": calibration,
        }

        games = prepare_games(test_rows, fold_model)
        current = infer_assignment(games, model_meta, preprocessed["soft_targets"])

        legacy_meta = dict(model_meta)
        legacy_meta["uncertainty_heuristics"] = {
            "enabled": False,
            "top3_early_penalty": 0.0,
            "top3_early_window": 6,
            "top1_certainty_bonus": 0.0,
        }
        previous = infer_assignment(games, legacy_meta, preprocessed["soft_targets"])

        hits_current = count_hits(games, current["assignment"])
        hits_previous = count_hits(games, previous["assignment"])
        hits_top1 = baseline_top1_hits(games)

        results.append(
            {
                "test_concurso": test_concurso,
                "train_concursos": len(train_concursos),
                "hits_current": hits_current,
                "hits_previous": hits_previous,
                "hits_top1_baseline": hits_top1,
                "objective_current": current["objective"],
                "expected_current": current["expected"],
                "soft_penalty_current": current["soft_penalty"],
                "objective_previous": previous["objective"],
                "expected_previous": previous["expected"],
                "soft_penalty_previous": previous["soft_penalty"],
                "delta_vs_previous": hits_current - hits_previous,
                "delta_vs_baseline": hits_current - hits_top1,
                "delta_objective_vs_previous": current["objective"] - previous["objective"],
                "delta_expected_vs_previous": current["expected"] - previous["expected"],
                "calibration_diagnostics": calib_diag,
            }
        )
        LOGGER.info(
            "Concurso %s | atual=%s anterior=%s baseline_top1=%s",
            test_concurso,
            hits_current,
            hits_previous,
            hits_top1,
        )

    summary = {
        "folds": len(results),
        "mean_hits_current": mean([r["hits_current"] for r in results]),
        "mean_hits_previous": mean([r["hits_previous"] for r in results]),
        "mean_hits_top1_baseline": mean([r["hits_top1_baseline"] for r in results]),
        "hit_distribution_current": {str(k): sum(1 for r in results if r["hits_current"] == k) for k in range(15)},
        "hit_distribution_previous": {str(k): sum(1 for r in results if r["hits_previous"] == k) for k in range(15)},
        "hit_distribution_top1_baseline": {
            str(k): sum(1 for r in results if r["hits_top1_baseline"] == k) for k in range(15)
        },
        "mean_delta_vs_previous": mean([r["delta_vs_previous"] for r in results]),
        "mean_delta_vs_baseline": mean([r["delta_vs_baseline"] for r in results]),
        "mean_delta_objective_vs_previous": mean([r["delta_objective_vs_previous"] for r in results]),
        "mean_delta_expected_vs_previous": mean([r["delta_expected_vs_previous"] for r in results]),
        "folds_with_hit_improvement": sum(1 for r in results if r["delta_vs_previous"] > 0),
        "folds_with_objective_improvement": sum(1 for r in results if r["delta_objective_vs_previous"] > 0),
    }

    return {"summary": summary, "fold_results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/concursos_anteriores.csv")
    parser.add_argument("--output", default="output/backtest_walk_forward.json")
    parser.add_argument("--model", default="models/model.json")
    parser.add_argument("--min-train-concursos", type=int, default=20)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    rows = load_csv(args.input)
    model = load_json(args.model)
    report = run_backtest(rows, args.min_train_concursos, model.get("meta", {}))
    dump_json(args.output, report)
    LOGGER.info("Backtest walk-forward salvo em %s", args.output)


if __name__ == "__main__":
    main()
