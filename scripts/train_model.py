import argparse
import json
import logging
from collections import Counter, defaultdict
from statistics import mean
from typing import Dict, List

from scripts.common import detect_hit_rank, enrich_probabilities, read_csv_semicolon
from scripts.predict_results import best_assignment_for_config, feasible_type_configs, local_search
from scripts.preprocess_data import preprocess

BACKTEST_WINDOWS = [20, 50, 100]
WALK_FORWARD_TRAIN_WINDOW = 50
WALK_FORWARD_STEPS = 20


def resolve_result_symbol(row: Dict[str, str]) -> str:
    if row.get("1", "0") == "1":
        return "1"
    if row.get("X", "0") == "1":
        return "X"
    if row.get("2", "0") == "1":
        return "2"
    return ""


def option_symbols(option: str, top_symbols: Dict[str, str]) -> List[str]:
    if option == "1":
        return [top_symbols["top1"]]
    if option == "2":
        return [top_symbols["top2"]]
    if option == "3":
        return [top_symbols["top3"]]
    if option == "12":
        return [top_symbols["top1"], top_symbols["top2"]]
    if option == "13":
        return [top_symbols["top1"], top_symbols["top3"]]
    if option == "23":
        return [top_symbols["top2"], top_symbols["top3"]]
    return ["1", "X", "2"]


def evaluate_concurso(
    jogos_ordenados: List[Dict[str, object]],
    targets: Dict[str, Dict[str, float]],
    penalty_weight: float,
    iterations: int,
    seed: int,
) -> int:
    configs = feasible_type_configs()
    best_global = None
    best_obj = None

    for config in configs:
        if sum(config.values()) != len(jogos_ordenados):
            continue

        _, assignment = best_assignment_for_config(jogos_ordenados, config)
        if not assignment:
            continue

        improved, debug = local_search(
            jogos_ordenados,
            assignment,
            targets,
            penalty_weight=penalty_weight,
            iterations=iterations,
            seed=seed,
        )

        if best_obj is None or debug["objective"] > best_obj:
            best_obj = debug["objective"]
            best_global = improved

    if best_global is None:
        return -1

    hits = 0
    for i, jogo in enumerate(jogos_ordenados):
        result_symbol = resolve_result_symbol(jogo)
        if not result_symbol:
            continue
        if result_symbol in option_symbols(best_global[i], jogo["_top_symbols"]):
            hits += 1
    return hits


def summarize_hits(hits_per_concurso: List[int], weight: float, scope: str) -> Dict[str, float]:
    hit_distribution = Counter(hits_per_concurso)
    n = len(hits_per_concurso)
    return {
        "scope": scope,
        "weight": weight,
        "avg_hits": mean(hits_per_concurso),
        "avg_coverage": mean(h / 14 for h in hits_per_concurso),
        "pct_11": hit_distribution.get(11, 0) / n,
        "pct_12": hit_distribution.get(12, 0) / n,
        "pct_13": hit_distribution.get(13, 0) / n,
        "pct_14": hit_distribution.get(14, 0) / n,
        "n_concursos": n,
    }


def backtest_penalty_weights(
    rows: List[Dict[str, str]],
    targets: Dict[str, Dict[str, float]],
    penalty_weights: List[float],
    iterations: int,
    seed: int,
    max_concursos: int,
) -> List[Dict[str, float]]:
    by_concurso: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_concurso[row["Concurso"]].append(enrich_probabilities(row))

    results = []
    concursos_ids = sorted(by_concurso.keys(), key=int)
    if max_concursos > 0:
        concursos_ids = concursos_ids[-max_concursos:]

    for penalty_weight in penalty_weights:
        hits_per_concurso = []
        for concurso_id in concursos_ids:
            jogos_ordenados = sorted(by_concurso[concurso_id], key=lambda x: int(x["Jogo"]))
            hits = evaluate_concurso(
                jogos_ordenados,
                targets,
                penalty_weight,
                iterations=iterations,
                seed=seed + int(concurso_id),
            )
            if hits >= 0:
                hits_per_concurso.append(hits)

        if hits_per_concurso:
            results.append(summarize_hits(hits_per_concurso, penalty_weight, f"last_{max_concursos}"))

    return sorted(results, key=lambda x: x["avg_hits"], reverse=True)


def walk_forward_backtest(
    rows: List[Dict[str, str]],
    targets: Dict[str, Dict[str, float]],
    penalty_weights: List[float],
    iterations: int,
    seed: int,
    train_window: int,
    max_steps: int,
) -> List[Dict[str, float]]:
    by_concurso: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_concurso[row["Concurso"]].append(enrich_probabilities(row))

    concursos_ids = sorted(by_concurso.keys(), key=int)
    if len(concursos_ids) <= train_window:
        return []

    eval_ids = concursos_ids[train_window:]
    if max_steps > 0:
        eval_ids = eval_ids[-max_steps:]

    results = []
    for penalty_weight in penalty_weights:
        hits_per_concurso = []
        for concurso_id in eval_ids:
            jogos_ordenados = sorted(by_concurso[concurso_id], key=lambda x: int(x["Jogo"]))
            hits = evaluate_concurso(
                jogos_ordenados,
                targets,
                penalty_weight,
                iterations=iterations,
                seed=seed + int(concurso_id),
            )
            if hits >= 0:
                hits_per_concurso.append(hits)
        if hits_per_concurso:
            results.append(summarize_hits(hits_per_concurso, penalty_weight, "walk_forward"))

    return sorted(results, key=lambda x: x["avg_hits"], reverse=True)


def select_best_weight(results: List[Dict[str, float]], default_weight: float) -> float:
    if not results:
        return default_weight

    per_weight: Dict[float, List[float]] = defaultdict(list)
    for entry in results:
        per_weight[entry["weight"]].append(entry["avg_hits"])

    aggregate = [
        {
            "weight": weight,
            "avg_hits": mean(scores),
            "n_evaluations": len(scores),
        }
        for weight, scores in per_weight.items()
    ]
    aggregate.sort(key=lambda x: x["avg_hits"], reverse=True)
    return aggregate[0]["weight"]


def train_model(history_path: str, model_path: str, preprocessed_path: str) -> dict:
    prep_summary = preprocess(history_path, preprocessed_path)
    rows = read_csv_semicolon(history_path)

    hit_ranks = []
    margin_top12 = []
    margin_top23 = []

    for row in rows:
        enriched = enrich_probabilities(row)
        hit_ranks.append(detect_hit_rank(row, enriched))
        margin_top12.append(enriched["_top_probs"]["top1"] - enriched["_top_probs"]["top2"])
        margin_top23.append(enriched["_top_probs"]["top2"] - enriched["_top_probs"]["top3"])

    total = len(hit_ranks) or 1
    rank_distribution = {
        "top1": sum(1 for x in hit_ranks if x == 1) / total,
        "top2": sum(1 for x in hit_ranks if x == 2) / total,
        "top3": sum(1 for x in hit_ranks if x == 3) / total,
    }

    candidate_weights = [0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
    backtest_by_window = {
        f"last_{window}": backtest_penalty_weights(
            rows,
            prep_summary["target_metrics"],
            penalty_weights=candidate_weights,
            iterations=20,
            seed=42,
            max_concursos=window,
        )
        for window in BACKTEST_WINDOWS
    }
    walk_forward = walk_forward_backtest(
        rows,
        prep_summary["target_metrics"],
        penalty_weights=candidate_weights,
        iterations=20,
        seed=42,
        train_window=WALK_FORWARD_TRAIN_WINDOW,
        max_steps=WALK_FORWARD_STEPS,
    )

    all_backtests = []
    for window_results in backtest_by_window.values():
        all_backtests.extend(window_results)
    all_backtests.extend(walk_forward)

    best_weight = select_best_weight(all_backtests, default_weight=0.35)

    model = {
        "version": "1.2",
        "hard_constraints": {
            "top1": 8,
            "top2": 5,
            "top3": 4,
            "secos": 11,
            "duplos": 3,
            "triplos": 0,
        },
        "soft_targets": prep_summary["target_metrics"],
        "historical_stats": {
            "rank_distribution": rank_distribution,
            "avg_margin_top12": mean(margin_top12) if margin_top12 else 0.0,
            "avg_margin_top23": mean(margin_top23) if margin_top23 else 0.0,
            "n_rows": len(rows),
            "n_concursos": prep_summary["n_concursos"],
        },
        "search": {
            "soft_penalty_weight": best_weight,
            "local_search_iterations": 8000,
            "random_seed": 42,
            "backtest_windows": BACKTEST_WINDOWS,
            "walk_forward": {
                "train_window": WALK_FORWARD_TRAIN_WINDOW,
                "steps": WALK_FORWARD_STEPS,
            },
            "weight_backtest_by_window": backtest_by_window,
            "weight_backtest_walk_forward": walk_forward,
            "weight_backtest_aggregate": all_backtests,
        },
    }

    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    logging.info("Modelo salvo em %s", model_path)
    logging.info("Distribuição histórica por rank: %s", rank_distribution)
    logging.info("Backtest por janela: %s", backtest_by_window)
    logging.info("Backtest walk-forward: %s", walk_forward)
    logging.info("Peso selecionado para busca local: %.2f", best_weight)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="data/concursos_anteriores.csv")
    parser.add_argument("--model", default="models/model.json")
    parser.add_argument("--preprocessed", default="output/preprocessed_history.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    train_model(args.history, args.model, args.preprocessed)


if __name__ == "__main__":
    main()
