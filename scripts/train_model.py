import argparse
import json
import logging
from statistics import mean

from scripts.common import detect_hit_rank, enrich_probabilities, read_csv_semicolon
from scripts.preprocess_data import preprocess


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

    model = {
        "version": "1.0",
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
            "soft_penalty_weight": 0.35,
            "local_search_iterations": 8000,
            "random_seed": 42,
        },
    }

    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    logging.info("Modelo salvo em %s", model_path)
    logging.info("Distribuição histórica por rank: %s", rank_distribution)
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
