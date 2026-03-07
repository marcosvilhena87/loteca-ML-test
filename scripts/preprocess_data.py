import argparse
import logging
from collections import defaultdict
from statistics import mean

from scripts.common import dump_json, group_by_concurso, load_csv, run_stats, setup_logging

LOGGER = logging.getLogger(__name__)


def build_soft_targets(historical_games):
    grouped = group_by_concurso(historical_games)
    metrics_per_rank = defaultdict(list)

    for concurso, games in grouped.items():
        for rank in (1, 2, 3):
            prob_col = f"p(top{rank})"
            indicator_col = f"top{rank}"
            ordered = sorted(games, key=lambda r: -r[prob_col])
            binary = [int(g[indicator_col]) for g in ordered]
            stats = run_stats(binary)
            metrics_per_rank[rank].append(stats)
        LOGGER.debug("Concurso %s processado para métricas de runs", concurso)

    soft_targets = {}
    for rank in (1, 2, 3):
        all_stats = metrics_per_rank[rank]
        soft_targets[f"top{rank}"] = {
            "avg_run_length": mean(s["avg_run_length"] for s in all_stats),
            "runs_count": mean(s["runs_count"] for s in all_stats),
        }
    return soft_targets


def build_hit_rates(historical_games):
    total = len(historical_games)
    hit_rates = {
        "1": sum(g["1"] for g in historical_games) / total,
        "X": sum(g["X"] for g in historical_games) / total,
        "2": sum(g["2"] for g in historical_games) / total,
    }
    return hit_rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/concursos_anteriores.csv")
    parser.add_argument("--output", default="output/preprocessed_stats.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    historical = load_csv(args.input)

    payload = {
        "soft_targets": build_soft_targets(historical),
        "base_hit_rates": build_hit_rates(historical),
        "total_rows": len(historical),
    }
    dump_json(args.output, payload)
    LOGGER.info("Preprocessamento concluído: %s", args.output)


if __name__ == "__main__":
    main()
