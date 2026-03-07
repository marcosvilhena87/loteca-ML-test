import argparse
import logging
from collections import defaultdict
from statistics import mean, pstdev

from scripts.common import dump_json, group_by_concurso, load_csv, rank_structure_stats, setup_logging

LOGGER = logging.getLogger(__name__)


BASE_PENALTY_WEIGHTS = {
    "top1": {
        "avg_run_length": 1.25,
        "runs_count": 1.15,
        "avg_position": 1.10,
        "run_share_len1": 0.60,
        "run_share_len2": 0.60,
        "run_share_len3p": 0.50,
        "first_occurrence": 0.55,
        "last_occurrence": 0.55,
        "avg_gap": 0.50,
    },
    "top2": {
        "avg_run_length": 0.95,
        "runs_count": 1.00,
        "avg_position": 1.00,
        "run_share_len1": 0.70,
        "run_share_len2": 0.70,
        "run_share_len3p": 0.60,
        "first_occurrence": 0.65,
        "last_occurrence": 0.65,
        "avg_gap": 0.60,
    },
    "top3": {
        "avg_run_length": 0.70,
        "runs_count": 0.85,
        "avg_position": 1.35,
        "run_share_len1": 0.80,
        "run_share_len2": 0.70,
        "run_share_len3p": 0.55,
        "first_occurrence": 1.00,
        "last_occurrence": 1.00,
        "avg_gap": 0.95,
    },
}


def _mean_and_std(stats_list):
    metrics = stats_list[0].keys()
    out = {}
    for metric in metrics:
        vals = [s[metric] for s in stats_list]
        out[metric] = {
            "mean": float(mean(vals)),
            "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
        }
    return out


def _build_rank_penalty_weights(summary):
    weights = {}
    for rank in (1, 2, 3):
        label = f"top{rank}"
        weights[label] = {}
        for metric, stats in summary[label].items():
            stability_scale = 1.0 / (stats["std"] + 0.15)
            base = BASE_PENALTY_WEIGHTS[label].get(metric, 0.5)
            weights[label][metric] = float(base * stability_scale)
    return weights


def build_soft_targets(historical_games):
    grouped = group_by_concurso(historical_games)
    metrics_per_rank = defaultdict(list)

    for concurso, games in grouped.items():
        for rank in (1, 2, 3):
            prob_col = f"p(top{rank})"
            indicator_col = f"top{rank}"
            ordered = sorted(games, key=lambda r: -r[prob_col])
            binary = [int(g[indicator_col]) for g in ordered]
            stats = rank_structure_stats(binary)
            metrics_per_rank[rank].append(stats)
        LOGGER.debug("Concurso %s processado para métricas estruturais", concurso)

    rank_summary = {}
    soft_targets = {}
    for rank in (1, 2, 3):
        label = f"top{rank}"
        rank_summary[label] = _mean_and_std(metrics_per_rank[rank])
        soft_targets[label] = {metric: data["mean"] for metric, data in rank_summary[label].items()}

    return {
        "targets": soft_targets,
        "metric_summary": rank_summary,
        "metric_weights": _build_rank_penalty_weights(rank_summary),
    }


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
