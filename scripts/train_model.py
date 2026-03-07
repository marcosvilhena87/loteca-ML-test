import argparse
import logging

from scripts.common import dump_json, load_csv, load_json, setup_logging

LOGGER = logging.getLogger(__name__)


def bucket_id(value, n_buckets=10):
    idx = int(value * n_buckets)
    if idx >= n_buckets:
        idx = n_buckets - 1
    return idx


def calibrate(rows, prob_col, target_col, n_buckets=10):
    buckets = {i: {"count": 0, "hits": 0} for i in range(n_buckets)}
    for row in rows:
        b = bucket_id(row[prob_col], n_buckets=n_buckets)
        buckets[b]["count"] += 1
        buckets[b]["hits"] += int(row[target_col])
    calibration = {}
    for b, info in buckets.items():
        if info["count"]:
            calibration[str(b)] = info["hits"] / info["count"]
        else:
            calibration[str(b)] = 0.0
    return calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/concursos_anteriores.csv")
    parser.add_argument("--preprocessed", default="output/preprocessed_stats.json")
    parser.add_argument("--output", default="models/model.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    historical = load_csv(args.input)
    preprocessed = load_json(args.preprocessed)

    model = {
        "meta": {
            "n_buckets": 10,
            "calibration_weight": 0.35,
            "raw_weight": 0.65,
        },
        "soft_targets": preprocessed["soft_targets"],
        "base_hit_rates": preprocessed["base_hit_rates"],
        "calibration": {
            "1": calibrate(historical, "p(1)", "1"),
            "X": calibrate(historical, "p(x)", "X"),
            "2": calibrate(historical, "p(2)", "2"),
        },
    }
    dump_json(args.output, model)
    LOGGER.info("Treino concluído: %s", args.output)


if __name__ == "__main__":
    main()
