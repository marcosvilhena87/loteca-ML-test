import argparse
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger("train_model")
OUTCOMES = ("1", "X", "2")
RANK_BUCKETS = ((1, 3), (4, 6), (7, 10), (11, 14))


def load_processed(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_rank_in_concurso(rows):
    by_concurso = defaultdict(list)
    for r in rows:
        by_concurso[r["concurso"]].append(r)

    for jogos in by_concurso.values():
        jogos.sort(key=lambda x: x["probs"][x["top_order"][0]], reverse=True)
        for idx, jogo in enumerate(jogos, start=1):
            jogo["rank_r"] = idx


def percentile(sorted_values, q):
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(sorted_values[low])
    frac = pos - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def rank_bucket_label(rank):
    for idx, (start, end) in enumerate(RANK_BUCKETS, start=1):
        if start <= rank <= end:
            return f"B{idx}"
    return "B4"


def gap_class_label(row, thresholds):
    top1 = row["top_order"][0]
    top2 = row["top_order"][1]
    gap = row["probs"][top1] - row["probs"][top2]
    if gap <= thresholds[0]:
        return "LOW"
    if gap <= thresholds[1]:
        return "MID"
    return "HIGH"


def run_metrics(binary_seq):
    if not binary_seq:
        return {"run_count": 0, "max_run": 0}

    run_count = 1
    max_run = 1
    curr = 1
    for i in range(1, len(binary_seq)):
        if binary_seq[i] == binary_seq[i - 1]:
            curr += 1
            max_run = max(max_run, curr)
        else:
            run_count += 1
            curr = 1
    return {"run_count": run_count, "max_run": max_run}


def train(rows, alpha=1.0):
    compute_rank_in_concurso(rows)

    gaps = []
    for r in rows:
        if not r["outcome"]:
            continue
        top1 = r["top_order"][0]
        top2 = r["top_order"][1]
        gaps.append(r["probs"][top1] - r["probs"][top2])

    gaps_sorted = sorted(gaps)
    gap_thresholds = [percentile(gaps_sorted, 0.33), percentile(gaps_sorted, 0.66)]

    rank_topk_hits = defaultdict(lambda: Counter())
    rank_totals = Counter()
    rank_class_outcome = defaultdict(lambda: Counter())
    global_topk_hits = Counter()

    macro_k_counts = Counter()
    micro_k_counts = defaultdict(lambda: Counter())
    micro_totals = Counter()

    by_concurso = defaultdict(list)

    for r in rows:
        if not r["outcome"]:
            continue
        rank = r["rank_r"]
        rank_totals[rank] += 1
        rank_class_outcome[rank][r["outcome"]] += 1
        by_concurso[r["concurso"]].append(r)

        realized_k = None
        for k in (1, 2, 3):
            predicted = r["top_order"][k - 1]
            if r["outcome"] == predicted:
                rank_topk_hits[rank][k] += 1
                global_topk_hits[k] += 1
                realized_k = k

        if realized_k is None:
            continue

        macro_k_counts[realized_k] += 1
        bucket = rank_bucket_label(rank)
        gap_class = gap_class_label(r, gap_thresholds)
        group = f"{bucket}|{gap_class}"
        micro_k_counts[group][realized_k] += 1
        micro_totals[group] += 1

    total_obs = sum(rank_totals.values())
    topk_cond_rank = {}
    outcome_cond_rank = {}
    for rank, n in rank_totals.items():
        topk_cond_rank[str(rank)] = {
            str(k): (rank_topk_hits[rank][k] + alpha) / (n + 3 * alpha) for k in (1, 2, 3)
        }
        outcome_cond_rank[str(rank)] = {
            c: (rank_class_outcome[rank][c] + alpha) / (n + 3 * alpha) for c in OUTCOMES
        }

    global_topk = {
        str(k): (global_topk_hits[k] + alpha) / (total_obs + 3 * alpha) for k in (1, 2, 3)
    }

    macro_total = sum(macro_k_counts.values())
    g_hist = {
        str(k): (macro_k_counts[k] + alpha) / (macro_total + 3 * alpha) for k in (1, 2, 3)
    }

    micro_targets = {}
    for group, n in micro_totals.items():
        micro_targets[group] = {str(k): (micro_k_counts[group][k] + alpha) / (n + 3 * alpha) for k in (1, 2, 3)}

    run_counts = []
    max_runs = []
    for jogos in by_concurso.values():
        jogos.sort(key=lambda x: x["rank_r"])
        top1_hit_seq = [1 if j["outcome"] == j["top_order"][0] else 0 for j in jogos]
        metrics = run_metrics(top1_hit_seq)
        run_counts.append(metrics["run_count"])
        max_runs.append(metrics["max_run"])

    run_counts.sort()
    max_runs.sort()
    run_stats = {
        "run_count_min": percentile(run_counts, 0.1),
        "run_count_max": percentile(run_counts, 0.9),
        "max_run_max": percentile(max_runs, 0.9),
    }

    model = {
        "alpha": alpha,
        "total_obs": total_obs,
        "rank_buckets": [list(b) for b in RANK_BUCKETS],
        "gap_thresholds": gap_thresholds,
        "topk_cond_rank": topk_cond_rank,
        "outcome_cond_rank": outcome_cond_rank,
        "global_topk": global_topk,
        "g_hist": g_hist,
        "micro_targets": micro_targets,
        "run_stats": run_stats,
    }
    return model


def save_model(model, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino do modelo Loteca")
    parser.add_argument("--input", default="data/concursos_anteriores.processed.json")
    parser.add_argument("--output", default="models/model.json")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    rows = load_processed(Path(args.input))
    model = train(rows, alpha=args.alpha)
    save_model(model, Path(args.output))

    logger.info("Modelo treinado com %s observações.", model["total_obs"])
    logger.debug("Distribuição global topk: %s", model["global_topk"])
    logger.info("Modelo salvo em %s", args.output)
