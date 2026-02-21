import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger("train_model")


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


def train(rows, alpha=1.0):
    compute_rank_in_concurso(rows)

    rank_topk_hits = defaultdict(lambda: Counter())
    rank_totals = Counter()
    rank_class_outcome = defaultdict(lambda: Counter())
    global_topk_hits = Counter()

    for r in rows:
        if not r["outcome"]:
            continue
        rank = r["rank_r"]
        rank_totals[rank] += 1
        rank_class_outcome[rank][r["outcome"]] += 1

        for k in (1, 2, 3):
            predicted = r["top_order"][k - 1]
            if r["outcome"] == predicted:
                rank_topk_hits[rank][k] += 1
                global_topk_hits[k] += 1

    total_obs = sum(rank_totals.values())
    topk_cond_rank = {}
    outcome_cond_rank = {}
    for rank, n in rank_totals.items():
        topk_cond_rank[str(rank)] = {
            str(k): (rank_topk_hits[rank][k] + alpha) / (n + 3 * alpha) for k in (1, 2, 3)
        }
        outcome_cond_rank[str(rank)] = {
            c: (rank_class_outcome[rank][c] + alpha) / (n + 3 * alpha) for c in ("1", "X", "2")
        }

    global_topk = {
        str(k): (global_topk_hits[k] + alpha) / (total_obs + 3 * alpha) for k in (1, 2, 3)
    }

    model = {
        "alpha": alpha,
        "total_obs": total_obs,
        "topk_cond_rank": topk_cond_rank,
        "outcome_cond_rank": outcome_cond_rank,
        "global_topk": global_topk,
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
