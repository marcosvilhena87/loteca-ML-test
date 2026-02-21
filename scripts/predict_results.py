import argparse
import csv
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger("predict_results")
OUTCOMES = ("1", "X", "2")
DOUBLE_FMT = {("1", "X"): "1X", ("1", "2"): "12", ("X", "2"): "X2"}


def parse_decimal(value: str) -> float:
    return float(value.strip().replace(".", "").replace(",", "."))


def normalize_probs(odds):
    inv = {k: 1.0 / max(v, 1e-9) for k, v in odds.items()}
    s = sum(inv.values())
    return {k: inv[k] / s for k in OUTCOMES}


def load_next(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            odds = {"1": parse_decimal(row["Odds_1"]), "X": parse_decimal(row["Odds_X"]), "2": parse_decimal(row["Odds_2"])}
            probs = normalize_probs(odds)
            top_order = sorted(OUTCOMES, key=lambda c: probs[c], reverse=True)
            rows.append({
                "concurso": int(row["Concurso"]),
                "jogo": int(row["Jogo"]),
                "mandante": row["Mandante"],
                "visitante": row["Visitante"],
                "odds": odds,
                "base_probs": probs,
                "top_order": top_order,
            })
    rows.sort(key=lambda x: x["base_probs"][x["top_order"][0]], reverse=True)
    for idx, r in enumerate(rows, start=1):
        r["rank_r"] = idx
    return rows


def load_model(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def adjusted_probs(row, model, beta=0.7):
    rank = str(row["rank_r"])
    topk_rank = model["topk_cond_rank"].get(rank, model["global_topk"])
    gtopk = model["global_topk"]
    out_rank = model["outcome_cond_rank"].get(rank, {c: 1 / 3 for c in OUTCOMES})

    weights = {}
    for i, cls in enumerate(row["top_order"], start=1):
        rel = beta * float(topk_rank[str(i)]) + (1 - beta) * float(gtopk[str(i)])
        cls_ctrl = float(out_rank.get(cls, 1 / 3))
        weights[cls] = row["base_probs"][cls] * rel * (0.5 + cls_ctrl)

    s = sum(weights.values()) or 1.0
    return {k: weights[k] / s for k in OUTCOMES}


def sample_outcome(prob_map):
    r = random.random()
    acc = 0.0
    for c in OUTCOMES:
        acc += prob_map[c]
        if r <= acc:
            return c
    return OUTCOMES[-1]


def evaluate_ticket(games, picks, n_sim=2500):
    total_hits = 0
    for _ in range(n_sim):
        hits = 0
        for g in games:
            realized = sample_outcome(g["adj_probs"])
            if realized in picks[g["jogo"]]:
                hits += 1
        total_hits += hits
    return total_hits / n_sim


def concentration_penalty(games, double_games, lam_rank=0.08, lam_class=0.06):
    rank_buckets = Counter()
    cls_counter = Counter()

    for g in games:
        if g["jogo"] in double_games:
            quartile = (g["rank_r"] - 1) // 4
            rank_buckets[quartile] += 1
            cls_counter[g["top_order"][0]] += 1

    if not double_games:
        return 0.0

    max_rank_share = max(rank_buckets.values()) / len(double_games)
    max_cls_share = max(cls_counter.values()) / len(double_games)
    return lam_rank * max_rank_share + lam_class * max_cls_share


def build_picks(games, double_games, mode_map):
    picks = {}
    for g in games:
        t1, t2, t3 = g["top_order"]
        if g["jogo"] in double_games:
            mode = mode_map[g["jogo"]]
            if mode == 0:
                pick = tuple(sorted((t1, t2)))
            elif mode == 1:
                pick = tuple(sorted((t1, t3)))
            else:
                pick = tuple(sorted((t2, t3)))
            picks[g["jogo"]] = set(pick)
        else:
            picks[g["jogo"]] = {t1}
    return picks


def monte_carlo_optimize(games, iterations=4000):
    best = None
    # score de incerteza para escolher duplos
    uncertainty = {g["jogo"]: 1.0 - g["adj_probs"][g["top_order"][0]] for g in games}
    jogos = [g["jogo"] for g in games]

    for _ in range(iterations):
        weighted = sorted(jogos, key=lambda j: random.random() * uncertainty[j], reverse=True)
        double_games = set(weighted[:2])

        mode_map = {}
        for g in games:
            if g["jogo"] in double_games:
                p = g["adj_probs"]
                t1, t2, t3 = g["top_order"]
                options = [p[t1] + p[t2], p[t1] + p[t3], p[t2] + p[t3]]
                mode_map[g["jogo"]] = max(range(3), key=lambda i: options[i] * random.uniform(0.9, 1.1))

        picks = build_picks(games, double_games, mode_map)
        ev = evaluate_ticket(games, picks, n_sim=900)
        penalty = concentration_penalty(games, double_games)
        obj = ev - penalty

        if (best is None) or (obj > best["objective"]):
            best = {
                "objective": obj,
                "ev": ev,
                "penalty": penalty,
                "picks": picks,
                "double_games": sorted(double_games),
            }
    return best


def pick_to_text(pick_set):
    if len(pick_set) == 1:
        return next(iter(pick_set))
    key = tuple(sorted(pick_set))
    return DOUBLE_FMT[key]


def save_predictions(games, best, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Concurso", "Jogo", "Mandante", "Visitante", "Palpite", "Tipo",
        "Prob_Acerto", "Top1", "Top2", "Top3", "Rank_R"
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for g in sorted(games, key=lambda x: x["jogo"]):
            pick = best["picks"][g["jogo"]]
            writer.writerow({
                "Concurso": g["concurso"],
                "Jogo": g["jogo"],
                "Mandante": g["mandante"],
                "Visitante": g["visitante"],
                "Palpite": pick_to_text(pick),
                "Tipo": "DUPLO" if len(pick) == 2 else "SECO",
                "Prob_Acerto": f"{sum(g['adj_probs'][k] for k in pick):.4f}",
                "Top1": g["top_order"][0],
                "Top2": g["top_order"][1],
                "Top3": g["top_order"][2],
                "Rank_R": g["rank_r"],
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predição Loteca")
    parser.add_argument("--model", default="models/model.json")
    parser.add_argument("--input", default="data/proximo_concurso.csv")
    parser.add_argument("--output", default="output/predictions.csv")
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    random.seed(args.seed)

    model = load_model(Path(args.model))
    games = load_next(Path(args.input))
    for g in games:
        g["adj_probs"] = adjusted_probs(g, model)

    best = monte_carlo_optimize(games, iterations=args.iterations)
    save_predictions(games, best, Path(args.output))

    logger.info("Melhor EV estimado: %.4f", best["ev"])
    logger.info("Penalidade de concentração: %.4f", best["penalty"])
    logger.info("Objetivo final: %.4f", best["objective"])
    logger.info("Jogos duplos: %s", best["double_games"])
    logger.info("Predições salvas em %s", args.output)
