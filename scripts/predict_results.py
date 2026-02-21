import argparse
import csv
import json
import logging
import random
from collections import Counter
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


def rank_bucket_label(rank):
    if rank <= 3:
        return "B1"
    if rank <= 6:
        return "B2"
    if rank <= 10:
        return "B3"
    return "B4"


def gap_class_label(row, thresholds):
    top1 = row["top_order"][0]
    top2 = row["top_order"][1]
    gap = row["base_probs"][top1] - row["base_probs"][top2]
    if gap <= thresholds[0]:
        return "LOW"
    if gap <= thresholds[1]:
        return "MID"
    return "HIGH"


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


def topk_selection_counts(games, picks):
    counts = Counter({1: 0.0, 2: 0.0, 3: 0.0})
    total = 0.0
    for g in games:
        chosen = picks[g["jogo"]]
        for idx, cls in enumerate(g["top_order"], start=1):
            if cls in chosen:
                counts[idx] += 1.0
                total += 1.0
    return counts, max(total, 1.0)


def macro_penalty(games, picks, model):
    target = model.get("g_hist", model["global_topk"])
    counts, total = topk_selection_counts(games, picks)
    mix = {str(k): counts[k] / total for k in (1, 2, 3)}
    pen = sum((mix[str(k)] - float(target[str(k)])) ** 2 for k in (1, 2, 3))
    return pen, mix


def micro_penalty(games, picks, model):
    targets = model.get("micro_targets", {})
    thresholds = model.get("gap_thresholds", [0.05, 0.12])

    grp_counts = {}
    grp_totals = Counter()

    for g in games:
        group = f"{rank_bucket_label(g['rank_r'])}|{gap_class_label(g, thresholds)}"
        if group not in grp_counts:
            grp_counts[group] = Counter({1: 0.0, 2: 0.0, 3: 0.0})

        chosen = picks[g["jogo"]]
        for idx, cls in enumerate(g["top_order"], start=1):
            if cls in chosen:
                grp_counts[group][idx] += 1.0
                grp_totals[group] += 1.0

    total_sel = sum(grp_totals.values()) or 1.0
    penalty = 0.0
    profile = {}
    for group, total in grp_totals.items():
        if total <= 0:
            continue
        q = {str(k): grp_counts[group][k] / total for k in (1, 2, 3)}
        t = targets.get(group, model.get("g_hist", model["global_topk"]))
        diff = sum((q[str(k)] - float(t[str(k)])) ** 2 for k in (1, 2, 3))
        weight = total / total_sel
        penalty += weight * diff
        profile[group] = q

    return penalty, profile


def run_metrics_top1(games, picks):
    ordered = sorted(games, key=lambda x: x["rank_r"])
    seq = [1 if ordered[i]["top_order"][0] in picks[ordered[i]["jogo"]] else 0 for i in range(len(ordered))]

    if not seq:
        return {"run_count": 0, "max_run": 0}

    run_count = 1
    max_run = 1
    curr = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            curr += 1
            max_run = max(max_run, curr)
        else:
            run_count += 1
            curr = 1
    return {"run_count": run_count, "max_run": max_run}


def struct_penalty(games, picks, model):
    stats = model.get("run_stats", {})
    rmin = float(stats.get("run_count_min", 4.0))
    rmax = float(stats.get("run_count_max", 10.0))
    lmax = float(stats.get("max_run_max", 6.0))

    m = run_metrics_top1(games, picks)
    run_count = m["run_count"]
    max_run = m["max_run"]

    pen = max(0.0, rmin - run_count) ** 2
    pen += max(0.0, run_count - rmax) ** 2
    pen += max(0.0, max_run - lmax) ** 2
    return pen, m


def concentration_penalty(games, picks):
    pair_counts = Counter()
    rank_buckets = Counter()
    for g in games:
        pick = picks[g["jogo"]]
        if len(pick) == 2:
            pair_counts[tuple(sorted(pick))] += 1
            rank_buckets[rank_bucket_label(g["rank_r"])] += 1

    double_total = sum(pair_counts.values())
    if double_total == 0:
        return 0.0

    max_pair_share = max(pair_counts.values()) / double_total
    max_bucket_share = max(rank_buckets.values()) / double_total
    return max_pair_share + max_bucket_share


def monte_carlo_optimize(games, model, iterations=4000, lambdas=None):
    if lambdas is None:
        lambdas = {"macro": 0.05, "micro": 0.08, "struct": 0.03, "conc": 0.05}

    best = None
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

        pen_macro, g_mix = macro_penalty(games, picks, model)
        pen_micro, micro_profile = micro_penalty(games, picks, model)
        pen_struct, run_profile = struct_penalty(games, picks, model)
        pen_conc = concentration_penalty(games, picks)

        total_pen = (
            lambdas["macro"] * pen_macro
            + lambdas["micro"] * pen_micro
            + lambdas["struct"] * pen_struct
            + lambdas["conc"] * pen_conc
        )
        obj = ev - total_pen

        if (best is None) or (obj > best["objective"]):
            best = {
                "objective": obj,
                "ev": ev,
                "picks": picks,
                "double_games": sorted(double_games),
                "score_breakdown": {
                    "pen_macro": pen_macro,
                    "pen_micro": pen_micro,
                    "pen_struct": pen_struct,
                    "pen_conc": pen_conc,
                    "weighted_penalty": total_pen,
                },
                "profiles": {
                    "g_mix": g_mix,
                    "micro": micro_profile,
                    "runs": run_profile,
                },
            }
            logger.debug(
                "best-so-far obj=%.4f ev=%.4f macro=%.4f micro=%.4f struct=%.4f conc=%.4f",
                obj,
                ev,
                pen_macro,
                pen_micro,
                pen_struct,
                pen_conc,
            )
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
    parser.add_argument("--lambda-macro", type=float, default=0.05)
    parser.add_argument("--lambda-micro", type=float, default=0.08)
    parser.add_argument("--lambda-struct", type=float, default=0.03)
    parser.add_argument("--lambda-conc", type=float, default=0.05)
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

    lambdas = {
        "macro": args.lambda_macro,
        "micro": args.lambda_micro,
        "struct": args.lambda_struct,
        "conc": args.lambda_conc,
    }
    best = monte_carlo_optimize(games, model, iterations=args.iterations, lambdas=lambdas)
    save_predictions(games, best, Path(args.output))

    logger.info("Melhor EV estimado: %.4f", best["ev"])
    logger.info("Objetivo final: %.4f", best["objective"])
    logger.info("Breakdown score: %s", best["score_breakdown"])
    logger.info("Perfil g (top1/top2/top3): %s", best["profiles"]["g_mix"])
    logger.info("Perfil runs: %s", best["profiles"]["runs"])
    logger.info("Perfil bucket×classe: %s", best["profiles"]["micro"])
    logger.info("Jogos duplos: %s", best["double_games"])
    logger.info("Predições salvas em %s", args.output)
