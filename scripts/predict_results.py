import argparse
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from scripts.common import (
    TOP_SLOT_NAMES,
    MatchRow,
    load_json,
    normalize_row,
    rank_outcomes,
    read_csv_semicolon,
    runs_metrics,
    setup_logging,
    write_csv_semicolon,
)

OPTIONS: List[Tuple[str, Tuple[int, ...]]] = [
    ("S1", (0,)),
    ("S2", (1,)),
    ("S3", (2,)),
    ("D12", (0, 1)),
    ("D13", (0, 2)),
    ("D23", (1, 2)),
    ("T123", (0, 1, 2)),
]


def option_counts(slots: Sequence[int]) -> Tuple[int, int, int, int, int]:
    is_seco = 1 if len(slots) == 1 else 0
    is_triplo = 1 if len(slots) == 3 else 0
    c1 = 1 if 0 in slots else 0
    c2 = 1 if 1 in slots else 0
    c3 = 1 if 2 in slots else 0
    return c1, c2, c3, is_seco, is_triplo


def option_probability(match: MatchRow, slots: Sequence[int]) -> float:
    slot_probs = [match.p_top1, match.p_top2, match.p_top3]
    return sum(slot_probs[idx] for idx in slots)


def can_still_finish(rem_games: int, used: Tuple[int, int, int, int, int], hard: Dict[str, int]) -> bool:
    top_targets = (hard["top1"], hard["top2"], hard["top3"])
    secos_target = hard["secos"]
    triplos_target = hard["triplos"]
    for i in range(3):
        needed = top_targets[i] - used[i]
        if needed < 0 or needed > rem_games:
            return False
    secos_needed = secos_target - used[3]
    triplos_needed = triplos_target - used[4]
    if secos_needed < 0 or triplos_needed < 0:
        return False
    if secos_needed > rem_games or triplos_needed > rem_games:
        return False
    if secos_needed + triplos_needed > rem_games:
        return False
    return True


def build_prediction(rows: List[MatchRow], hard: Dict[str, int]) -> List[int]:
    scores = [[option_probability(row, slots) for _, slots in OPTIONS] for row in rows]
    option_vectors = [option_counts(slots) for _, slots in OPTIONS]

    @lru_cache(maxsize=None)
    def dp(idx: int, c1: int, c2: int, c3: int, secos: int, triplos: int) -> float:
        if idx == len(rows):
            if c1 == hard["top1"] and c2 == hard["top2"] and c3 == hard["top3"] and secos == hard["secos"] and triplos == hard["triplos"]:
                return 0.0
            return float("-inf")

        best = float("-inf")
        rem_games = len(rows) - idx - 1
        for option_idx, vec in enumerate(option_vectors):
            n_state = (c1 + vec[0], c2 + vec[1], c3 + vec[2], secos + vec[3], triplos + vec[4])
            if not can_still_finish(rem_games, n_state, hard):
                continue
            future = dp(idx + 1, *n_state)
            if future == float("-inf"):
                continue
            best = max(best, scores[idx][option_idx] + future)
        return best

    picks: List[int] = []
    state = (0, 0, 0, 0, 0)
    for idx in range(len(rows)):
        best_idx: Optional[int] = None
        best_val = float("-inf")
        rem_games = len(rows) - idx - 1
        for option_idx, vec in enumerate(option_vectors):
            n_state = (state[0] + vec[0], state[1] + vec[1], state[2] + vec[2], state[3] + vec[3], state[4] + vec[4])
            if not can_still_finish(rem_games, n_state, hard):
                continue
            future = dp(idx + 1, *n_state)
            if future == float("-inf"):
                continue
            candidate = scores[idx][option_idx] + future
            if candidate > best_val:
                best_val = candidate
                best_idx = option_idx

        if best_idx is None:
            raise RuntimeError("Não foi possível construir palpite viável com as restrições fornecidas.")
        picks.append(best_idx)
        vec = option_vectors[best_idx]
        state = (state[0] + vec[0], state[1] + vec[1], state[2] + vec[2], state[3] + vec[3], state[4] + vec[4])

    return picks


def compute_soft_metrics(rows: List[MatchRow], picks: List[int]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for slot_idx, slot_name in enumerate(TOP_SLOT_NAMES):
        sorted_idx = sorted(range(len(rows)), key=lambda i: getattr(rows[i], f"p_top{slot_idx+1}"), reverse=True)
        seq = [1 if slot_idx in OPTIONS[picks[i]][1] else 0 for i in sorted_idx]
        stats = runs_metrics(seq)
        result[slot_name] = {"avg_length": stats.avg_length, "avg_count": stats.avg_count, "avg_position": stats.avg_position}
    return result


def soft_penalty(soft_metrics: Dict[str, Dict[str, float]], targets: Dict[str, Dict[str, float]]) -> float:
    penalty = 0.0
    for slot in TOP_SLOT_NAMES:
        current = soft_metrics.get(slot, {})
        target = targets.get(slot, {})
        for metric in ["avg_length", "avg_count", "avg_position"]:
            tv = float(target.get(metric, 0.0))
            cv = float(current.get(metric, 0.0))
            scale = max(1.0, abs(tv))
            penalty += abs(cv - tv) / scale
    return penalty


def hard_counts_from_picks(picks: List[int]) -> Tuple[int, int, int, int, int]:
    c1 = c2 = c3 = secos = triplos = 0
    for pick in picks:
        slots = OPTIONS[pick][1]
        c1 += 1 if 0 in slots else 0
        c2 += 1 if 1 in slots else 0
        c3 += 1 if 2 in slots else 0
        secos += 1 if len(slots) == 1 else 0
        triplos += 1 if len(slots) == 3 else 0
    return c1, c2, c3, secos, triplos


def improve_with_soft_constraints(rows: List[MatchRow], picks: List[int], targets: Dict[str, Dict[str, float]], hard: Dict[str, int]) -> List[int]:
    def base_score(local_picks: List[int]) -> float:
        return sum(option_probability(row, OPTIONS[pick][1]) for row, pick in zip(rows, local_picks))

    def objective(local_picks: List[int]) -> float:
        # prioriza cobertura; usa soft como desempate leve
        return base_score(local_picks) - 0.08 * soft_penalty(compute_soft_metrics(rows, local_picks), targets)

    current = picks[:]
    best_obj = objective(current)
    improved = True
    iters = 0
    while improved and iters < 8:
        improved = False
        iters += 1
        for i in range(len(current)):
            for oi, _ in enumerate(OPTIONS):
                if oi == current[i]:
                    continue
                for j in range(i + 1, len(current)):
                    for oj, _ in enumerate(OPTIONS):
                        if oj == current[j]:
                            continue
                        candidate = current[:]
                        candidate[i] = oi
                        candidate[j] = oj
                        c1, c2, c3, secos, triplos = hard_counts_from_picks(candidate)
                        if c1 != hard["top1"] or c2 != hard["top2"] or c3 != hard["top3"] or secos != hard["secos"] or triplos != hard["triplos"]:
                            continue
                        cand_obj = objective(candidate)
                        if cand_obj > best_obj + 1e-9:
                            current = candidate
                            best_obj = cand_obj
                            improved = True
                            logging.debug("Melhoria soft encontrada: obj=%.6f", cand_obj)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return current


def outcomes_from_slots(match: MatchRow, slots: Sequence[int]) -> str:
    ranking = rank_outcomes(match.p1, match.px, match.p2)
    selected = {ranking[idx][0] for idx in slots}
    ordered = [symbol for symbol in ["1", "X", "2"] if symbol in selected]
    if len(ordered) == 3:
        return "1X2"
    return "".join(ordered)


def predict(next_path: str, model_path: str, out_path: str) -> None:
    model = load_json(model_path)
    rows = [normalize_row(row) for row in read_csv_semicolon(next_path)]
    if not rows:
        raise RuntimeError("Arquivo de próximo concurso sem jogos.")

    hard_raw = model.get("hard_constraints", {})
    hard = {"top1": int(hard_raw.get("top1", 9)), "top2": int(hard_raw.get("top2", 5)), "top3": int(hard_raw.get("top3", 4)), "secos": int(hard_raw.get("secos", 12)), "triplos": int(hard_raw.get("triplos", 2))}

    logging.info("Restrições hard em uso: %s", hard)
    picks = build_prediction(rows, hard)
    picks = improve_with_soft_constraints(rows, picks, model.get("targets", {}), hard)

    out_rows = []
    sum_top = [0, 0, 0]
    secos = 0
    triplos = 0
    for row, option_idx in zip(rows, picks):
        option_name, slots = OPTIONS[option_idx]
        palpite = outcomes_from_slots(row, slots)
        for slot in slots:
            sum_top[slot] += 1
        secos += 1 if len(slots) == 1 else 0
        triplos += 1 if len(slots) == 3 else 0
        out_rows.append({
            "Concurso": row.concurso,
            "Jogo": row.jogo,
            "Mandante": row.mandante,
            "Visitante": row.visitante,
            "Palpite": palpite,
            "Tipo": option_name,
            "Prob_Cobertura": f"{option_probability(row, slots):.6f}",
            "p(top1)": f"{row.p_top1:.6f}",
            "p(top2)": f"{row.p_top2:.6f}",
            "p(top3)": f"{row.p_top3:.6f}",
        })
        logging.debug("Jogo %s %s x %s | tipo=%s palpite=%s", row.jogo, row.mandante, row.visitante, option_name, palpite)

    soft_metrics = compute_soft_metrics(rows, picks)
    logging.info("Soma top slots no palpite: top1=%s top2=%s top3=%s", *sum_top)
    logging.info("Secos=%s Triplos=%s", secos, triplos)
    logging.info("Métricas soft obtidas: %s", soft_metrics)
    logging.info("Métricas soft alvo: %s", model.get("targets", {}))

    write_csv_semicolon(
        out_path,
        out_rows,
        fieldnames=["Concurso", "Jogo", "Mandante", "Visitante", "Palpite", "Tipo", "Prob_Cobertura", "p(top1)", "p(top2)", "p(top3)"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera palpites Loteca com restrições hard/soft")
    parser.add_argument("--next", default="data/proximo_concurso.csv")
    parser.add_argument("--model", default="models/model.json")
    parser.add_argument("--output", default="output/predictions.csv")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))
    predict(args.next, args.model, args.output)
