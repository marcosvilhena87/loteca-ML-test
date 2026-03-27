import argparse
import logging
import math
import random
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def resolve_soft_metric_weights(model: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    default_metric_weights = {"avg_length": 1.0, "avg_count": 1.0, "avg_position": 1.0}
    configured = model.get("soft_metric_weights")
    if isinstance(configured, dict):
        resolved: Dict[str, Dict[str, float]] = {}
        for slot in TOP_SLOT_NAMES:
            slot_config = configured.get(slot, {})
            if isinstance(slot_config, dict):
                resolved[slot] = {metric: float(slot_config.get(metric, default_metric_weights[metric])) for metric in default_metric_weights}
            else:
                resolved[slot] = default_metric_weights.copy()
        return resolved

    legacy_slot_weights = model.get("soft_slot_weights", {})
    resolved = {}
    for slot in TOP_SLOT_NAMES:
        slot_weight = float(legacy_slot_weights.get(slot, 1.0))
        resolved[slot] = {metric: slot_weight for metric in default_metric_weights}
    return resolved


def soft_penalty(
    soft_metrics: Dict[str, Dict[str, float]],
    targets: Dict[str, Dict[str, float]],
    metric_weights: Dict[str, Dict[str, float]],
) -> float:
    penalty = 0.0
    for slot in TOP_SLOT_NAMES:
        slot_metric_weights = metric_weights.get(slot, {})
        current = soft_metrics.get(slot, {})
        target = targets.get(slot, {})
        for metric in ["avg_length", "avg_count", "avg_position"]:
            metric_weight = float(slot_metric_weights.get(metric, 1.0))
            tv = float(target.get(metric, 0.0))
            cv = float(current.get(metric, 0.0))
            std = float(target.get(f"{metric}_std", 0.0))
            if std > 1e-9:
                penalty += metric_weight * (abs(cv - tv) / std)
            else:
                scale = max(1.0, abs(tv))
                penalty += metric_weight * (abs(cv - tv) / scale)
    return penalty


def soft_penalty_breakdown(
    soft_metrics: Dict[str, Dict[str, float]],
    targets: Dict[str, Dict[str, float]],
    metric_weights: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    breakdown: Dict[str, Dict[str, float]] = {}
    for slot in TOP_SLOT_NAMES:
        breakdown[slot] = {}
        slot_metric_weights = metric_weights.get(slot, {})
        current = soft_metrics.get(slot, {})
        target = targets.get(slot, {})
        for metric in ["avg_length", "avg_count", "avg_position"]:
            metric_weight = float(slot_metric_weights.get(metric, 1.0))
            tv = float(target.get(metric, 0.0))
            cv = float(current.get(metric, 0.0))
            std = float(target.get(f"{metric}_std", 0.0))
            normalized_diff = (abs(cv - tv) / std) if std > 1e-9 else (abs(cv - tv) / max(1.0, abs(tv)))
            breakdown[slot][metric] = metric_weight * normalized_diff
        breakdown[slot]["total"] = sum(breakdown[slot][metric] for metric in ["avg_length", "avg_count", "avg_position"])
    return breakdown


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


def improve_with_soft_constraints(
    rows: List[MatchRow],
    picks: List[int],
    targets: Dict[str, Dict[str, float]],
    hard: Dict[str, int],
    metric_weights: Dict[str, Dict[str, float]],
    top3_avg_count_gap_tolerance: float = 0.30,
) -> List[int]:
    eps = 1e-9

    def base_score(local_picks: List[int]) -> float:
        return sum(option_probability(row, OPTIONS[pick][1]) for row, pick in zip(rows, local_picks))

    def feasible(local_picks: List[int]) -> bool:
        c1, c2, c3, secos, triplos = hard_counts_from_picks(local_picks)
        return c1 == hard["top1"] and c2 == hard["top2"] and c3 == hard["top3"] and secos == hard["secos"] and triplos == hard["triplos"]

    def objective(local_picks: List[int], soft_weight: float) -> float:
        return base_score(local_picks) - soft_weight * soft_penalty(compute_soft_metrics(rows, local_picks), targets, metric_weights)

    def top3_avg_count_gap(local_picks: List[int]) -> float:
        top3_stats = compute_soft_metrics(rows, local_picks).get("top3", {})
        target = float(targets.get("top3", {}).get("avg_count", top3_stats.get("avg_count", 0.0)))
        current = float(top3_stats.get("avg_count", 0.0))
        return abs(current - target)

    def lexicographic_score(local_picks: List[int], soft_weight: float) -> Tuple[int, float, float]:
        # Prioridade com faixa de tolerância:
        # 1) entrar na faixa de gap_top3.avg_count <= tolerância;
        # 2) dentro da faixa, maximizar objetivo global;
        # 3) fora da faixa, minimizar gap e depois maximizar objetivo.
        gap = top3_avg_count_gap(local_picks)
        in_tolerance_band = 1 if gap <= top3_avg_count_gap_tolerance + eps else 0
        return in_tolerance_band, gap, objective(local_picks, soft_weight)

    def is_better_score(candidate_score: Tuple[int, float, float], best_score: Tuple[int, float, float]) -> bool:
        cand_in_band, cand_gap, cand_obj = candidate_score
        best_in_band, best_gap, best_obj = best_score
        if cand_in_band != best_in_band:
            return cand_in_band > best_in_band
        if cand_in_band == 1:
            return cand_obj > best_obj + eps or (abs(cand_obj - best_obj) <= eps and cand_gap < best_gap - eps)
        return cand_gap < best_gap - eps or (abs(cand_gap - best_gap) <= eps and cand_obj > best_obj + eps)

    def local_search(seed_picks: List[int], soft_weight: float) -> List[int]:
        current = seed_picks[:]
        best_score = lexicographic_score(current, soft_weight)
        improved = True
        iters = 0
        while improved and iters < 10:
            improved = False
            iters += 1
            if _targeted_top3_run_repair(rows, current, targets, feasible, lambda p: lexicographic_score(p, soft_weight)):
                best_score = lexicographic_score(current, soft_weight)
                improved = True
                logging.debug(
                    "Melhoria cirúrgica para runs top3 encontrada: w=%.2f in_band=%s gap=%.6f obj=%.6f",
                    soft_weight,
                    best_score[0],
                    best_score[1],
                    best_score[2],
                )
                continue
            for k in (2, 3):
                if _try_k_swap_neighborhood(current, k, feasible, lambda p: lexicographic_score(p, soft_weight), best_score):
                    best_score = lexicographic_score(current, soft_weight)
                    improved = True
                    logging.debug(
                        "Melhoria local (k=%s) encontrada: w=%.2f in_band=%s gap=%.6f obj=%.6f",
                        k,
                        soft_weight,
                        best_score[0],
                        best_score[1],
                        best_score[2],
                    )
                    break
            if improved:
                continue
            if _try_k_swap_neighborhood(current, 4, feasible, lambda p: lexicographic_score(p, soft_weight), best_score):
                best_score = lexicographic_score(current, soft_weight)
                improved = True
                logging.debug(
                    "Melhoria local (k=4) encontrada: w=%.2f in_band=%s gap=%.6f obj=%.6f",
                    soft_weight,
                    best_score[0],
                    best_score[1],
                    best_score[2],
                )
                continue
            if improved:
                continue
            annealed = _simulated_annealing(
                current,
                feasible,
                lambda p: lexicographic_score(p, soft_weight),
                max_steps=350,
                max_move_k=5,
            )
            annealed_score = lexicographic_score(annealed, soft_weight)
            if is_better_score(annealed_score, best_score):
                current = annealed
                best_score = annealed_score
                improved = True
                logging.debug(
                    "Melhoria por annealing encontrada: w=%.2f in_band=%s gap=%.6f obj=%.6f",
                    soft_weight,
                    best_score[0],
                    best_score[1],
                    best_score[2],
                )
        return current

    if not feasible(picks):
        raise RuntimeError("Solução inicial inviável para as restrições hard.")

    staged_weights = [0.08, 0.15, 0.25, 0.40]
    current = picks[:]
    for weight in staged_weights:
        before_score = lexicographic_score(current, weight)
        seed_candidates = [current[:]]
        for _ in range(4):
            perturbed = _sample_feasible_perturbation(current, feasible, max_k=5, attempts=220)
            if perturbed is not None:
                seed_candidates.append(perturbed)

        best_candidate = current[:]
        best_candidate_score = before_score
        for seed in seed_candidates:
            refined = local_search(seed, weight)
            refined_score = lexicographic_score(refined, weight)
            if is_better_score(refined_score, best_candidate_score):
                best_candidate = refined
                best_candidate_score = refined_score

        current = best_candidate
        best_in_band, best_top3_gap, after = best_candidate_score
        logging.info(
            "Busca soft (peso=%.2f, starts=%s): obj antes=%.6f | obj depois=%.6f | in_band=%s | gap_top3.avg_count=%.6f (tol=%.6f)",
            weight,
            len(seed_candidates),
            before_score[2],
            after,
            bool(best_in_band),
            best_top3_gap,
            top3_avg_count_gap_tolerance,
        )
    return current


def _try_k_swap_neighborhood(
    picks: List[int],
    k: int,
    feasible_fn,
    score_fn,
    current_score: Tuple[int, float, float],
) -> bool:
    eps = 1e-9
    n = len(picks)
    if k > n:
        return False

    def backtrack(depth: int, start: int, idxs: List[int]) -> bool:
        if depth == k:
            return evaluate_indices(idxs)
        for idx in range(start, n):
            idxs.append(idx)
            if backtrack(depth + 1, idx + 1, idxs):
                return True
            idxs.pop()
        return False

    def enumerate_options(indices: List[int], pos: int, candidate: List[int]) -> bool:
        if pos == len(indices):
            if not feasible_fn(candidate):
                return False
            cand_score = score_fn(candidate)
            cand_in_band, cand_gap, cand_obj = cand_score
            current_in_band, current_gap, current_obj = current_score
            if cand_in_band != current_in_band:
                is_better = cand_in_band > current_in_band
            elif cand_in_band == 1:
                is_better = cand_obj > current_obj + eps or (abs(cand_obj - current_obj) <= eps and cand_gap < current_gap - eps)
            else:
                is_better = cand_gap < current_gap - eps or (abs(cand_gap - current_gap) <= eps and cand_obj > current_obj + eps)
            if is_better:
                picks[:] = candidate
                return True
            return False
        idx = indices[pos]
        current_pick = picks[idx]
        for option_idx, _ in enumerate(OPTIONS):
            if option_idx == current_pick:
                continue
            candidate[idx] = option_idx
            if enumerate_options(indices, pos + 1, candidate):
                return True
        candidate[idx] = current_pick
        return False

    def evaluate_indices(indices: List[int]) -> bool:
        candidate = picks[:]
        return enumerate_options(indices, 0, candidate)

    return backtrack(0, 0, [])


def _simulated_annealing(
    picks: List[int],
    feasible_fn,
    score_fn,
    max_steps: int = 200,
    max_move_k: int = 3,
) -> List[int]:
    eps = 1e-9
    current = picks[:]
    current_in_band, current_gap, current_obj = score_fn(current)
    best = current[:]
    best_in_band, best_gap, best_obj = current_in_band, current_gap, current_obj
    temperature = 0.5

    for step in range(max_steps):
        k_choices = [2, 3]
        if max_move_k >= 4:
            k_choices.extend([4, 4])
        if max_move_k >= 5:
            k_choices.append(5)
        k = random.choice(k_choices)
        idxs = random.sample(range(len(current)), k=min(k, len(current)))
        candidate = current[:]
        changed = False
        for idx in idxs:
            options_pool = [oi for oi, _ in enumerate(OPTIONS) if oi != candidate[idx]]
            if not options_pool:
                continue
            candidate[idx] = random.choice(options_pool)
            changed = True
        if not changed or not feasible_fn(candidate):
            continue

        cand_in_band, cand_gap, cand_obj = score_fn(candidate)
        delta = cand_obj - current_obj
        accept = False
        if cand_in_band > current_in_band:
            accept = True
        elif cand_in_band < current_in_band:
            accept = False
        elif cand_in_band == 1:
            accept = delta > 0 or random.random() < math.exp(delta / max(1e-6, temperature))
        elif cand_gap < current_gap - eps:
            accept = True
        elif abs(cand_gap - current_gap) <= eps:
            accept = delta > 0 or random.random() < math.exp(delta / max(1e-6, temperature))

        if accept:
            current = candidate
            current_in_band = cand_in_band
            current_gap = cand_gap
            current_obj = cand_obj
            if (
                cand_in_band > best_in_band
                or (cand_in_band == best_in_band == 1 and (cand_obj > best_obj + eps or (abs(cand_obj - best_obj) <= eps and cand_gap < best_gap - eps)))
                or (cand_in_band == best_in_band == 0 and (cand_gap < best_gap - eps or (abs(cand_gap - best_gap) <= eps and cand_obj > best_obj + eps)))
            ):
                best = candidate
                best_in_band = cand_in_band
                best_gap = cand_gap
                best_obj = cand_obj
        temperature *= 0.985
        if temperature < 0.01 and step > max_steps * 0.6:
            break
    return best


def _sample_feasible_perturbation(
    picks: List[int],
    feasible_fn,
    max_k: int = 5,
    attempts: int = 120,
) -> Optional[List[int]]:
    n = len(picks)
    if n < 2:
        return None
    for _ in range(attempts):
        candidate = picks[:]
        k = random.randint(2, min(max_k, n))
        idxs = random.sample(range(n), k)
        changed = False
        for idx in idxs:
            choices = [option_idx for option_idx, _ in enumerate(OPTIONS) if option_idx != candidate[idx]]
            if not choices:
                continue
            candidate[idx] = random.choice(choices)
            changed = True
        if changed and feasible_fn(candidate):
            return candidate
    return None


def _targeted_top3_run_repair(
    rows: List[MatchRow],
    picks: List[int],
    targets: Dict[str, Dict[str, float]],
    feasible_fn,
    score_fn,
) -> bool:
    target_avg_count = float(targets.get("top3", {}).get("avg_count", 0.0))
    if target_avg_count <= 0:
        return False

    sorted_idx = sorted(range(len(rows)), key=lambda i: rows[i].p_top3, reverse=True)

    def sequence(local_picks: List[int]) -> List[int]:
        return [1 if 2 in OPTIONS[local_picks[i]][1] else 0 for i in sorted_idx]

    seq = sequence(picks)
    current_avg_count = runs_metrics(seq).avg_count
    if current_avg_count <= target_avg_count + 1e-9:
        return False

    current_score = score_fn(picks)
    current_gap = abs(current_avg_count - target_avg_count)
    best_candidate: Optional[List[int]] = None
    best_score = current_score
    best_gap = current_gap

    for pos in range(len(sorted_idx) - 1):
        if seq[pos] == seq[pos + 1]:
            continue
        i = sorted_idx[pos]
        j = sorted_idx[pos + 1]
        pi = picks[i]
        pj = picks[j]
        for oi, _ in enumerate(OPTIONS):
            for oj, _ in enumerate(OPTIONS):
                if oi == pi and oj == pj:
                    continue
                candidate = picks[:]
                candidate[i] = oi
                candidate[j] = oj
                if not feasible_fn(candidate):
                    continue
                candidate_seq = sequence(candidate)
                candidate_avg_count = runs_metrics(candidate_seq).avg_count
                candidate_gap = abs(candidate_avg_count - target_avg_count)
                if candidate_gap > current_gap + 1e-9:
                    continue
                cand_score = score_fn(candidate)
                cand_in_band, _, cand_obj = cand_score
                best_in_band, _, best_obj = best_score
                better = False
                if cand_in_band != best_in_band:
                    better = cand_in_band > best_in_band
                elif cand_in_band == 1:
                    better = cand_obj > best_obj + 1e-9 or (abs(cand_obj - best_obj) <= 1e-9 and candidate_gap < best_gap - 1e-9)
                else:
                    better = candidate_gap < best_gap - 1e-9 or (abs(candidate_gap - best_gap) <= 1e-9 and cand_obj > best_obj + 1e-9)
                if better:
                    best_candidate = candidate
                    best_score = cand_score
                    best_gap = candidate_gap

    if best_candidate is not None:
        curr_in_band, _, curr_obj = current_score
        best_in_band, _, best_obj = best_score
        improved = False
        if best_in_band != curr_in_band:
            improved = best_in_band > curr_in_band
        elif best_in_band == 1:
            improved = best_obj > curr_obj + 1e-9 or best_gap < current_gap - 1e-9
        else:
            improved = best_gap < current_gap - 1e-9 or best_obj > curr_obj + 1e-9
    else:
        improved = False

    if best_candidate is not None and improved:
        picks[:] = best_candidate
        return True
    return False


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
    top3_avg_count_gap_tolerance = float(model.get("top3_avg_count_gap_tolerance", 0.30))
    soft_metric_weights = resolve_soft_metric_weights(model)

    logging.info("Restrições hard em uso: %s", hard)
    logging.info("Tolerância do gap top3.avg_count em uso: %.6f", top3_avg_count_gap_tolerance)
    logging.info("Pesos soft por slot/métrica em uso: %s", soft_metric_weights)
    picks = build_prediction(rows, hard)
    picks = improve_with_soft_constraints(
        rows,
        picks,
        model.get("targets", {}),
        hard,
        soft_metric_weights,
        top3_avg_count_gap_tolerance=top3_avg_count_gap_tolerance,
    )

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
    penalty_breakdown = soft_penalty_breakdown(soft_metrics, model.get("targets", {}), soft_metric_weights)
    slot_totals = {slot: values["total"] for slot, values in penalty_breakdown.items()}
    logging.info("Penalidade soft por slot/métrica: %s", penalty_breakdown)
    logging.info("Penalidade total por slot: %s", slot_totals)

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
