import argparse
import json
import logging
import random
from functools import lru_cache
from typing import Dict, List, Tuple

from scripts.common import enrich_probabilities, read_csv_semicolon, run_features, write_csv_semicolon

OPTION_RANKS = {
    "1": (1,),
    "2": (2,),
    "3": (3,),
    "12": (1, 2),
    "13": (1, 3),
    "23": (2, 3),
}

BASE_SCORE_TIE_EPSILON = 0.02


def option_base_score(prob: Dict[str, float], option: str) -> float:
    return sum(prob[f"top{rank}"] for rank in OPTION_RANKS[option])


def compute_counts(assignments: List[str]) -> Dict[str, int]:
    top = {1: 0, 2: 0, 3: 0}
    duplos = 0
    triplos = 0
    for opt in assignments:
        ranks = OPTION_RANKS[opt]
        if len(ranks) == 2:
            duplos += 1
        elif len(ranks) == 3:
            triplos += 1
        for r in ranks:
            top[r] += 1
    secos = len(assignments) - duplos - triplos
    return {
        "top1": top[1],
        "top2": top[2],
        "top3": top[3],
        "secos": secos,
        "duplos": duplos,
        "triplos": triplos,
    }


def rank_binary_metrics(jogos: List[Dict[str, object]], assignments: List[str], rank_idx: int) -> Dict[str, float]:
    rank_key = f"top{rank_idx}"
    data = []
    for i, jogo in enumerate(jogos):
        prob = jogo["_top_probs"][rank_key]
        chosen = 1 if rank_idx in OPTION_RANKS[assignments[i]] else 0
        data.append((prob, chosen))

    data.sort(key=lambda x: x[0], reverse=True)
    binary = [x[1] for x in data]
    return run_features(binary)


def soft_penalty(jogos: List[Dict[str, object]], assignments: List[str], targets: Dict[str, Dict[str, float]]) -> float:
    penalty = 0.0
    for rank_idx in (1, 2, 3):
        m = rank_binary_metrics(jogos, assignments, rank_idx)
        target = targets[f"top{rank_idx}"]
        penalty += abs(m["avg_run_length"] - target["avg_run_length"])
        penalty += abs(m["avg_run_count"] - target["avg_run_count"])
        penalty += abs(m["avg_run_position"] - target["avg_run_position"])
    return penalty


def run_guardrails(targets: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    guardrails = {}
    for rank_idx in (1, 2, 3):
        rank = f"top{rank_idx}"
        target = targets[rank]
        guardrails[rank] = {
            "max_avg_run_length": target["avg_run_length"] + 1.25,
            "min_avg_run_count": max(0.5, target["avg_run_count"] - 1.25),
        }
    return guardrails


def structural_run_penalty(
    jogos: List[Dict[str, object]],
    assignments: List[str],
    targets: Dict[str, Dict[str, float]],
) -> float:
    penalty = 0.0
    guardrails = run_guardrails(targets)
    for rank_idx in (1, 2, 3):
        rank = f"top{rank_idx}"
        metrics = rank_binary_metrics(jogos, assignments, rank_idx)
        max_len = guardrails[rank]["max_avg_run_length"]
        min_count = guardrails[rank]["min_avg_run_count"]

        if metrics["avg_run_length"] > max_len:
            penalty += metrics["avg_run_length"] - max_len
        if metrics["avg_run_count"] < min_count:
            penalty += min_count - metrics["avg_run_count"]
    return penalty


def state_score(
    jogos: List[Dict[str, object]],
    state: List[str],
    targets: Dict[str, Dict[str, float]],
    penalty_weight: float,
) -> Dict[str, float]:
    base = sum(option_base_score(jogos[i]["_top_probs"], state[i]) for i in range(len(state)))
    structural = structural_run_penalty(jogos, state, targets)
    return {
        "base_score": base,
        "structural_penalty": structural,
        "objective": base - penalty_weight * structural,
    }


def is_better_state(candidate: Dict[str, float], reference: Dict[str, float]) -> bool:
    if candidate["base_score"] > reference["base_score"] + BASE_SCORE_TIE_EPSILON:
        return True
    if abs(candidate["base_score"] - reference["base_score"]) <= BASE_SCORE_TIE_EPSILON:
        if candidate["structural_penalty"] < reference["structural_penalty"]:
            return True
        if candidate["structural_penalty"] == reference["structural_penalty"]:
            return candidate["objective"] > reference["objective"]
    return False


def feasible_type_configs() -> List[Dict[str, int]]:
    configs = []
    for n12 in range(4):
        for n13 in range(4 - n12):
            n23 = 3 - n12 - n13
            s1 = 8 - n12 - n13
            s2 = 5 - n12 - n23
            s3 = 4 - n13 - n23
            if min(s1, s2, s3) < 0:
                continue
            if s1 + s2 + s3 != 11:
                continue
            configs.append({"1": s1, "2": s2, "3": s3, "12": n12, "13": n13, "23": n23})
    return configs


def best_assignment_for_config(jogos: List[Dict[str, object]], config: Dict[str, int]) -> Tuple[float, List[str]]:
    n = len(jogos)
    types = tuple(["1", "2", "3", "12", "13", "23"])
    counts_init = tuple(config[t] for t in types)

    @lru_cache(maxsize=None)
    def dp(i: int, counts: Tuple[int, ...]) -> Tuple[float, Tuple[str, ...]]:
        if i == n:
            if all(c == 0 for c in counts):
                return 0.0, tuple()
            return -10**9, tuple()

        best_score = -10**9
        best_path: Tuple[str, ...] = tuple()
        prob = jogos[i]["_top_probs"]

        for idx, typ in enumerate(types):
            if counts[idx] <= 0:
                continue
            new_counts = list(counts)
            new_counts[idx] -= 1
            next_score, next_path = dp(i + 1, tuple(new_counts))
            if next_score < -10**8:
                continue
            score = option_base_score(prob, typ) + next_score
            if score > best_score:
                best_score = score
                best_path = (typ,) + next_path

        return best_score, best_path

    score, path = dp(0, counts_init)
    return score, list(path)


def local_search(
    jogos: List[Dict[str, object]],
    assignments: List[str],
    targets: Dict[str, Dict[str, float]],
    penalty_weight: float,
    iterations: int,
    seed: int,
) -> Tuple[List[str], Dict[str, float]]:
    rnd = random.Random(seed)

    best = list(assignments)
    current = list(assignments)
    best_score = state_score(jogos, best, targets, penalty_weight)
    current_score = dict(best_score)

    for _ in range(iterations):
        i, j = rnd.sample(range(len(current)), 2)
        if current[i] == current[j]:
            continue
        candidate = list(current)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        cand_score = state_score(jogos, candidate, targets, penalty_weight)

        if is_better_state(cand_score, current_score) or rnd.random() < 0.02:
            current = candidate
            current_score = cand_score
            if is_better_state(cand_score, best_score):
                best = candidate
                best_score = cand_score

    debug = {
        "objective": best_score["objective"],
        "base_score": best_score["base_score"],
        "structural_penalty": best_score["structural_penalty"],
        "legacy_soft_penalty": soft_penalty(jogos, best, targets),
    }
    return best, debug


def option_to_palpite(option: str, symbols: Dict[str, str]) -> str:
    if option == "1":
        return symbols["top1"]
    if option == "2":
        return symbols["top2"]
    if option == "3":
        return symbols["top3"]
    if option == "12":
        return f"{symbols['top1']}{symbols['top2']}"
    if option == "13":
        return f"{symbols['top1']}{symbols['top3']}"
    if option == "23":
        return f"{symbols['top2']}{symbols['top3']}"
    return "1X2"


def format_palpite(raw: str) -> str:
    if len(raw) == 1:
        return raw
    order = {"1": 0, "X": 1, "2": 2}
    chars = "".join(sorted(set(raw), key=lambda c: order[c]))
    if chars == "1X2":
        return "1X2"
    return chars


def predict(next_path: str, model_path: str, output_path: str) -> Dict[str, object]:
    jogos_raw = read_csv_semicolon(next_path)
    jogos = [enrich_probabilities(row) for row in jogos_raw]

    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    targets = model["soft_targets"]
    search = model["search"]

    best_global = None
    best_debug = None

    for config in feasible_type_configs():
        if sum(config.values()) != len(jogos):
            continue
        base_score, assignment = best_assignment_for_config(jogos, config)
        if not assignment:
            continue
        improved, debug = local_search(
            jogos,
            assignment,
            targets,
            penalty_weight=search["soft_penalty_weight"],
            iterations=search["local_search_iterations"],
            seed=search["random_seed"],
        )
        debug["config"] = config
        debug["initial_base_score"] = base_score

        if best_debug is None or debug["objective"] > best_debug["objective"]:
            best_global = improved
            best_debug = debug

    if best_global is None:
        raise RuntimeError("Não foi possível gerar palpite viável com as hard constraints.")

    rows_out = []
    for i, jogo in enumerate(jogos):
        raw = option_to_palpite(best_global[i], jogo["_top_symbols"])
        palpite = format_palpite(raw)
        rows_out.append({
            "Concurso": jogo["Concurso"],
            "Jogo": jogo["Jogo"],
            "Mandante": jogo["Mandante"],
            "Visitante": jogo["Visitante"],
            "p(top1)": f"{jogo['_top_probs']['top1']:.6f}",
            "p(top2)": f"{jogo['_top_probs']['top2']:.6f}",
            "p(top3)": f"{jogo['_top_probs']['top3']:.6f}",
            "tipo": best_global[i],
            "Palpite": palpite,
        })

    rows_out.sort(key=lambda x: int(x["Jogo"]))
    write_csv_semicolon(output_path, list(rows_out[0].keys()), rows_out)

    counts = compute_counts(best_global)
    logging.info("Hard constraints obtidas: %s", counts)
    logging.info("Debug busca: %s", best_debug)

    metrics_debug = {
        "top1": rank_binary_metrics(jogos, best_global, 1),
        "top2": rank_binary_metrics(jogos, best_global, 2),
        "top3": rank_binary_metrics(jogos, best_global, 3),
    }
    logging.info("Métricas das runs do palpite: %s", metrics_debug)

    return {"counts": counts, "search_debug": best_debug, "run_metrics": metrics_debug}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--next", default="data/proximo_concurso.csv")
    parser.add_argument("--model", default="models/model.json")
    parser.add_argument("--output", default="output/predictions.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    summary = predict(args.next, args.model, args.output)
    logging.info("Resumo final: %s", summary)


if __name__ == "__main__":
    main()
