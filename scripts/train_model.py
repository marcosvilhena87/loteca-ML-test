import argparse
import logging
import math
from bisect import bisect_right

from scripts.common import dump_json, load_csv, load_json, setup_logging

LOGGER = logging.getLogger(__name__)

BANDS = {
    "high": (0.60, 1.01),
    "mid": (0.45, 0.60),
    "low": (0.0, 0.45),
}


def bucket_id(value, n_buckets=10):
    idx = int(value * n_buckets)
    if idx >= n_buckets:
        idx = n_buckets - 1
    return idx


def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def clamp_prob(p, eps=1e-6):
    return min(1.0 - eps, max(eps, p))


def brier_score(preds, ys):
    if not preds:
        return float("inf")
    return sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(preds)


def fit_bucket(xs, ys, n_buckets=10):
    buckets = {i: {"count": 0, "hits": 0} for i in range(n_buckets)}
    for x, y in zip(xs, ys):
        b = bucket_id(x, n_buckets=n_buckets)
        buckets[b]["count"] += 1
        buckets[b]["hits"] += int(y)
    values = []
    for b in range(n_buckets):
        info = buckets[b]
        values.append((info["hits"] / info["count"]) if info["count"] else 0.0)
    return {"type": "bucket", "n_buckets": n_buckets, "values": values}


def fit_isotonic(xs, ys):
    pairs = sorted(zip(xs, ys), key=lambda kv: kv[0])
    blocks = []
    for x, y in pairs:
        blocks.append({"w": 1.0, "sum": float(y), "min_x": x, "max_x": x})
        while len(blocks) >= 2:
            y1 = blocks[-2]["sum"] / blocks[-2]["w"]
            y2 = blocks[-1]["sum"] / blocks[-1]["w"]
            if y1 <= y2:
                break
            b2 = blocks.pop()
            b1 = blocks.pop()
            blocks.append(
                {
                    "w": b1["w"] + b2["w"],
                    "sum": b1["sum"] + b2["sum"],
                    "min_x": b1["min_x"],
                    "max_x": b2["max_x"],
                }
            )

    thresholds = []
    values = []
    for b in blocks:
        thresholds.append(b["max_x"])
        values.append(b["sum"] / b["w"])
    return {"type": "isotonic", "thresholds": thresholds, "values": values}


def fit_linear_logistic(features, ys, lr=0.08, epochs=350, l2=1e-3):
    n = len(features[0])
    w = [0.0] * n
    for _ in range(epochs):
        grad = [0.0] * n
        for x, y in zip(features, ys):
            z = sum(wi * xi for wi, xi in zip(w, x))
            p = sigmoid(z)
            err = p - y
            for i in range(n):
                grad[i] += err * x[i]
        for i in range(n):
            grad[i] = grad[i] / len(features) + l2 * w[i]
            w[i] -= lr * grad[i]
    return w


def fit_platt(xs, ys):
    feats = [[1.0, math.log(clamp_prob(x) / (1.0 - clamp_prob(x)))] for x in xs]
    w = fit_linear_logistic(feats, ys)
    return {"type": "platt", "weights": w}


def fit_beta(xs, ys):
    feats = [[1.0, math.log(clamp_prob(x)), math.log(1.0 - clamp_prob(x))] for x in xs]
    w = fit_linear_logistic(feats, ys)
    return {"type": "beta", "weights": w}


def apply_calibrator(calibrator, x):
    t = calibrator["type"]
    if t == "bucket":
        b = bucket_id(x, calibrator["n_buckets"])
        return calibrator["values"][b]
    if t == "isotonic":
        idx = bisect_right(calibrator["thresholds"], x)
        idx = min(idx, len(calibrator["values"]) - 1)
        return calibrator["values"][idx]
    if t == "platt":
        x2 = math.log(clamp_prob(x) / (1.0 - clamp_prob(x)))
        z = calibrator["weights"][0] + calibrator["weights"][1] * x2
        return sigmoid(z)
    if t == "beta":
        z = (
            calibrator["weights"][0]
            + calibrator["weights"][1] * math.log(clamp_prob(x))
            + calibrator["weights"][2] * math.log(1.0 - clamp_prob(x))
        )
        return sigmoid(z)
    return x


def band_of(max_prob):
    for name, (low, high) in BANDS.items():
        if low <= max_prob < high:
            return name
    return "low"


def fit_best_calibrator(xs, ys):
    if not xs:
        return {"type": "identity"}, {"brier": None}
    if len(set(ys)) == 1:
        constant = float(ys[0])
        return {"type": "constant", "value": constant}, {"brier": 0.0, "method": "constant"}

    candidates = [fit_bucket(xs, ys), fit_isotonic(xs, ys), fit_platt(xs, ys), fit_beta(xs, ys)]
    best = None
    for cal in candidates:
        preds = [apply_calibrator(cal, x) for x in xs]
        score = brier_score(preds, ys)
        if best is None or score < best["brier"]:
            best = {"calibrator": cal, "brier": score, "method": cal["type"]}
    return best["calibrator"], {"brier": best["brier"], "method": best["method"]}


def build_calibration(historical):
    out = {}
    diag = {}
    for symbol, prob_col, target_col in (("1", "p(1)", "1"), ("X", "p(x)", "X"), ("2", "p(2)", "2")):
        out[symbol] = {}
        diag[symbol] = {}
        for band in BANDS:
            subset = [r for r in historical if band_of(max(r["p(1)"], r["p(x)"], r["p(2)"])) == band]
            xs = [r[prob_col] for r in subset]
            ys = [int(r[target_col]) for r in subset]
            calibrator, info = fit_best_calibrator(xs, ys)
            out[symbol][band] = calibrator
            diag[symbol][band] = {"n": len(xs), **info}
    return out, diag


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

    calibration, calib_diag = build_calibration(historical)

    model = {
        "meta": {
            "raw_weight": 0.60,
            "calibration_weight": 0.40,
            "confidence_bands": BANDS,
            "uncertainty_heuristics": {
                "enabled": True,
                "top3_early_penalty": 0.09,
                "top3_early_window": 6,
                "top1_certainty_bonus": 0.04,
            },
        },
        "soft_targets": preprocessed["soft_targets"],
        "base_hit_rates": preprocessed["base_hit_rates"],
        "calibration": calibration,
        "calibration_diagnostics": calib_diag,
    }
    dump_json(args.output, model)
    LOGGER.info("Treino concluído: %s", args.output)


if __name__ == "__main__":
    main()
