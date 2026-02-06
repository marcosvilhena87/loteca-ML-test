import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss


@dataclass
class TrainMetrics:
    logloss: float
    market_logloss: float
    logloss_delta_pct: float
    macro_f1: float
    brier_1: float
    brier_x: float
    brier_2: float
    ece: float


def temporal_split(df, validation_split: float) -> Tuple[np.ndarray, np.ndarray]:
    df_sorted = df.sort_values(["Concurso", "Jogo"]) if {"Concurso", "Jogo"}.issubset(df.columns) else df
    n_total = len(df_sorted)
    n_val = max(1, int(n_total * validation_split))
    split_idx = n_total - n_val
    train_idx = df_sorted.index[:split_idx]
    val_idx = df_sorted.index[split_idx:]
    return train_idx, val_idx


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == y_true
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.any():
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += abs(bin_acc - bin_conf) * mask.mean()
    return float(ece)


def train_model(
    df,
    feature_cols: List[str],
    validation_split: float,
    seed: int,
    class_weight: str,
    model_path: str,
) -> Tuple[LogisticRegression, TrainMetrics, Dict[str, float]]:
    train_idx, val_idx = temporal_split(df, validation_split)

    X_train = df.loc[train_idx, feature_cols].to_numpy()
    y_train = df.loc[train_idx, "target"].map({"1": 0, "X": 1, "2": 2}).to_numpy()
    X_val = df.loc[val_idx, feature_cols].to_numpy()
    y_val = df.loc[val_idx, "target"].map({"1": 0, "X": 1, "2": 2}).to_numpy()

    model = LogisticRegression(
        multi_class="multinomial",
        class_weight=class_weight,
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)
    logloss = log_loss(y_val, val_probs, labels=[0, 1, 2])
    market_probs = df.loc[val_idx, ["p1_market", "px_market", "p2_market"]].to_numpy()
    market_logloss = log_loss(y_val, market_probs, labels=[0, 1, 2])
    if market_logloss == 0:
        logloss_delta_pct = 0.0
    else:
        logloss_delta_pct = (market_logloss - logloss) / market_logloss * 100
    macro_f1 = f1_score(y_val, model.predict(X_val), average="macro")
    brier = (val_probs - np.eye(3)[y_val]) ** 2
    brier_1 = brier[:, 0].mean()
    brier_x = brier[:, 1].mean()
    brier_2 = brier[:, 2].mean()
    ece = expected_calibration_error(val_probs, y_val)

    metrics = TrainMetrics(
        logloss=float(logloss),
        market_logloss=float(market_logloss),
        logloss_delta_pct=float(logloss_delta_pct),
        macro_f1=float(macro_f1),
        brier_1=float(brier_1),
        brier_x=float(brier_x),
        brier_2=float(brier_2),
        ece=float(ece),
    )

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "class_map": {"1": 0, "X": 1, "2": 2},
        },
        model_path,
    )

    split_stats = {
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "validation_split": float(validation_split),
    }

    return model, metrics, split_stats


def metrics_to_dict(metrics: TrainMetrics) -> Dict[str, float]:
    return {
        "logloss": metrics.logloss,
        "market_logloss": metrics.market_logloss,
        "logloss_delta_pct": metrics.logloss_delta_pct,
        "macro_f1": metrics.macro_f1,
        "brier_1": metrics.brier_1,
        "brier_x": metrics.brier_x,
        "brier_2": metrics.brier_2,
        "ece": metrics.ece,
    }
