import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


FEATURE_VARIANTS: Dict[str, List[str]] = {
    "market_plus_deltas": [
        "P_market(1)", "P_market(X)", "P_market(2)",
        "d_elo_1", "d_elo_X", "d_elo_2",
        "d_pois_1", "d_pois_X", "d_pois_2",
        "bookmaker_margin", "gap_market", "entropia_market", "draw_boost",
        "elo_diff", "elo_uncertainty_home", "elo_uncertainty_away",
        "form_home", "form_away", "form_diff",
    ],
    "pure_probabilities": [
        "P_market(1)", "P_market(X)", "P_market(2)",
        "P_elo(1)", "P_elo(X)", "P_elo(2)",
        "P_pois(1)", "P_pois(X)", "P_pois(2)",
        "bookmaker_margin", "gap_market", "entropia_market",
        "elo_diff", "elo_uncertainty_home", "elo_uncertainty_away",
        "form_home", "form_away", "form_diff",
    ],
    "full_mix": [
        "P_market(1)", "P_market(X)", "P_market(2)",
        "P_elo(1)", "P_elo(X)", "P_elo(2)",
        "P_pois(1)", "P_pois(X)", "P_pois(2)",
        "d_elo_1", "d_elo_X", "d_elo_2",
        "d_pois_1", "d_pois_X", "d_pois_2",
        "bookmaker_margin", "gap_market", "entropia_market", "draw_boost",
        "elo_diff", "elo_uncertainty_home", "elo_uncertainty_away",
        "form_home", "form_away", "form_diff",
    ],
}

DEFAULT_FEATURE_VARIANT = "market_plus_deltas"
DEFAULT_C_VALUES = (0.1, 0.3, 1.0, 3.0, 10.0)
CLASS_ORDER = np.array(['1', 'X', '2'])


def get_feature_columns(variant: str = DEFAULT_FEATURE_VARIANT) -> List[str]:
    if variant not in FEATURE_VARIANTS:
        raise KeyError(f"Feature variant desconhecida: {variant}. Opções: {list(FEATURE_VARIANTS)}")
    return FEATURE_VARIANTS[variant]


FEATURE_COLUMNS: List[str] = get_feature_columns()


def _time_ordered_split(df: pd.DataFrame, test_size: float = 0.2, embargo_contests: int = 1):
    df_sorted = df.sort_values(["Concurso", "Jogo"])
    split_index = int(len(df_sorted) * (1 - test_size))
    val_start_contest = df_sorted.iloc[split_index]["Concurso"]

    unique_contests = pd.Index(sorted(df_sorted["Concurso"].unique()))
    val_start_idx = unique_contests.get_loc(val_start_contest)
    embargo_cut = max(val_start_idx - embargo_contests, 0)

    train_contests = unique_contests[:embargo_cut]
    val_contests = unique_contests[val_start_idx:]

    train_df = df_sorted[df_sorted["Concurso"].isin(train_contests)]
    val_df = df_sorted[df_sorted["Concurso"].isin(val_contests)]

    logging.info(
        "Split temporal com embargo: %s concursos treino, %s concursos validação (embargo=%s)",
        len(train_contests), len(val_contests), embargo_contests,
    )
    return train_df, val_df


def _multiclass_brier(y_true: pd.Series, probas: np.ndarray, classes: np.ndarray) -> float:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_indices = np.array([class_to_idx[label] for label in y_true])
    y_onehot = np.eye(len(classes))[y_indices]
    return float(np.mean(np.sum((y_onehot - probas) ** 2, axis=1)))


def _reorder_probas(probas: np.ndarray, source_classes: Sequence[str], target_classes: Sequence[str]) -> np.ndarray:
    """Reorder probability columns to match the target class order."""

    index_map = [list(source_classes).index(cls) for cls in target_classes]
    return probas[:, index_map]


def train(input_file, model_file, scaler_file=None, feature_variant: str = DEFAULT_FEATURE_VARIANT,
          c_values: Optional[Sequence[float]] = None):
    """Train a stacked meta-model on engineered features and persist artifacts."""
    try:
        logging.info("Carregando os dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        feature_columns = get_feature_columns(feature_variant)
        logging.info("Usando variant de features '%s' (%d colunas)", feature_variant, len(feature_columns))
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise KeyError(f"Features ausentes do conjunto processado: {missing_features}")

        logging.info("Separando treino e validação por ordem temporal com embargo...")
        train_df, val_df = _time_ordered_split(df, test_size=0.2, embargo_contests=1)
        X_train, X_val = train_df[feature_columns], val_df[feature_columns]
        y_train, y_val = train_df['Resultado'], val_df['Resultado']

        logging.info("Montando pipeline de mistura (logistic regression multiclasse)...")

        def _fit_and_score(C_value: float):
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, C=C_value)),
            ])
            pipeline.fit(X_train, y_train)
            val_proba_raw = pipeline.predict_proba(X_val)
            labels = pipeline.classes_
            val_proba = _reorder_probas(val_proba_raw, labels, CLASS_ORDER)
            logloss_model = log_loss(y_val, val_proba, labels=CLASS_ORDER)
            brier_score = _multiclass_brier(y_val, val_proba, CLASS_ORDER)
            return {
                "model": pipeline,
                "val_proba": val_proba,
                "logloss_model": logloss_model,
                "brier": brier_score,
            }

        c_grid = list(dict.fromkeys(c_values if c_values is not None else DEFAULT_C_VALUES))
        if not c_grid:
            raise ValueError("A grade de valores de C não pode ser vazia.")
        market_cols = [f"P_market({c})" for c in CLASS_ORDER]
        val_market = val_df[market_cols].values
        logloss_market = log_loss(y_val, val_market, labels=CLASS_ORDER)

        best_run = None
        for c_val in c_grid:
            run = _fit_and_score(c_val)
            delta = logloss_market - run["logloss_model"]
            logging.info(
                "C=%.3g | LogLoss model=%.4f | delta vs mercado=%.4f | Brier=%.4f",
                c_val, run["logloss_model"], delta, run["brier"],
            )
            if best_run is None or run["logloss_model"] < best_run["logloss_model"]:
                best_run = {"C": c_val, **run}

        assert best_run is not None
        logging.info(
            "Melhor C=%.3g com LogLoss=%.4f (mercado=%.4f, delta=%.4f)",
            best_run["C"], best_run["logloss_model"], logloss_market, logloss_market - best_run["logloss_model"],
        )
        logging.info(f"Brier Score no conjunto de validação: {best_run['brier']:.4f}")

        entropia_final = -np.sum(best_run["val_proba"] * np.log(best_run["val_proba"] + 1e-12), axis=1)
        top_two = np.sort(best_run["val_proba"], axis=1)[:, ::-1][:, :2]
        gap_final = top_two[:, 0] - top_two[:, 1]

        def _duplo_hit_rate(alpha: float, top_k: int = 50) -> float:
            scores = entropia_final + alpha * (1 - gap_final)
            k = min(top_k, len(scores))
            selected = np.argsort(scores)[::-1][:k]
            hits = 0
            for idx in selected:
                top2_idx = best_run["val_proba"][idx].argsort()[::-1][:2]
                if y_val.iloc[idx] in CLASS_ORDER[top2_idx]:
                    hits += 1
            return hits / k if k else 0.0

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            hit_rate = _duplo_hit_rate(alpha)
            logging.info(f"Hit-rate top-2 para duplos (alpha={alpha:.2f}): {hit_rate:.3f}")

        logging.info(f"Salvando o modelo em {model_file} (features={feature_variant})...")
        artifact = {
            "model": best_run["model"],
            "feature_variant": feature_variant,
            "feature_columns": feature_columns,
        }
        dump(artifact, model_file)

        if scaler_file:
            dump(None, scaler_file)

        logging.info("Treinamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except KeyError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
