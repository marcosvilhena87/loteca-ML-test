import logging
from typing import List, Sequence

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


FEATURE_COLUMNS: List[str] = [
    "P_market(1)", "P_market(X)", "P_market(2)",
    "P_elo(1)", "P_elo(X)", "P_elo(2)",
    "P_pois(1)", "P_pois(X)", "P_pois(2)",
    "d_elo_1", "d_elo_X", "d_elo_2",
    "d_pois_1", "d_pois_X", "d_pois_2",
    "bookmaker_margin", "gap_market", "entropia_market", "draw_boost",
    "elo_diff", "elo_uncertainty_home", "elo_uncertainty_away",
    "form_home", "form_away", "form_diff",
]


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


def train(input_file, model_file, scaler_file=None):
    """Train a stacked meta-model on engineered features and persist artifacts."""
    try:
        logging.info("Carregando os dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            raise KeyError(f"Features ausentes do conjunto processado: {missing_features}")

        logging.info("Separando treino e validação por ordem temporal com embargo...")
        train_df, val_df = _time_ordered_split(df, test_size=0.2, embargo_contests=1)
        X_train, X_val = train_df[FEATURE_COLUMNS], val_df[FEATURE_COLUMNS]
        y_train, y_val = train_df['Resultado'], val_df['Resultado']

        logging.info("Montando pipeline de mistura (logistic regression multiclasse)...")
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ])

        model.fit(X_train, y_train)

        logging.info("Avaliando métricas probabilísticas...")
        val_proba_raw = model.predict_proba(X_val)
        labels = model.classes_
        class_order = np.array(['1', 'X', '2'])

        val_proba = _reorder_probas(val_proba_raw, labels, class_order)
        logloss_model = log_loss(y_val, val_proba, labels=class_order)
        brier = _multiclass_brier(y_val, val_proba, class_order)

        market_cols = [f"P_market({c})" for c in class_order]
        val_market = val_df[market_cols].values
        logloss_market = log_loss(y_val, val_market, labels=class_order)
        delta = logloss_market - logloss_model

        logging.info(f"Log Loss (modelo) no conjunto de validação: {logloss_model:.4f}")
        logging.info(f"Log Loss (mercado) no conjunto de validação: {logloss_market:.4f}")
        logging.info(f"Delta vs. mercado (positivo = ganho): {delta:.4f}")
        logging.info(f"Brier Score no conjunto de validação: {brier:.4f}")

        entropia_final = -np.sum(val_proba * np.log(val_proba + 1e-12), axis=1)
        top_two = np.sort(val_proba, axis=1)[:, ::-1][:, :2]
        gap_final = top_two[:, 0] - top_two[:, 1]

        def _duplo_hit_rate(alpha: float, top_k: int = 50) -> float:
            scores = entropia_final + alpha * (1 - gap_final)
            k = min(top_k, len(scores))
            selected = np.argsort(scores)[::-1][:k]
            hits = 0
            for idx in selected:
                top2_idx = val_proba[idx].argsort()[::-1][:2]
                if y_val.iloc[idx] in class_order[top2_idx]:
                    hits += 1
            return hits / k if k else 0.0

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            hit_rate = _duplo_hit_rate(alpha)
            logging.info(f"Hit-rate top-2 para duplos (alpha={alpha:.2f}): {hit_rate:.3f}")

        logging.info(f"Salvando o modelo em {model_file}...")
        dump(model, model_file)

        if scaler_file:
            dump(None, scaler_file)

        logging.info("Treinamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except KeyError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
