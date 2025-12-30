import logging
from typing import List

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
    "bookmaker_margin", "gap_market", "entropia_market", "draw_boost",
    "elo_diff", "elo_uncertainty", "form_elo_lastN"
]


def _time_ordered_split(df: pd.DataFrame, test_size: float = 0.2):
    df_sorted = df.sort_values(["Concurso", "Jogo"])
    split_index = int(len(df_sorted) * (1 - test_size))
    return df_sorted.iloc[:split_index], df_sorted.iloc[split_index:]


def _multiclass_brier(y_true: pd.Series, probas: np.ndarray, classes: np.ndarray) -> float:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_indices = np.array([class_to_idx[label] for label in y_true])
    y_onehot = np.eye(len(classes))[y_indices]
    return float(np.mean(np.sum((y_onehot - probas) ** 2, axis=1)))


def train(input_file, model_file, scaler_file=None):
    """Train a stacked meta-model on engineered features and persist artifacts."""
    try:
        logging.info("Carregando os dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            raise KeyError(f"Features ausentes do conjunto processado: {missing_features}")

        logging.info("Separando treino e validação por ordem temporal...")
        train_df, val_df = _time_ordered_split(df, test_size=0.2)
        X_train, X_val = train_df[FEATURE_COLUMNS], val_df[FEATURE_COLUMNS]
        y_train, y_val = train_df['Resultado'], val_df['Resultado']

        logging.info("Montando pipeline de mistura (logistic regression multinomial)...")
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, multi_class='multinomial', class_weight='balanced')),
        ])

        model.fit(X_train, y_train)

        logging.info("Avaliando métricas probabilísticas...")
        val_proba = model.predict_proba(X_val)
        labels = model.classes_

        logloss = log_loss(y_val, val_proba, labels=labels)
        brier = _multiclass_brier(y_val, val_proba, labels)
        logging.info(f"Log Loss no conjunto de validação: {logloss:.4f}")
        logging.info(f"Brier Score no conjunto de validação: {brier:.4f}")

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
