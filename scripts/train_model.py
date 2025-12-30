import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from joblib import dump  # Usando joblib para salvar os modelos

from .feature_engineering import CLASSES, MODEL_FEATURES, PROB_COLUMNS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def _reorder_probabilities(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return probability matrix with columns aligned to ``CLASSES`` order."""
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    missing = [cls for cls in CLASSES if cls not in class_to_index]
    if missing:
        raise ValueError(
            f"Classes esperadas ausentes do modelo: {missing}. Classes disponíveis: {list(classes)}"
        )
    if len(classes) != len(set(classes)):
        raise ValueError("Classes duplicadas detectadas no estimador.")
    return np.column_stack([proba[:, class_to_index[cls]] for cls in CLASSES])


def train(input_file, model_file, scaler_file=None):
    """Train a classifier on processed data and persist artifacts."""
    try:
        logging.info("Carregando os dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        logging.info("Selecionando as features e o target...")
        X = df[MODEL_FEATURES]
        y = df['Resultado']

        logging.info("Dividindo os dados em treino e teste...")
        if "Concurso" in df.columns:
            splitter = GroupShuffleSplit(test_size=0.2, random_state=42, n_splits=1)
            train_idx, test_idx = next(splitter.split(X, y, groups=df["Concurso"]))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        logging.info("Treinando o modelo com calibração de probabilidades (isotonic)...")
        base_model = RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=100,
            min_samples_split=200,
            class_weight='balanced'
        )
        model = CalibratedClassifierCV(
            estimator=base_model,
            method="isotonic",
            cv=5
        )
        model.fit(X_train, y_train)

        logging.info(f"Ordem das classes aprendida pelo modelo: {list(model.classes_)}")

        logging.info("Calculando métricas de avaliação...")
        y_test_proba_raw = model.predict_proba(X_test)
        y_test_proba = _reorder_probabilities(y_test_proba_raw, model.classes_)
        accuracy = model.score(X_test, y_test)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Labels passed.*ordered lexicographically",
                category=UserWarning,
            )
            logloss = log_loss(y_test, y_test_proba, labels=CLASSES)

        true_one_hot = pd.get_dummies(y_test).reindex(columns=CLASSES, fill_value=0).values
        brier = np.mean(np.sum((true_one_hot - y_test_proba) ** 2, axis=1))

        logging.info("Calculando baseline de mercado (argmax das probabilidades das odds)...")
        market_to_result = {
            'P(1)': '1',
            'P(X)': 'X',
            'P(2)': '2'
        }
        market_prob_test = df.loc[X_test.index, PROB_COLUMNS]
        market_preds = market_prob_test.idxmax(axis=1).map(market_to_result)
        market_accuracy = (market_preds == y_test).mean()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Labels passed.*ordered lexicographically",
                category=UserWarning,
            )
            market_logloss = log_loss(
                y_test,
                market_prob_test[PROB_COLUMNS].to_numpy(),
                labels=CLASSES,
            )

        logging.info(f"Acurácia no conjunto de teste: {accuracy:.4f}")
        logging.info(f"Log loss no conjunto de teste: {logloss:.4f}")
        logging.info(f"Brier score (multi-classe) no conjunto de teste: {brier:.4f}")
        logging.info(f"Acurácia baseline do mercado (seco pelas odds): {market_accuracy:.4f}")
        logging.info(f"Log loss baseline do mercado: {market_logloss:.4f}")

        logging.info(f"Salvando o modelo em {model_file}...")
        dump(model, model_file)

        logging.info("Treinamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except ValueError as e:
        logging.error(f"Erro nos dados: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
