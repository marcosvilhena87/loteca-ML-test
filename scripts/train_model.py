import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump  # Usando joblib para salvar os modelos
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_engineering import (
    CLASSES,
    CORRECTOR_FORM_FEATURES,
    CORRECTOR_MARKET_FEATURES,
    MARKET_CORRECTOR_FEATURES,
    MODEL_FEATURES,
    PROB_COLUMNS,
)

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


def _brier_score_multi(y_true: pd.Series, proba: np.ndarray) -> float:
    """Compute the multi-class Brier score."""

    true_one_hot = (
        pd.get_dummies(y_true)
        .reindex(columns=CLASSES, fill_value=0)
        .values
    )
    return float(np.mean(np.sum((true_one_hot - proba) ** 2, axis=1)))


def _log_metrics(prefix: str, accuracy: float, logloss: float, brier: float) -> None:
    logging.info(f"{prefix} acurácia no conjunto de teste: {accuracy:.4f}")
    logging.info(f"{prefix} log loss no conjunto de teste: {logloss:.4f}")
    logging.info(f"{prefix} Brier score (multi-classe): {brier:.4f}")


def _evaluate_probabilities(
    prefix: str, y_true: pd.Series, proba: np.ndarray
) -> tuple[float, float, float]:
    """Compute metrics and log them for a given probability matrix."""

    preds = np.asarray([CLASSES[idx] for idx in proba.argmax(axis=1)])
    accuracy = float((preds == y_true.to_numpy()).mean())

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Labels passed.*ordered lexicographically",
            category=UserWarning,
        )
        logloss = log_loss(y_true, proba, labels=CLASSES)

    brier = _brier_score_multi(y_true, proba)
    _log_metrics(prefix, accuracy, logloss, brier)
    return accuracy, logloss, brier


def train(input_file, model_file, scaler_file=None, corrector_file=None):
    """Train classifiers on processed data and persist artifacts.

    Parameters
    ----------
    input_file : str
        Caminho para o CSV processado.
    model_file : str
        Caminho de saída para o modelo calibrado principal.
    scaler_file : str | None
        Reservado para compatibilidade; não utilizado atualmente.
    corrector_file : str | None
        Caminho opcional para salvar o modelo "corretor do mercado".
    """
    try:
        logging.info("Carregando os dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        logging.info("Selecionando as features e o target...")
        X = df[MODEL_FEATURES]
        y = df['Resultado']

        logging.info("Dividindo os dados em treino e teste...")
        if "Concurso" in df.columns:
            concursos_numeric = pd.to_numeric(df["Concurso"], errors="coerce")
            unique_concursos = concursos_numeric.dropna().astype(int).unique()
            concursos = pd.Index(np.sort(unique_concursos))

            if len(concursos) >= 2:
                full_range = np.arange(concursos.min(), concursos.max() + 1)
                missing_concursos = np.setdiff1d(full_range, concursos)
                if len(missing_concursos) > 0:
                    logging.warning(
                        "Concursos faltantes detectados no intervalo %s-%s: %s",
                        concursos.min(),
                        concursos.max(),
                        missing_concursos.tolist(),
                    )
                else:
                    logging.info(
                        "Nenhum concurso faltante detectado no intervalo temporal avaliado."
                    )

                n_test = max(1, int(np.ceil(len(concursos) * 0.2)))
                train_concursos = concursos[:-n_test]
                test_concursos = concursos[-n_test:]

                train_idx = df.index[concursos_numeric.isin(train_concursos)]
                test_idx = df.index[concursos_numeric.isin(test_concursos)]
                logging.info(
                    "Split temporal por concurso. Treinando em %s e testando em %s",
                    train_concursos.tolist(),
                    test_concursos.tolist(),
                )
            else:
                logging.warning(
                    "Apenas um concurso disponível; usando divisão estratificada padrão."
                )
                train_idx, test_idx = train_test_split(
                    df.index, test_size=0.2, random_state=42, stratify=y
                )
        else:
            train_idx, test_idx = train_test_split(
                df.index, test_size=0.2, random_state=42, stratify=y
            )

        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

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
        accuracy, logloss, brier = _evaluate_probabilities(
            "Modelo principal", y_test, y_test_proba
        )

        logging.info("Calculando baseline de mercado (argmax das probabilidades das odds)...")
        market_prob_test = df.loc[X_test.index, PROB_COLUMNS]
        market_accuracy, market_logloss, market_brier = _evaluate_probabilities(
            "Baseline do mercado", y_test, market_prob_test[PROB_COLUMNS].to_numpy()
        )

        logging.info("Treinando modelo corretor do mercado (LogisticRegression pura)...")
        corrector_transformer = ColumnTransformer(
            [
                ("form_scaler", StandardScaler(), CORRECTOR_FORM_FEATURES),
                (
                    "pass_probabilities",
                    "passthrough",
                    CORRECTOR_MARKET_FEATURES,
                ),
            ]
        )
        corrector_pipeline = Pipeline([
            ("preprocess", corrector_transformer),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=500,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ])

        corrector_X = df[MARKET_CORRECTOR_FEATURES]
        corrector_X_train, corrector_X_test = (
            corrector_X.loc[train_idx],
            corrector_X.loc[test_idx],
        )

        # Versão não calibrada (recomendada)
        corrector_pipeline.fit(corrector_X_train, y_train)
        corrector_proba_raw = corrector_pipeline.predict_proba(corrector_X_test)
        corrector_proba = _reorder_probabilities(
            corrector_proba_raw, corrector_pipeline.classes_
        )
        corrector_accuracy, corrector_logloss, corrector_brier = _evaluate_probabilities(
            "Modelo corretor do mercado (sem calibração)", y_test, corrector_proba
        )

        # Misturas usando o corretor sem calibração (opcional para produção)
        blended_alphas = [0.2, 0.4, 0.6, 0.8]
        for alpha in blended_alphas:
            blended_proba = alpha * corrector_proba + (1 - alpha) * market_prob_test[PROB_COLUMNS].to_numpy()
            _evaluate_probabilities(
                f"Corretor sem calibração com mistura mercado (alpha={alpha:.1f})",
                y_test,
                blended_proba,
            )

        # Experimento: calibração sigmoid (Platt scaling)
        logging.info("Treinando versão calibrada com sigmoid para comparação...")
        sigmoid_corrector = CalibratedClassifierCV(
            estimator=corrector_pipeline,
            method="sigmoid",
            cv=5,
        )
        sigmoid_corrector.fit(corrector_X_train, y_train)
        sigmoid_proba_raw = sigmoid_corrector.predict_proba(corrector_X_test)
        sigmoid_proba = _reorder_probabilities(
            sigmoid_proba_raw, sigmoid_corrector.classes_
        )
        _evaluate_probabilities(
            "Modelo corretor do mercado (calibrado sigmoid)", y_test, sigmoid_proba
        )

        logging.info(f"Salvando o modelo em {model_file}...")
        dump(model, model_file)

        if corrector_file is None:
            model_path = Path(model_file)
            corrector_file = model_path.with_name(
                f"{model_path.stem}_market_corrector{model_path.suffix}"
            )

        logging.info(f"Salvando o modelo corretor em {corrector_file}...")
        dump(corrector_pipeline, corrector_file)

        logging.info("Treinamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except ValueError as e:
        logging.error(f"Erro nos dados: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
