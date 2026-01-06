import logging
import warnings
from typing import Dict, Iterable, Tuple

import pandas as pd
from joblib import dump  # Usando joblib para salvar os modelos
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from scripts.preprocess_data import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def _temporal_train_test_split(
    df: pd.DataFrame, target_col: str, test_size: float = 0.2
):
    """Perform a chronological train/test split based on the Concurso order.

    The split is performed on full "Concurso" blocks to avoid leaking games
    from the same contest across train and test.
    """

    if "Concurso" not in df.columns:
        logging.warning(
            "Coluna 'Concurso' ausente; utilizando a ordem atual para o split temporal."
        )
        split_index = int(len(df) * (1 - test_size))
        if split_index == 0 or split_index == len(df):
            raise ValueError(
                "Dataset pequeno demais para realizar um split temporal confiável."
            )

        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]
    else:
        df = df.sort_values("Concurso").reset_index(drop=True)
        concursos = pd.unique(df["Concurso"])
        split_index = int(len(concursos) * (1 - test_size))

        if split_index == 0 or split_index == len(concursos):
            raise ValueError(
                "Dataset pequeno demais para realizar um split temporal confiável."
            )

        train_concursos = set(concursos[:split_index])
        test_concursos = set(concursos[split_index:])

        train_df = df[df["Concurso"].isin(train_concursos)]
        test_df = df[df["Concurso"].isin(test_concursos)]

    return (
        train_df.drop(columns=[target_col]),
        test_df.drop(columns=[target_col]),
        train_df[target_col],
        test_df[target_col],
    )


def _evaluate_model(
    model: CalibratedClassifierCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_order: Iterable[str],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    raw_proba = model.predict_proba(X_test)
    proba_df = pd.DataFrame(raw_proba, columns=model.classes_)
    missing_classes = [label for label in class_order if label not in proba_df.columns]
    if missing_classes:
        raise ValueError(
            f"O modelo não retornou probabilidades para as classes: {missing_classes}"
        )

    y_proba = proba_df[list(class_order)].to_numpy()
    y_true_bin = pd.get_dummies(y_test).reindex(columns=class_order, fill_value=0).to_numpy()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Labels passed were.*ordered lexicographically",
            category=UserWarning,
        )
        logloss = log_loss(y_test, y_proba, labels=list(class_order))

    # Brier normalizado: média dos erros quadráticos por classe
    brier = ((y_proba - y_true_bin) ** 2).mean()
    accuracy = model.score(X_test, y_test)

    metrics = {
        "logloss": logloss,
        "brier": brier,
        "accuracy": accuracy,
    }
    return metrics, proba_df


def train(input_file, model_file, calibration_methods: Iterable[str] = ("isotonic", "sigmoid")):
    """Train a classifier on processed data and persist artifacts.

    Parameters
    ----------
    input_file : str
        CSV file containing training features and labels.
    model_file : str
        Path to save the fitted model.

    Returns
    -------
    None
        The trained model is written to disk.
    """
    try:
        # Carregando os dados
        logging.info("Carregando os dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        # Selecionando as features (probabilidades e contexto) e o target (resultado real)
        logging.info("Selecionando as features e o target...")
        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            raise KeyError(
                f"As seguintes colunas de feature estão ausentes no dataset: {missing_features}"
            )

        split_columns = FEATURE_COLUMNS + ['Resultado']
        if 'Concurso' in df.columns:
            split_columns = ['Concurso'] + split_columns

        # Dividindo os dados em treino e teste de forma temporal
        logging.info("Dividindo os dados em treino e teste (split temporal)...")
        X_train, X_test, y_train, y_test = _temporal_train_test_split(df[split_columns], 'Resultado')
        X_train = X_train.drop(columns=['Concurso'], errors='ignore')
        X_test = X_test.drop(columns=['Concurso'], errors='ignore')

        class_order = ["1", "X", "2"]

        # Treinando e avaliando múltiplas calibrações
        metrics_by_method = {}
        best_method = None
        best_model = None

        for method in calibration_methods:
            logging.info("Treinando o modelo com calibração '%s'...", method)
            base_model = RandomForestClassifier(
                random_state=42, n_estimators=100, max_depth=None
            )
            calibrated_model = CalibratedClassifierCV(base_model, method=method, cv=3)
            calibrated_model.fit(X_train, y_train)

            metrics, _ = _evaluate_model(calibrated_model, X_test, y_test, class_order)
            metrics_by_method[method] = metrics

            logging.info(
                "[%s] LogLoss: %.4f | Brier: %.4f | Acurácia: %.4f",
                method,
                metrics["logloss"],
                metrics["brier"],
                metrics["accuracy"],
            )

            if best_method is None:
                best_method = method
                best_model = calibrated_model
                continue

            current_best = metrics_by_method[best_method]
            if metrics["logloss"] < current_best["logloss"] or (
                metrics["logloss"] == current_best["logloss"]
                and metrics["accuracy"] > current_best["accuracy"]
            ):
                best_method = method
                best_model = calibrated_model

        best_metrics = metrics_by_method[best_method]
        logging.info(
            "Melhor calibração: %s — LogLoss: %.4f | Brier: %.4f | Acurácia: %.4f",
            best_method,
            best_metrics["logloss"],
            best_metrics["brier"],
            best_metrics["accuracy"],
        )

        # Salvando o modelo vencedor
        logging.info(f"Salvando o modelo em {model_file}...")
        dump(best_model, model_file)

        logging.info("Treinamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except ValueError as e:
        logging.error(f"Erro nos dados: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
