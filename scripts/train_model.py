import logging
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import label_binarize

from scripts.features import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def _reorder_probabilities(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    ordem_desejada = ['1', 'X', '2']
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    ordered = np.zeros_like(probabilities)
    for pos, label in enumerate(ordem_desejada):
        ordered[:, pos] = probabilities[:, class_to_idx[label]]
    return ordered


def _evaluate_draw_thresholds(probabilities: np.ndarray, classes: np.ndarray, y_true: pd.Series, thresholds) -> None:
    ordered = _reorder_probabilities(probabilities, classes)
    best_threshold = None
    best_accuracy = -np.inf

    for threshold in thresholds:
        preds_idx = ordered.argmax(axis=1)
        adjusted = []
        for pred, prob_row in zip(preds_idx, ordered):
            if pred == 1 and prob_row[1] < threshold:
                fallback = 0 if prob_row[0] >= prob_row[2] else 2
                adjusted.append(fallback)
            else:
                adjusted.append(pred)
        preds = np.array(['1', 'X', '2'])[adjusted]
        acc = accuracy_score(y_true, preds)
        logging.info(f"Acurácia com draw_threshold={threshold:.2f}: {acc:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    logging.info(
        "Melhor draw_threshold observado: %.2f (acurácia %.4f)",
        best_threshold,
        best_accuracy,
    )


def train(input_file, model_file, scaler_file):
    """Treina o classificador com as novas features e persiste os artefatos."""
    try:
        logging.info("Carregando os dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        logging.info("Selecionando as features e o target...")
        X = df[FEATURE_COLUMNS]
        y = df['Resultado']

        logging.info("Dividindo os dados em treino e teste por concurso...")
        if 'Concurso' not in df.columns:
            raise KeyError("Coluna 'Concurso' é necessária para o group split.")

        groups = df['Concurso']
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        logging.info("Criando pipeline de imputação...")
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
        ])
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        logging.info("Treinando o modelo de floresta aleatória com calibração isotônica...")
        base_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None)
        calibrator = CalibratedClassifierCV(
            estimator=base_model,
            method='isotonic',
            cv=GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
        )
        calibrator.fit(X_train_processed, y_train, groups=groups.iloc[train_idx])

        y_proba = calibrator.predict_proba(X_test_processed)
        y_pred = calibrator.predict(X_test_processed)

        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_proba, labels=calibrator.classes_)

        y_true_bin = label_binarize(y_test, classes=calibrator.classes_)
        brier = float(np.mean(np.sum((y_proba - y_true_bin) ** 2, axis=1)))
        brier_avg = brier / len(calibrator.classes_)
        logging.info(f"Acurácia (sanity check) no conjunto de teste: {accuracy:.4f}")
        logging.info(f"LogLoss no conjunto de teste: {logloss:.4f}")
        logging.info(f"Brier Score no conjunto de teste: {brier:.4f}")
        logging.info(f"Brier Score médio por classe: {brier_avg:.4f}")

        logging.info("Testando thresholds para empates (draw_threshold)...")
        _evaluate_draw_thresholds(
            probabilities=y_proba,
            classes=calibrator.classes_,
            y_true=y_test,
            thresholds=[0.25, 0.30, 0.35, 0.40],
        )

        logging.info(f"Salvando o modelo em {model_file} e o pré-processador em {scaler_file}...")
        dump(calibrator, model_file)
        dump(preprocessor, scaler_file)
        logging.info("Treinamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except ValueError as e:
        logging.error(f"Erro nos dados: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
