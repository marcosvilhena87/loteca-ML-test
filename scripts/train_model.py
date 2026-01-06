import logging
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.preprocessing import label_binarize

from scripts.features import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def _reorder_probabilities(
    probabilities: np.ndarray,
    classes: np.ndarray,
    order: tuple[str, str, str] = ('1', 'X', '2'),
) -> np.ndarray:
    """Reordena as probabilidades seguindo a ordem desejada."""
    ordem_desejada = list(order)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    ordered = np.zeros_like(probabilities)
    for pos, label in enumerate(ordem_desejada):
        if label not in class_to_idx:
            raise ValueError("O modelo não contém todas as classes esperadas ('1', 'X', '2').")
        ordered[:, pos] = probabilities[:, class_to_idx[label]]
    return ordered


def _compute_brier_score(y_true: pd.Series, probabilities: np.ndarray, classes: np.ndarray) -> float:
    y_true_bin = label_binarize(y_true, classes=classes)
    return float(np.mean(np.sum((probabilities - y_true_bin) ** 2, axis=1)))


def _create_calibrated_classifier() -> CalibratedClassifierCV:
    base_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None)
    calibrator = CalibratedClassifierCV(
        estimator=base_model,
        method='isotonic',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    )
    return calibrator


def _evaluate_draw_thresholds(probabilities: np.ndarray, classes: np.ndarray, y_true: pd.Series, thresholds):
    ordered = _reorder_probabilities(probabilities, classes)
    class_order = np.array(['1', 'X', '2'])

    # LogLoss/Brier "verdadeiros" (probabilidades originais, sem hacks)
    base_logloss = log_loss(y_true, np.clip(ordered, 1e-15, 1 - 1e-15), labels=class_order)
    base_brier = _compute_brier_score(y_true, ordered, class_order)

    best = None
    for threshold in thresholds:
        preds_idx = ordered.argmax(axis=1)
        preds = class_order[preds_idx].copy()

        # Ajuste igual ao predict_results.py: só troca o SECO quando previu X com prob baixa
        mask = (preds == 'X') & (ordered[:, 1] < threshold)
        if mask.any():
            # fallback: escolhe 1 se P(1) >= P(2) senão 2
            fallback = np.where(ordered[:, 0] >= ordered[:, 2], '1', '2')
            preds[mask] = fallback[mask]

        acc = accuracy_score(y_true, preds)

        # Métricas focadas em empate (opcional, mas útil)
        y_true_x = (y_true.values == 'X')
        y_pred_x = (preds == 'X')
        prec_x = precision_score(y_true_x, y_pred_x, zero_division=0)
        rec_x = recall_score(y_true_x, y_pred_x, zero_division=0)
        f1_x = f1_score(y_true_x, y_pred_x, zero_division=0)

        logging.info(
            "Threshold %.2f => Acurácia %.4f | PrecX %.4f | RecX %.4f | F1X %.4f | (LogLoss fixo %.4f | Brier fixo %.4f)",
            threshold, acc, prec_x, rec_x, f1_x, base_logloss, base_brier
        )

        # escolha do melhor threshold: maximize acurácia, desempata por F1 do X (ou rec_x, você decide)
        cand = {'threshold': threshold, 'accuracy': acc, 'f1_x': f1_x, 'prec_x': prec_x, 'rec_x': rec_x}
        if (best is None) or (cand['accuracy'] > best['accuracy']) or (
            np.isclose(cand['accuracy'], best['accuracy']) and cand['f1_x'] > best['f1_x']
        ):
            best = cand

    if best:
        logging.info(
            "Melhor draw_threshold por decisão: %.2f (Acurácia %.4f | F1X %.4f | PrecX %.4f | RecX %.4f)",
            best['threshold'], best['accuracy'], best['f1_x'], best['prec_x'], best['rec_x']
        )

    return best


def temporal_backtest(df: pd.DataFrame, block_size: int = 50) -> pd.DataFrame:
    """Executa backtest temporal em blocos de concursos ordenados."""

    if 'Concurso' not in df.columns:
        raise KeyError("Coluna 'Concurso' é necessária para o backtest temporal.")

    concursos = sorted(df['Concurso'].unique())
    if len(concursos) <= block_size:
        logging.info(
            "Backtest temporal não executado: são necessários pelo menos %d concursos (encontrados %d).",
            block_size + 1,
            len(concursos),
        )
        return pd.DataFrame()

    resultados = []
    class_order = np.array(['1', 'X', '2'])

    for start in range(0, len(concursos) - block_size, block_size):
        train_concursos = concursos[:start + block_size]
        test_concursos = concursos[start + block_size: start + 2 * block_size]

        if not test_concursos:
            continue

        train_df = df[df['Concurso'].isin(train_concursos)]
        test_df = df[df['Concurso'].isin(test_concursos)]

        if train_df.empty or test_df.empty:
            continue

        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
        ])

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df['Resultado']
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df['Resultado']

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        calibrator = _create_calibrated_classifier()
        calibrator.fit(X_train_processed, y_train)

        y_proba = calibrator.predict_proba(X_test_processed)
        probas_ordered = _reorder_probabilities(y_proba, calibrator.classes_, tuple(class_order))
        y_pred = calibrator.predict(X_test_processed)

        acc = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, probas_ordered, labels=class_order)
        brier = _compute_brier_score(y_test, probas_ordered, class_order)

        resultado_bloco = {
            'train_range': f"{train_concursos[0]}-{train_concursos[-1]}",
            'test_range': f"{test_concursos[0]}-{test_concursos[-1]}",
            'amostras_teste': len(test_df),
            'acuracia': acc,
            'logloss': logloss,
            'brier': brier,
        }
        resultados.append(resultado_bloco)

        logging.info(
            "Backtest %s => %s | Acurácia %.4f | LogLoss %.4f | Brier %.4f",
            resultado_bloco['train_range'],
            resultado_bloco['test_range'],
            acc,
            logloss,
            brier,
        )

    return pd.DataFrame(resultados)


def train(input_file, model_file, scaler_file, run_backtest: bool = False):
    """Treina o classificador com as novas features e persiste os artefatos.

    Parâmetros
    ----------
    run_backtest: bool
        Quando verdadeiro, executa o backtest temporal completo (mais demorado).
    """
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
        calibrator = _create_calibrated_classifier()
        calibrator.fit(X_train_processed, y_train)

        y_proba = calibrator.predict_proba(X_test_processed)
        metrics_order = np.array(['1', '2', 'X'])
        probas_for_metrics = _reorder_probabilities(y_proba, calibrator.classes_, tuple(metrics_order))
        y_pred = calibrator.predict(X_test_processed)

        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, probas_for_metrics, labels=metrics_order)

        brier = _compute_brier_score(y_test, probas_for_metrics, metrics_order)
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

        if run_backtest:
            logging.info("Rodando backtest temporal em blocos de 50 concursos...")
            backtest_df = temporal_backtest(df, block_size=50)
            if not backtest_df.empty:
                logging.info("Resumo do backtest temporal:\n%s", backtest_df.to_string(index=False))

        logging.info(f"Salvando o modelo em {model_file} e o pré-processador em {scaler_file}...")
        dump(calibrator, model_file)
        dump(preprocessor, scaler_file)
        logging.info("Treinamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
        raise
    except ValueError as e:
        logging.error(f"Erro nos dados: {e}")
        raise
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
        raise
