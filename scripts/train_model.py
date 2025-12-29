import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold, train_test_split

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

FEATURE_COLUMNS: List[str] = [
    'P(1)',
    'P(X)',
    'P(2)',
    'pmax',
    'gap12',
    'entropy',
    'overround',
    'h2h_m_has',
    'h2h_m_w5',
    'h2h_m_e5',
    'h2h_m_d5',
    'h2h_m_pts5',
    'h2h_m_last',
    'h2h_v_has',
    'h2h_v_w5',
    'h2h_v_e5',
    'h2h_v_d5',
    'h2h_v_pts5',
    'h2h_v_last',
    'h2h_pts_diff',
    'h2h_has_both',
]


def _temporal_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by ``Concurso`` ordering to avoid leakage."""
    if 'Concurso' not in df.columns:
        logging.warning(
            "Coluna 'Concurso' ausente; usando split estratificado aleatório como fallback."
        )
        return train_test_split(df, test_size=test_size, random_state=42, stratify=df['Resultado'])

    concursos_ordenados = df['Concurso'].dropna().sort_values().unique()
    if len(concursos_ordenados) == 0:
        logging.warning(
            "Nenhum valor válido em 'Concurso'; usando split estratificado aleatório como fallback."
        )
        return train_test_split(df, test_size=test_size, random_state=42, stratify=df['Resultado'])

    n_test = max(1, int(len(concursos_ordenados) * test_size))
    concursos_teste = concursos_ordenados[-n_test:]
    df_teste = df[df['Concurso'].isin(concursos_teste)]
    df_treino = df[~df['Concurso'].isin(concursos_teste)]

    if df_treino.empty or df_teste.empty:
        logging.warning(
            "Split temporal resultou em conjunto vazio; usando split estratificado aleatório como fallback."
        )
        return train_test_split(df, test_size=test_size, random_state=42, stratify=df['Resultado'])

    logging.info(
        "Split temporal: %s concursos para treino, %s para teste.",
        df_treino['Concurso'].nunique(),
        df_teste['Concurso'].nunique(),
    )
    return df_treino, df_teste


def _multiclass_brier(y_true: pd.Series, proba: np.ndarray, class_labels: List[str]) -> float:
    """Compute multiclass Brier score as mean squared error between probs and one-hot labels."""
    y_true_onehot = pd.get_dummies(y_true).reindex(columns=class_labels, fill_value=0).to_numpy()
    return float(np.mean(np.sum((proba - y_true_onehot) ** 2, axis=1)))


def train(input_file, model_file):
    """Train a classifier on processed data and persist a single pipeline.

    Parameters
    ----------
    input_file : str
        CSV file containing training features and labels.
    model_file : str
        Path to save the fitted pipeline.

    Returns
    -------
        None
        The trained pipeline is written to disk.
    """
    try:
        # Carregando os dados
        logging.info("Carregando os dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        # Garantir que todas as colunas de features existam
        logging.info("Validando colunas de features...")
        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            raise KeyError(f"Colunas ausentes nos dados processados: {missing_features}")

        # Selecionando as features e o target (resultado real)
        logging.info("Selecionando as features e o target...")
        y = df['Resultado']  # Target: 1, X ou 2
        X = df[FEATURE_COLUMNS]

        if X.isna().any().any():
            na_counts = X.isna().sum()
            raise ValueError(
                "Foram encontrados valores NaN nas features, revise o preprocessamento: "
                f"{na_counts[na_counts > 0].to_dict()}"
            )

        # Dividindo os dados em treino e teste
        logging.info("Dividindo os dados em treino e teste sem vazamento temporal...")
        df_train, df_test = _temporal_train_test_split(df)
        X_train, y_train = df_train[FEATURE_COLUMNS], df_train['Resultado']
        X_test, y_test = df_test[FEATURE_COLUMNS], df_test['Resultado']

        def _log_metrics(description: str, y_true, proba, class_labels: List[str]):
            logloss_val = log_loss(y_true, proba, labels=class_labels)
            brier_val = _multiclass_brier(y_true, proba, class_labels)
            preds = [class_labels[idx] for idx in np.argmax(proba, axis=1)]
            accuracy_val = accuracy_score(y_true, preds)
            logging.info("[%s] Log loss: %.4f", description, logloss_val)
            logging.info("[%s] Brier score: %.4f", description, brier_val)
            logging.info("[%s] Acurácia (referência): %.4f", description, accuracy_val)
            return logloss_val, brier_val, accuracy_val

        # Baseline usando probabilidades derivadas das odds (sem modelo)
        logging.info("Calculando baseline A (odds puras)...")
        baseline_labels = ['1', '2', 'X']
        baseline_proba = df_test[['P(1)', 'P(2)', 'P(X)']].to_numpy()
        _log_metrics("Baseline odds", y_test, baseline_proba, baseline_labels)

        # Modelo de comparação: RandomForest (sem imputação)
        logging.info("Treinando RandomForest para comparação...")
        rf_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None)
        rf_model.fit(X_train, y_train)
        rf_proba_test = rf_model.predict_proba(X_test)
        rf_labels = list(rf_model.classes_)
        _log_metrics("RandomForest", y_test, rf_proba_test, rf_labels)

        # Modelo principal: Regressão Logística Multinomial (sem calibração)
        logging.info("Treinando Regressão Logística Multinomial sem calibração...")
        log_reg = LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            solver='lbfgs',
        )
        log_reg.fit(X_train, y_train)
        log_reg_proba_test = log_reg.predict_proba(X_test)
        log_reg_labels = list(log_reg.classes_)
        _log_metrics("LogisticRegression", y_test, log_reg_proba_test, log_reg_labels)

        # Calibração opcional com Platt (sigmoid) respeitando grupos de Concurso
        calibrated_model = None
        if 'Concurso' in df_train.columns and df_train['Concurso'].nunique(dropna=True) >= 2:
            n_groups = df_train['Concurso'].nunique(dropna=True)
            n_splits = min(3, n_groups)
            if n_splits >= 2:
                logging.info(
                    "Treinando calibração Platt (sigmoid) com GroupKFold (%s splits) por Concurso...",
                    n_splits,
                )
                group_cv = GroupKFold(n_splits=n_splits)
                calibrated_model = CalibratedClassifierCV(
                    estimator=LogisticRegression(max_iter=1000, n_jobs=-1, solver='lbfgs'),
                    method='sigmoid',
                    cv=group_cv,
                    n_jobs=-1,
                )
                calibrated_model.fit(X_train, y_train, groups=df_train['Concurso'])
                calibrated_proba_test = calibrated_model.predict_proba(X_test)
                calibrated_labels = list(calibrated_model.classes_)
                _log_metrics(
                    "LogisticRegression + calibração sigmoid (GroupKFold)",
                    y_test,
                    calibrated_proba_test,
                    calibrated_labels,
                )
            else:
                logging.warning(
                    "Calibração sigmoid ignorada: são necessários ao menos 2 grupos distintos em 'Concurso'.",
                )
        elif 'Concurso' not in df_train.columns:
            logging.warning(
                "Calibração sigmoid ignorada: coluna 'Concurso' ausente para definição de grupos.",
            )
        else:
            logging.warning(
                "Calibração sigmoid ignorada: menos de 2 concursos distintos disponíveis.",
            )

        # Salvando o melhor modelo disponível
        model_to_save = calibrated_model if calibrated_model is not None else log_reg
        logging.info("Salvando modelo em %s...", model_file)
        dump(model_to_save, model_file)

        logging.info("Treinamento concluído com sucesso!")
    
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except ValueError as e:
        logging.error(f"Erro nos dados: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
