import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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
        Path to save the fitted pipeline (imputer + model).

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
        logging.info("Preparando colunas de features...")
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                logging.warning("Coluna %s ausente nos dados. Preenchendo com NaN.", col)
                df[col] = np.nan

        # Selecionando as features e o target (resultado real)
        logging.info("Selecionando as features e o target...")
        y = df['Resultado']  # Target: 1, X ou 2

        # Dividindo os dados em treino e teste
        logging.info("Dividindo os dados em treino e teste sem vazamento temporal...")
        df_train, df_test = _temporal_train_test_split(df)
        X_train, y_train = df_train[FEATURE_COLUMNS], df_train['Resultado']
        X_test, y_test = df_test[FEATURE_COLUMNS], df_test['Resultado']

        # Pipeline de imputação + modelo
        logging.info("Ajustando pipeline (imputer + RandomForest)...")
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None)),
        ])
        pipeline.fit(X_train, y_train)

        # Treinando o modelo
        model = pipeline.named_steps['model']
        logging.info("Modelo treinado com %s árvores.", model.n_estimators)

        # Avaliando o modelo
        proba_test = pipeline.predict_proba(X_test)
        class_labels = list(model.classes_)
        logloss = log_loss(y_test, proba_test, labels=class_labels)
        brier = _multiclass_brier(y_test, proba_test, class_labels)
        accuracy = accuracy_score(y_test, pipeline.predict(X_test))
        logging.info("Log loss no conjunto de teste: %.4f", logloss)
        logging.info("Brier score no conjunto de teste: %.4f", brier)
        logging.info("Acurácia no conjunto de teste (referência): %.4f", accuracy)

        # Salvando o modelo e o scaler
        logging.info("Salvando pipeline completo em %s...", model_file)
        dump(pipeline, model_file)

        logging.info("Treinamento concluído com sucesso!")
    
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except ValueError as e:
        logging.error(f"Erro nos dados: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
