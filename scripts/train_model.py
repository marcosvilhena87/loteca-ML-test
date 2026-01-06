import logging
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import label_binarize

from scripts.features import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


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

        logging.info("Treinando o modelo de floresta aleatória...")
        model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None)
        model.fit(X_train_processed, y_train)

        y_proba = model.predict_proba(X_test_processed)
        y_pred = model.predict(X_test_processed)

        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_proba, labels=model.classes_)

        y_true_bin = label_binarize(y_test, classes=model.classes_)
        brier = float(np.mean(np.sum((y_proba - y_true_bin) ** 2, axis=1)))
        logging.info(f"Acurácia (sanity check) no conjunto de teste: {accuracy:.4f}")
        logging.info(f"LogLoss no conjunto de teste: {logloss:.4f}")
        logging.info(f"Brier Score no conjunto de teste: {brier:.4f}")

        logging.info(f"Salvando o modelo em {model_file} e o pré-processador em {scaler_file}...")
        dump(model, model_file)
        dump(preprocessor, scaler_file)
        logging.info("Treinamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except ValueError as e:
        logging.error(f"Erro nos dados: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
