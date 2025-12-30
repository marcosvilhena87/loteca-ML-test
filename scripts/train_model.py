import logging
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump  # Usando joblib para salvar os modelos

from .feature_engineering import MODEL_FEATURES

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def train(input_file, model_file, scaler_file):
    """Train a classifier on processed data and persist artifacts."""
    try:
        logging.info("Carregando os dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        logging.info("Selecionando as features e o target...")
        X = df[MODEL_FEATURES]
        y = df['Resultado']

        logging.info("Dividindo os dados em treino e teste...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logging.info("Escalando as features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logging.info("Treinando o modelo com calibração de probabilidades (sigmoid)...")
        base_model = RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=50,
            min_samples_split=100,
            class_weight='balanced'
        )
        model = CalibratedClassifierCV(
            base_estimator=base_model,
            method="sigmoid",
            cv=5
        )
        model.fit(X_train_scaled, y_train)

        logging.info(f"Ordem das classes aprendida pelo modelo: {list(model.classes_)}")

        logging.info("Calculando métricas de avaliação...")
        y_test_proba = model.predict_proba(X_test_scaled)
        accuracy = model.score(X_test_scaled, y_test)
        logloss = log_loss(y_test, y_test_proba, labels=model.classes_)

        true_one_hot = pd.get_dummies(y_test).reindex(columns=model.classes_, fill_value=0).values
        brier = np.mean(np.sum((true_one_hot - y_test_proba) ** 2, axis=1))

        logging.info(f"Acurácia no conjunto de teste: {accuracy:.4f}")
        logging.info(f"Log loss no conjunto de teste: {logloss:.4f}")
        logging.info(f"Brier score (multi-classe) no conjunto de teste: {brier:.4f}")

        logging.info(f"Salvando o modelo em {model_file} e o scaler em {scaler_file}...")
        dump(model, model_file)
        dump(scaler, scaler_file)

        logging.info("Treinamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except ValueError as e:
        logging.error(f"Erro nos dados: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
