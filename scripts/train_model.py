import logging
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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

        logging.info("Dividindo os dados em treino e teste...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logging.info("Criando pipeline de imputação e padronização...")
        scaler = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logging.info("Treinando o modelo de floresta aleatória...")
        model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None)
        model.fit(X_train_scaled, y_train)

        accuracy = model.score(X_test_scaled, y_test)
        logging.info(f"Acurácia no conjunto de teste: {accuracy:.4f}")

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
