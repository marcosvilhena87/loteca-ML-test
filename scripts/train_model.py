import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from joblib import dump  # Usando joblib para salvar os modelos

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def train(input_file, model_file):
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

        # Selecionando as features (probabilidades) e o target (resultado real)
        logging.info("Selecionando as features e o target...")
        X = df[['P(1)', 'P(X)', 'P(2)']]  # Features
        y = df['Resultado']  # Target: 1, X ou 2

        # Dividindo os dados em treino e teste
        logging.info("Dividindo os dados em treino e teste...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Treinando o modelo
        logging.info("Treinando o modelo...")
        model = LogisticRegression(
            random_state=42,
            solver="lbfgs",
            max_iter=1000,
        )
        model.fit(X_train, y_train)

        # Avaliando o modelo
        probabilities = model.predict_proba(X_test)
        predictions = model.predict(X_test)
        classes = model.classes_
        accuracy = accuracy_score(y_test, predictions)
        logloss = log_loss(y_test, probabilities, labels=classes)
        y_true = (
            pd.get_dummies(y_test)
            .reindex(columns=classes, fill_value=0)
            .to_numpy()
        )
        brier_score = np.mean(np.sum((probabilities - y_true) ** 2, axis=1))
        conf_matrix = confusion_matrix(y_test, predictions, labels=classes)
        logging.info(f"Acurácia no conjunto de teste: {accuracy:.4f}")
        logging.info(f"Log loss no conjunto de teste: {logloss:.4f}")
        logging.info(f"Brier score no conjunto de teste: {brier_score:.4f}")
        logging.info("Matriz de confusão no conjunto de teste:")
        logging.info("\n%s", conf_matrix)

        # Salvando o modelo
        logging.info(f"Salvando o modelo em {model_file}...")
        dump(model, model_file)

        logging.info("Treinamento concluído com sucesso!")
    
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except ValueError as e:
        logging.error(f"Erro nos dados: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
