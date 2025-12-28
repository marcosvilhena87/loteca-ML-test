import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
        model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None)
        model.fit(X_train, y_train)

        # Avaliando o modelo
        accuracy = model.score(X_test, y_test)
        logging.info(f"Acurácia no conjunto de teste: {accuracy:.4f}")

        # Baselines para comparação
        prob_cols = ['P(1)', 'P(X)', 'P(2)']
        bookmaker_guess = X_test[prob_cols].idxmax(axis=1).map({
            'P(1)': '1',
            'P(X)': 'X',
            'P(2)': '2'
        })
        bookmaker_accuracy = (bookmaker_guess == y_test).mean()
        logging.info(f"Baseline bookmaker (argmax das probabilidades): {bookmaker_accuracy:.4f}")

        majority_class = y_train.value_counts().idxmax()
        majority_accuracy = (y_test == majority_class).mean()
        logging.info(f"Baseline classe majoritária ({majority_class}): {majority_accuracy:.4f}")

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
