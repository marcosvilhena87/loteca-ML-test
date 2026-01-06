import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump  # Usando joblib para salvar os modelos

from scripts.preprocess_data import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def _temporal_train_test_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2):
    """Perform a chronological train/test split based on the Concurso order."""

    if "Concurso" in df.columns:
        df = df.sort_values("Concurso").reset_index(drop=True)
    else:
        logging.warning(
            "Coluna 'Concurso' ausente; utilizando a ordem atual para o split temporal."
        )

    split_index = int(len(df) * (1 - test_size))
    if split_index == 0 or split_index == len(df):
        raise ValueError("Dataset pequeno demais para realizar um split temporal confiável.")

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    return train_df.drop(columns=[target_col]), test_df.drop(columns=[target_col]), train_df[target_col], test_df[target_col]


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

        # Treinando o modelo
        logging.info("Treinando o modelo...")
        model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None)
        model.fit(X_train, y_train)

        # Avaliando o modelo
        accuracy = model.score(X_test, y_test)
        logging.info(f"Acurácia no conjunto de teste: {accuracy:.4f}")

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
