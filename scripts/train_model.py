import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, brier_score_loss
from joblib import dump  # Usando joblib para salvar os modelos

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def train(input_file, model_file, scaler_file):
    """Train a classifier on processed data and persist artifacts.

    Parameters
    ----------
    input_file : str
        CSV file containing training features and labels.
    model_file : str
        Path to save the fitted model.
    scaler_file : str
        Path to save the fitted scaler.

    Returns
    -------
    None
        The trained model and scaler are written to disk.
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

        # Escalando as features
        logging.info("Escala não aplicada (árvores não requerem padronização).")
        scaler = None
        X_train_scaled = X_train
        X_test_scaled = X_test

        # Treinando o modelo
        logging.info("Treinando o modelo...")
        model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None)
        model.fit(X_train_scaled, y_train)

        # Avaliando o modelo
        accuracy = model.score(X_test_scaled, y_test)
        logging.info(f"Acurácia no conjunto de teste: {accuracy:.4f}")

        # Baseline: sempre escolher a maior probabilidade implícita pelas odds
        baseline_mapping = {'P(1)': '1', 'P(X)': 'X', 'P(2)': '2'}
        baseline_pred = X_test.idxmax(axis=1).map(baseline_mapping)
        baseline_accuracy = (baseline_pred == y_test).mean()
        logging.info(f"Acurácia do baseline (maior probabilidade implícita): {baseline_accuracy:.4f}")

        # Métricas baseadas em probabilidade
        proba = model.predict_proba(X_test_scaled)
        model_log_loss = log_loss(y_test, proba, labels=model.classes_)
        logging.info(f"Log loss no conjunto de teste: {model_log_loss:.4f}")

        for class_index, class_label in enumerate(model.classes_):
            class_true = (y_test == class_label).astype(int)
            class_brier = brier_score_loss(class_true, proba[:, class_index])
            logging.info(f"Brier score para a classe {class_label}: {class_brier:.4f}")

        # Salvando o modelo e o scaler
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
