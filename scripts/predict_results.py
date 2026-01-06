import logging
import pandas as pd
import numpy as np
from joblib import load  # Para carregar os modelos previamente treinados

from scripts.preprocess_data import (
    FEATURE_COLUMNS,
    add_common_features,
    build_team_history,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def predict(input_file, model_file, scaler_file, output_file):
    """Generate predictions for future games and write them to CSV.

    Parameters
    ----------
    input_file : str
        CSV file with upcoming games and odds or probabilities.
    model_file : str
        Path of the trained model to load.
    scaler_file : str
        Path of the scaler used during training.
    output_file : str
        Destination path for the predictions CSV.

    Returns
    -------
    None
        The predictions are saved to ``output_file``.
    """
    try:
        logging.info("Carregando dados dos jogos futuros...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        logging.info("Construindo histórico dos times a partir dos concursos anteriores...")
        historical_df = pd.read_csv("data/raw/concursos_anteriores.csv", delimiter=';', decimal='.')
        history = build_team_history(historical_df)

        logging.info("Gerando o mesmo conjunto de features do treinamento...")
        df_features = add_common_features(df, base_history=history)

        required_columns = FEATURE_COLUMNS
        
        # Carregando o modelo e o scaler
        logging.info("Carregando modelo e scaler...")
        model = load(model_file)
        scaler = load(scaler_file)

        # Selecionando as features para predição e escalando
        logging.info("Preparando dados para predição...")
        X_future = df_features[required_columns]
        X_future_scaled = scaler.transform(X_future)

        # Gerando as predições
        logging.info("Gerando predições...")
        probabilities = model.predict_proba(X_future_scaled)
        predictions = model.predict(X_future_scaled)

        # Adicionando as predições ao DataFrame
        logging.info("Adicionando predições ao DataFrame...")
        df_features['Probabilidade (1)'] = np.round(probabilities[:, 0], 5)
        df_features['Probabilidade (X)'] = np.round(probabilities[:, 1], 5)
        df_features['Probabilidade (2)'] = np.round(probabilities[:, 2], 5)
        df_features['Secos'] = predictions

        # Adicionando um valor pequeno para evitar problemas com log(0)
        epsilon = 1e-10
        adjusted_probabilities = probabilities + epsilon

        # Calculando a entropia com as probabilidades ajustadas
        logging.info("Calculando entropia para determinar os jogos mais incertos...")
        df_features['Entropia'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)

        # Identificar os 5 jogos mais incertos para aplicar os "duplos"
        jogos_duplos_idxs = df_features.nlargest(5, 'Entropia').index
        logging.info(f"Índices dos jogos mais incertos para duplos: {jogos_duplos_idxs.tolist()}")

        # Gerar a coluna "Aposta"
        logging.info("Gerando a coluna de aposta com 9 secos e 5 duplos...")
        df_features['Aposta'] = df_features['Secos']  # Copia as apostas secas inicialmente

        # Escolhendo os "duplos" para os 5 jogos mais incertos
        duplo_opcoes = ['1', 'X', '2']
        for idx in jogos_duplos_idxs:
            mais_provaveis = probabilities[idx].argsort()[-2:][::-1]  # Duas maiores probabilidades
            df_features.loc[idx, 'Aposta'] = f"{duplo_opcoes[mais_provaveis[0]]}, {duplo_opcoes[mais_provaveis[1]]}"

        # Salvando as predições no arquivo
        logging.info(f"Salvando predições no arquivo {output_file}...")
        df_features.to_csv(output_file, sep=';', index=False)
        logging.info(f"Previsões salvas com sucesso em {output_file}!")
    
    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo não encontrado - {e}")
    except ValueError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
