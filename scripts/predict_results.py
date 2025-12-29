import logging
import pandas as pd
import numpy as np
from joblib import load  # Para carregar os modelos previamente treinados

from .feature_engineering import (
    MODEL_FEATURES,
    add_domain_features,
    compute_probabilities,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def predict(input_file, model_file, scaler_file, output_file):
    """Generate predictions for future games and write them to CSV."""
    try:
        logging.info("Carregando dados dos jogos futuros...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        logging.info("Calculando probabilidades das partidas...")
        df = compute_probabilities(df)

        logging.info("Criando features específicas da Loteca...")
        df = add_domain_features(df)

        logging.info("Carregando modelo e scaler...")
        model = load(model_file)
        scaler = load(scaler_file)

        logging.info("Preparando dados para predição...")
        X_future = df[MODEL_FEATURES]
        X_future_scaled = scaler.transform(X_future)

        logging.info("Gerando predições...")
        probabilities = model.predict_proba(X_future_scaled)
        predictions = model.predict(X_future_scaled)

        logging.info("Adicionando predições ao DataFrame...")
        df['Probabilidade (1)'] = np.round(probabilities[:, 0], 5)
        df['Probabilidade (X)'] = np.round(probabilities[:, 1], 5)
        df['Probabilidade (2)'] = np.round(probabilities[:, 2], 5)
        df['Seco'] = predictions

        epsilon = 1e-10
        adjusted_probabilities = probabilities + epsilon

        logging.info("Calculando entropia para determinar os jogos mais incertos...")
        df['Entropia'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)

        jogos_duplos_idxs = df.nlargest(5, 'Entropia').index
        logging.info(f"Índices dos jogos mais incertos para duplos: {jogos_duplos_idxs.tolist()}")

        logging.info("Gerando a coluna de aposta com 9 secos e 5 duplos...")
        df['Aposta'] = df['Seco']

        duplo_opcoes = ['1', 'X', '2']
        for idx in jogos_duplos_idxs:
            mais_provaveis = probabilities[idx].argsort()[-2:][::-1]
            df.loc[idx, 'Aposta'] = f"{duplo_opcoes[mais_provaveis[0]]}, {duplo_opcoes[mais_provaveis[1]]}"

        logging.info(f"Salvando predições no arquivo {output_file}...")
        df.to_csv(output_file, sep=';', index=False)
        logging.info(f"Previsões salvas com sucesso em {output_file}!")

    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo não encontrado - {e}")
    except ValueError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
