import logging
import pandas as pd
import numpy as np
from joblib import load  # Para carregar os modelos previamente treinados

from .feature_engineering import (
    CLASSES,
    MODEL_FEATURES,
    PROB_COLUMNS,
    add_domain_features,
    compute_probabilities,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def _reorder_probabilities(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return probability matrix with columns aligned to ``CLASSES`` order."""
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    missing = [cls for cls in CLASSES if cls not in class_to_index]
    if missing:
        raise ValueError(
            f"Classes esperadas ausentes do modelo: {missing}. Classes disponíveis: {list(classes)}"
        )
    return np.column_stack([proba[:, class_to_index[cls]] for cls in CLASSES])


def predict(input_file, model_file, scaler_file, output_file):
    """Generate predictions for future games and write them to CSV."""
    try:
        logging.info("Carregando dados dos jogos futuros...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        logging.info("Calculando probabilidades das partidas...")
        df = compute_probabilities(df)

        logging.info("Criando features específicas da Loteca...")
        df = add_domain_features(df)

        logging.info("Carregando modelo...")
        model = load(model_file)

        logging.info(f"Ordem das classes do modelo: {list(model.classes_)}")

        logging.info("Preparando dados para predição...")
        X_future = df[MODEL_FEATURES]

        logging.info("Gerando predições...")
        probabilities = _reorder_probabilities(model.predict_proba(X_future), model.classes_)

        logging.info("Adicionando predições ao DataFrame com mapeamento correto das classes...")
        df['Probabilidade (1)'] = np.round(probabilities[:, 0], 5)
        df['Probabilidade (X)'] = np.round(probabilities[:, 1], 5)
        df['Probabilidade (2)'] = np.round(probabilities[:, 2], 5)

        market_top = df[PROB_COLUMNS].to_numpy().argmax(axis=1)
        df['Seco_Mercado'] = [CLASSES[idx] for idx in market_top]

        predictions_mapped = [CLASSES[idx] for idx in probabilities.argmax(axis=1)]
        df['Seco_Modelo'] = predictions_mapped

        epsilon = 1e-10
        adjusted_probabilities = probabilities + epsilon

        logging.info("Calculando entropia para determinar os jogos mais incertos...")
        df['Entropia'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)
        df['Pmax_Modelo'] = probabilities.max(axis=1)
        df['Score_Duplo'] = df['Entropia'] * (1 - df['Pmax_Modelo'])

        candidatos = df[(df['Pmax_Modelo'] <= 0.60) & (df['Probabilidade (X)'] >= 0.22)]
        if len(candidatos) < 5:
            candidatos = df[(df['Pmax_Modelo'] <= 0.70) & (df['Probabilidade (X)'] >= 0.18)]
        if len(candidatos) < 5:
            candidatos = df

        jogos_duplos_idxs = candidatos.nlargest(5, 'Score_Duplo').index
        logging.info(f"Índices dos jogos mais incertos para duplos: {jogos_duplos_idxs.tolist()}")

        logging.info("Gerando a coluna de aposta com 9 secos e 5 duplos...")
        df['Aposta'] = df['Seco_Mercado']

        df['Duplo_Modelo'] = ""

        duplo_opcoes = ['1', 'X', '2']
        for idx in jogos_duplos_idxs:
            mais_provaveis = probabilities[idx].argsort()[-2:][::-1]
            duplos_ordenados = sorted(
                (duplo_opcoes[mais_provaveis[0]], duplo_opcoes[mais_provaveis[1]]),
                key=['1', 'X', '2'].index,
            )
            duplo = ", ".join(duplos_ordenados)
            df.loc[idx, 'Aposta'] = duplo
            df.loc[idx, 'Duplo_Modelo'] = duplo

        logging.info(f"Salvando predições no arquivo {output_file}...")
        df.to_csv(output_file, sep=';', index=False)
        logging.info(f"Previsões salvas com sucesso em {output_file}!")

    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo não encontrado - {e}")
    except ValueError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
