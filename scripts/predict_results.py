import logging
from typing import List

import numpy as np
import pandas as pd
from joblib import load  # Para carregar os modelos previamente treinados

from scripts.train_model import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que todas as colunas de features existam e estejam na ordem correta."""
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[FEATURE_COLUMNS]


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
        # Carregando os dados dos jogos futuros
        logging.info("Carregando dados dos jogos futuros...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        # Verificando se as colunas de probabilidades estão presentes
        required_prob_cols: List[str] = ['P(1)', 'P(X)', 'P(2)']
        if not all(col in df.columns for col in required_prob_cols):
            # Caso as colunas de probabilidade não existam, calculá-las a partir das odds
            odds_cols = ['Odds 1', 'Odds X', 'Odds 2']
            if all(col in df.columns for col in odds_cols):
                logging.info("Colunas de probabilidade ausentes. Calculando a partir das odds...")
                df['P(1)'] = 1 / df['Odds 1']
                df['P(X)'] = 1 / df['Odds X']
                df['P(2)'] = 1 / df['Odds 2']

                prob_sum = df['P(1)'] + df['P(X)'] + df['P(2)']
                df['overround'] = prob_sum
                df['P(1)'] /= prob_sum
                df['P(X)'] /= prob_sum
                df['P(2)'] /= prob_sum
            else:
                raise ValueError(
                    f"As colunas de odds {odds_cols} são necessárias para calcular as probabilidades no arquivo {input_file}."
                )

        # Features derivadas de probabilidade
        logging.info("Calculando features adicionais para as probabilidades...")
        probs = df[['P(1)', 'P(X)', 'P(2)']].to_numpy()
        df['pmax'] = probs.max(axis=1)
        sorted_probs = np.sort(probs, axis=1)
        df['gap12'] = sorted_probs[:, 2] - sorted_probs[:, 1]
        epsilon = 1e-10
        df['entropy'] = -np.sum((probs + epsilon) * np.log(probs + epsilon), axis=1)
        if 'overround' not in df.columns:
            df['overround'] = probs.sum(axis=1)

        # Carregando o modelo e o scaler
        logging.info("Carregando modelo e scaler...")
        model = load(model_file)
        scaler = load(scaler_file)

        # Selecionando as features para predição e escalando
        logging.info("Preparando dados para predição...")
        feature_frame = _ensure_feature_columns(df)
        X_future_scaled = scaler.transform(feature_frame)

        # Gerando as predições
        logging.info("Gerando predições...")
        probabilities = model.predict_proba(X_future_scaled)
        predictions = model.predict(X_future_scaled)
        class_labels = list(model.classes_)

        # Adicionando as predições ao DataFrame
        logging.info("Adicionando predições ao DataFrame...")
        df['Probabilidade (1)'] = np.round(probabilities[:, 0], 5)
        df['Probabilidade (X)'] = np.round(probabilities[:, 1], 5)
        df['Probabilidade (2)'] = np.round(probabilities[:, 2], 5)
        df['Seco'] = predictions

        # Adicionando um valor pequeno para evitar problemas com log(0)
        epsilon = 1e-10
        adjusted_probabilities = probabilities + epsilon

        # Calculando a entropia com as probabilidades ajustadas
        logging.info("Calculando entropia para determinar os jogos mais incertos...")
        df['Entropia'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)

        # Heurística de EV para definição dos 5 duplos
        logging.info("Calculando score de EV para seleção de duplos...")
        top_two_sorted = np.sort(probabilities, axis=1)[:, ::-1][:, :2]
        seco_prob = top_two_sorted[:, 0]
        duplo_prob = top_two_sorted.sum(axis=1)
        ganho_marginal = duplo_prob - seco_prob

        # Fator de rateio heurístico
        favoritismo_medio = df['pmax'].mean()
        fator_rateio_heuristico = 1.3 if favoritismo_medio < 0.55 else 1.0
        if 'jackpot_14' in df.columns:
            jackpot_flag = df['jackpot_14'].eq(True)
            fator_rateio = np.where(jackpot_flag, 1.3, fator_rateio_heuristico)
        else:
            fator_rateio = fator_rateio_heuristico

        score_ev = ganho_marginal * fator_rateio
        jogos_duplos_idxs = pd.Series(score_ev).nlargest(5).index
        logging.info(f"Índices selecionados para duplos: {jogos_duplos_idxs.tolist()}")

        # Gerar a coluna "Aposta"
        logging.info("Gerando a coluna de aposta com 9 secos e 5 duplos...")
        df['Aposta'] = df['Seco']  # Copia as apostas secas inicialmente

        # Escolhendo os "duplos" para os 5 jogos mais incertos
        for idx in jogos_duplos_idxs:
            mais_provaveis = probabilities[idx].argsort()[-2:][::-1]  # Duas maiores probabilidades
            escolhas = [class_labels[mais_provaveis[0]], class_labels[mais_provaveis[1]]]
            df.loc[idx, 'Aposta'] = ", ".join(escolhas)

        # Salvando as predições no arquivo
        logging.info(f"Salvando predições no arquivo {output_file}...")
        df.to_csv(output_file, sep=';', index=False)
        logging.info(f"Previsões salvas com sucesso em {output_file}!")
    
    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo não encontrado - {e}")
    except ValueError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
