import logging
from typing import List

import numpy as np
import pandas as pd
from joblib import load

from scripts.train_model import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que todas as colunas de features existam e estejam na ordem correta."""
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas ausentes para predição: {missing_cols}")

    feature_frame = df[FEATURE_COLUMNS]
    if feature_frame.isna().any().any():
        na_counts = feature_frame.isna().sum()
        raise ValueError(
            "Foram encontrados valores NaN nas features de predição, revise os dados de entrada: "
            f"{na_counts[na_counts > 0].to_dict()}"
        )

    return feature_frame


def predict(input_file, model_file, output_file):
    """Generate predictions for future games and write them to CSV.

    Parameters
    ----------
    input_file : str
        CSV file with upcoming games and odds or probabilities.
    model_file : str
        Path of the trained model to load.
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

        # Carregando o modelo calibrado
        logging.info("Carregando modelo de predição...")
        model = load(model_file)

        # Selecionando as features para predição e escalando
        logging.info("Preparando dados para predição...")
        feature_frame = _ensure_feature_columns(df)

        # Gerando as predições
        logging.info("Gerando predições...")
        probabilities = model.predict_proba(feature_frame)
        predictions = model.predict(feature_frame)
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

        # Fator de payout contínuo baseado em incerteza e rollover
        entropy_norm = df['entropy'] / np.log(3)
        underdog_bonus = (0.6 - df['pmax']).clip(lower=0) * 0.8
        volatilidade_bonus = entropy_norm * 0.5
        payout_prior = 1.0 + underdog_bonus + volatilidade_bonus

        if 'rollover_streak' in df.columns:
            payout_prior *= 1 + 0.03 * df['rollover_streak'].fillna(0).clip(upper=20)
        if 'log_rateio_14' in df.columns:
            median_rateio = df['log_rateio_14'].median()
            payout_prior *= 1 + 0.05 * (df['log_rateio_14'].fillna(median_rateio) - median_rateio)

        score_ev = ganho_marginal * payout_prior
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
