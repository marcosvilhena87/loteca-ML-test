import logging
import os

import pandas as pd
import numpy as np
from joblib import load  # Para carregar os modelos previamente treinados

from scripts.feature_engineering import FEATURE_COLUMNS, build_features
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

SYMBOLS_ORDER = ["1", "X", "2"]
CONF_ALTA = 0.52
GAP_ALTO = 0.10
MAX_DIVERGENCIA = 3


def parse_pick(value):
    if pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    parts = [part.strip().upper() for part in text.split(",")]
    return {part for part in parts if part in SYMBOLS_ORDER}


def format_pick(pick_set):
    if not pick_set:
        return ""
    ordered = [symbol for symbol in SYMBOLS_ORDER if symbol in pick_set]
    return ", ".join(ordered)


def build_complementar_aposta(probabilities, entropias, bolao_series):
    h_alta = np.percentile(entropias, 70)
    apostas = []
    apostas_sets = []
    divergencias = []

    for idx, probs in enumerate(probabilities):
        sorted_idx = np.argsort(probs)[::-1]
        top1_idx, top2_idx = sorted_idx[:2]
        top1_symbol = SYMBOLS_ORDER[top1_idx]
        top2_symbol = SYMBOLS_ORDER[top2_idx]
        pmax = probs[top1_idx]
        gap = pmax - probs[top2_idx]

        bolao_set = bolao_series.iloc[idx]
        pick_set = set()

        if len(bolao_set) >= 2:
            pick_set = {top1_symbol}
        elif len(bolao_set) == 1:
            bolao_symbol = next(iter(bolao_set))
            if top1_symbol == bolao_symbol:
                pick_set = {top1_symbol}
            elif pmax >= CONF_ALTA and gap >= GAP_ALTO:
                pick_set = {top1_symbol}
            else:
                pick_set = {top1_symbol, top2_symbol}
        else:
            pick_set = {top1_symbol}

        apostas_sets.append(pick_set)
        apostas.append(format_pick(pick_set))
        divergencias.append(len(bolao_set & pick_set) == 0)

    excesso = sum(divergencias) - MAX_DIVERGENCIA
    if excesso > 0:
        candidatos = [
            (idx, probabilities[idx].max(), entropias[idx])
            for idx, diverge in enumerate(divergencias)
            if diverge and bolao_series.iloc[idx]
        ]
        candidatos.sort(key=lambda item: (item[1], -item[2]))
        for idx, _, _ in candidatos[:excesso]:
            bolao_set = bolao_series.iloc[idx]
            pick_set = set(apostas_sets[idx])
            pick_set |= bolao_set
            apostas_sets[idx] = pick_set
            apostas[idx] = format_pick(pick_set)

    return apostas, apostas_sets, h_alta

def predict(
    input_file,
    model_file,
    output_file,
    bolao_file="data/raw/boloes.csv",
    bolao_column="Aposta1",
    cartoes_output=None,
):
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
        required_prob_cols = ['P(1)', 'P(X)', 'P(2)']
        if not all(col in df.columns for col in required_prob_cols):
            # Caso as colunas de probabilidade não existam, calculá-las a partir das odds
            odds_cols = ['Odds_1', 'Odds_X', 'Odds_2']
            if all(col in df.columns for col in odds_cols):
                logging.info("Colunas de probabilidade ausentes. Calculando a partir das odds...")
                df['P(1)'] = 1 / df['Odds_1']
                df['P(X)'] = 1 / df['Odds_X']
                df['P(2)'] = 1 / df['Odds_2']

                prob_sum = df['P(1)'] + df['P(X)'] + df['P(2)']
                df['P(1)'] /= prob_sum
                df['P(X)'] /= prob_sum
                df['P(2)'] /= prob_sum
            else:
                raise ValueError(
                    f"As colunas de odds {odds_cols} são necessárias para calcular as probabilidades no arquivo {input_file}."
                )

        logging.info("Gerando features...")
        df = build_features(df)

        # Carregando o modelo
        logging.info("Carregando modelo...")
        model = load(model_file)

        # Selecionando as features para predição e escalando
        logging.info("Preparando dados para predição...")
        X_future = df[FEATURE_COLUMNS]

        # Gerando as predições
        logging.info("Gerando predições...")
        probabilities = model.predict_proba(X_future)
        predictions = model.predict(X_future)

        # Adicionando as predições ao DataFrame
        logging.info("Adicionando predições ao DataFrame...")
        df['Probabilidade (1)'] = np.round(probabilities[:, 0], 5)
        df['Probabilidade (X)'] = np.round(probabilities[:, 1], 5)
        df['Probabilidade (2)'] = np.round(probabilities[:, 2], 5)        
        df['Secos'] = predictions

        # Adicionando um valor pequeno para evitar problemas com log(0)
        epsilon = 1e-10
        adjusted_probabilities = probabilities + epsilon

        # Calculando a entropia com as probabilidades ajustadas
        logging.info("Calculando entropia para determinar os jogos mais incertos...")
        df['Entropia'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)

        # Identificar os 5 jogos mais incertos para aplicar os "duplos"
        jogos_duplos_idxs = df.nlargest(5, 'Entropia').index
        logging.info(f"Índices dos jogos mais incertos para duplos: {jogos_duplos_idxs.tolist()}")

        # Gerar a coluna "Aposta"
        logging.info("Gerando a coluna de aposta com 9 secos e 5 duplos...")
        df['Aposta'] = df['Secos']  # Copia as apostas secas inicialmente

        # Escolhendo os "duplos" para os 5 jogos mais incertos
        duplo_opcoes = ['1', 'X', '2']
        for idx in jogos_duplos_idxs:
            mais_provaveis = probabilities[idx].argsort()[-2:][::-1]  # Duas maiores probabilidades
            df.loc[idx, 'Aposta'] = f"{duplo_opcoes[mais_provaveis[0]]}, {duplo_opcoes[mais_provaveis[1]]}"

        bolao_series = None
        if bolao_file and os.path.exists(bolao_file):
            logging.info("Carregando bolões para aposta complementar...")
            bolao_df = pd.read_csv(bolao_file, delimiter=';', decimal='.', encoding="utf-8-sig")
            if bolao_column not in bolao_df.columns:
                raise ValueError(
                    f"Coluna '{bolao_column}' não encontrada em {bolao_file}."
                )
            bolao_series = bolao_df[bolao_column].reset_index(drop=True)
            if len(bolao_series) < len(df):
                bolao_series = bolao_series.reindex(range(len(df)))
            else:
                bolao_series = bolao_series.iloc[:len(df)]
            df["Bolao_Aposta"] = bolao_series
            bolao_sets = bolao_series.apply(parse_pick)

            logging.info("Gerando apostas complementares ao bolão...")
            complementar_apostas, complementar_sets, h_alta = build_complementar_aposta(
                probabilities, df["Entropia"].to_numpy(), bolao_sets
            )
            df["Aposta_Modelo_Complementar"] = complementar_apostas
            df["Uniao_Bolao_Modelo"] = [
                format_pick(bolao_sets.iloc[idx] | complementar_sets[idx])
                for idx in range(len(df))
            ]

            divergencias = [
                len(bolao_sets.iloc[idx] & complementar_sets[idx]) == 0
                for idx in range(len(df))
            ]
            redundancias = [
                bolao_sets.iloc[idx] == complementar_sets[idx]
                for idx in range(len(df))
            ]
            logging.info(
                "Resumo complementar: divergências=%s redundâncias=%s H_alta=%.4f",
                sum(divergencias),
                sum(redundancias),
                h_alta,
            )

            if cartoes_output is None:
                output_dir = os.path.dirname(output_file) or "."
                cartoes_output = os.path.join(output_dir, "cartoes.csv")

            cartoes_cols = [col for col in ["Jogo", "Mandante", "Visitante"] if col in df.columns]
            cartoes_df = df[
                cartoes_cols
                + [
                    "Bolao_Aposta",
                    "Aposta",
                    "Aposta_Modelo_Complementar",
                    "Uniao_Bolao_Modelo",
                ]
            ].copy()
            cartoes_df.rename(columns={"Aposta": "Modelo_Base"}, inplace=True)
            cartoes_df.to_csv(cartoes_output, sep=";", index=False)
            logging.info("Cartões salvos em %s.", cartoes_output)
        else:
            logging.info("Arquivo de bolões não encontrado. Pulando aposta complementar.")

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
