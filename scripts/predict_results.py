import logging
import numpy as np
import pandas as pd
from joblib import load

from scripts.features import FEATURE_COLUMNS, INDEX_TO_RESULT, add_engineered_features

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def _selecionar_melhor_combo(valor_cartao_path: str, budget: float) -> dict:
    valores = pd.read_csv(valor_cartao_path, delimiter=';', decimal='.')
    valores.columns = valores.columns.str.replace('\ufeff', '', regex=False).str.strip()
    valores = valores.rename(columns={'Nº de Apostas': 'Num_de_Apostas', 'Valor': 'Valor_da_Aposta'})

    opcoes_validas = valores[valores['Valor_da_Aposta'] <= budget]
    if opcoes_validas.empty:
        raise ValueError("Nenhuma configuração de cartão cabe no orçamento especificado.")

    # Prioriza mais combinações cobrindo mais jogos dentro do orçamento
    melhor = opcoes_validas.sort_values(by=['Num_de_Apostas', 'Valor_da_Aposta'], ascending=[False, True]).iloc[0]
    return {
        'secos': int(melhor['Secos']),
        'duplos': int(melhor['Duplos']),
        'triplos': int(melhor['Triplos']),
        'custo': float(melhor['Valor_da_Aposta']),
    }


def predict(input_file, model_file, scaler_file, output_file, budget: float = 50.0,
            valor_cartao_path: str = 'data/raw/valor_cartao.csv'):
    """Gera predições enriquecidas e sugestões de aposta dentro de um orçamento."""
    try:
        logging.info("Carregando dados dos jogos futuros...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        logging.info("Adicionando engenharia de features para o próximo concurso...")
        df = add_engineered_features(df)

        logging.info("Carregando modelo e scaler...")
        model = load(model_file)
        scaler = load(scaler_file)

        logging.info("Preparando dados para predição...")
        X_future = df[FEATURE_COLUMNS]
        X_future_scaled = scaler.transform(X_future)

        logging.info("Gerando predições...")
        probabilities = model.predict_proba(X_future_scaled)
        predictions = model.predict(X_future_scaled)

        logging.info("Adicionando predições ao DataFrame...")
        df['Probabilidade (1)'] = np.round(probabilities[:, 0], 5)
        df['Probabilidade (X)'] = np.round(probabilities[:, 1], 5)
        df['Probabilidade (2)'] = np.round(probabilities[:, 2], 5)
        df['Secos'] = predictions

        epsilon = 1e-10
        adjusted_probabilities = probabilities + epsilon
        df['Entropia'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)
        df['p_seco'] = probabilities.max(axis=1)
        df['risk'] = 1 - df['p_seco']

        logging.info("Carregando opções de cartão para respeitar o orçamento...")
        combo = _selecionar_melhor_combo(valor_cartao_path, budget)
        logging.info(f"Melhor combinação encontrada: {combo}")

        df['Aposta'] = df['Secos']
        df['score_duplo_tripo'] = df['Entropia'] * df['risk']

        candidatos = df.sort_values(by='score_duplo_tripo', ascending=False).index.tolist()
        triplos_idxs = candidatos[:combo['triplos']]
        duplos_idxs = candidatos[combo['triplos']: combo['triplos'] + combo['duplos']]

        logging.info("Aplicando triplos e duplos nos jogos mais incertos...")
        for idx in triplos_idxs:
            df.loc[idx, 'Aposta'] = '1, X, 2'

        for idx in duplos_idxs:
            mais_provaveis = probabilities[idx].argsort()[-2:][::-1]
            df.loc[idx, 'Aposta'] = f"{INDEX_TO_RESULT[mais_provaveis[0]]}, {INDEX_TO_RESULT[mais_provaveis[1]]}"

        logging.info(f"Custo estimado do cartão: R$ {combo['custo']:.2f}")

        logging.info(f"Salvando predições no arquivo {output_file}...")
        df.to_csv(output_file, sep=';', index=False)
        logging.info(f"Previsões salvas com sucesso em {output_file}!")

    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo não encontrado - {e}")
    except ValueError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
