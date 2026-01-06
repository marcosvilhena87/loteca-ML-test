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


def _reordenar_probabilidades(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    ordem_desejada = ['1', 'X', '2']
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    if not all(label in class_to_idx for label in ordem_desejada):
        raise ValueError("O modelo não contém todas as classes esperadas ('1', 'X', '2').")

    ordered = np.zeros_like(probabilities)
    for pos, label in enumerate(ordem_desejada):
        ordered[:, pos] = probabilities[:, class_to_idx[label]]
    return ordered


def _ajustar_empates(predictions: np.ndarray, probabilities: np.ndarray, threshold: float) -> np.ndarray:
    if threshold is None:
        return predictions

    ajustados = []
    for pred, prob_row in zip(predictions, probabilities):
        if pred == 'X' and prob_row[1] < threshold:
            fallback_idx = 0 if prob_row[0] >= prob_row[2] else 2
            ajustados.append(INDEX_TO_RESULT[fallback_idx])
        else:
            ajustados.append(pred)
    return np.array(ajustados)


def predict(
    input_file,
    model_file,
    scaler_file,
    output_file,
    budget: float = 50.0,
    valor_cartao_path: str = 'data/raw/valor_cartao.csv',
    draw_threshold: float = 0.35,
    score_strategy: str = 'risk',
):
    """Gera predições enriquecidas e sugestões de aposta dentro de um orçamento.

    draw_threshold: probabilidade mínima para aceitar empate como seco.
    score_strategy: como ranquear jogos para duplos/triplos ("risk", "risk_squared", "entropy_risk").
    """
    try:
        logging.info("Carregando dados dos jogos futuros...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        logging.info("Adicionando engenharia de features para o próximo concurso...")
        df = add_engineered_features(df)

        logging.info("Carregando modelo e pré-processador...")
        model = load(model_file)
        preprocessor = load(scaler_file)

        logging.info("Preparando dados para predição...")
        X_future = df[FEATURE_COLUMNS]
        X_future_processed = preprocessor.transform(X_future)

        logging.info("Gerando predições...")
        probabilities = model.predict_proba(X_future_processed)
        ordered_probabilities = _reordenar_probabilidades(probabilities, model.classes_)
        predictions = ordered_probabilities.argmax(axis=1)

        logging.info("Adicionando predições ao DataFrame...")
        df['Probabilidade (1)'] = np.round(ordered_probabilities[:, 0], 5)
        df['Probabilidade (X)'] = np.round(ordered_probabilities[:, 1], 5)
        df['Probabilidade (2)'] = np.round(ordered_probabilities[:, 2], 5)
        secos = np.array([INDEX_TO_RESULT[idx] for idx in predictions])
        df['Secos'] = _ajustar_empates(secos, ordered_probabilities, draw_threshold)
        df['Seco'] = df['Secos']  # coluna legada para compatibilidade com análises anteriores

        epsilon = 1e-10
        adjusted_probabilities = ordered_probabilities + epsilon
        df['Entropia'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)
        df['p_seco'] = ordered_probabilities.max(axis=1)
        df['risk'] = 1 - df['p_seco']

        logging.info("Carregando opções de cartão para respeitar o orçamento...")
        combo = _selecionar_melhor_combo(valor_cartao_path, budget)
        logging.info(f"Melhor combinação encontrada: {combo}")

        df['Aposta'] = df['Secos']
        if score_strategy == 'risk':
            base_score = df['risk']
        elif score_strategy == 'risk_squared':
            base_score = df['risk'] ** 2
        elif score_strategy == 'entropy_risk':
            base_score = df['Entropia'] * df['risk']
        else:
            raise ValueError("score_strategy deve ser 'risk', 'risk_squared' ou 'entropy_risk'.")

        df['score_duplo_tripo'] = base_score
        df['duplo_score'] = base_score * df['p2nd'] * (1 - df['gap'])
        df['triplo_score'] = df['Entropia'] * base_score

        triplos_idxs = df.sort_values(by='triplo_score', ascending=False).index.tolist()[:combo['triplos']]
        duplo_pool = df.drop(index=triplos_idxs)
        duplos_idxs = duplo_pool.sort_values(by='duplo_score', ascending=False).index.tolist()[:combo['duplos']]

        logging.info("Aplicando triplos e duplos nos jogos mais incertos...")
        for idx in triplos_idxs:
            df.loc[idx, 'Aposta'] = '1, X, 2'

        for idx in duplos_idxs:
            mais_provaveis = ordered_probabilities[idx].argsort()[-2:][::-1]
            df.loc[idx, 'Aposta'] = f"{INDEX_TO_RESULT[mais_provaveis[0]]}, {INDEX_TO_RESULT[mais_provaveis[1]]}"

        logging.info(f"Custo estimado do cartão: R$ {combo['custo']:.2f}")

        df['Config_Secos'] = combo['secos']
        df['Config_Duplos'] = combo['duplos']
        df['Config_Triplos'] = combo['triplos']
        df['Config_Custo'] = combo['custo']
        df['Score_Strategy'] = score_strategy

        logging.info(f"Salvando predições no arquivo {output_file}...")
        df.to_csv(output_file, sep=';', index=False)
        logging.info(f"Previsões salvas com sucesso em {output_file}!")

    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo não encontrado - {e}")
    except ValueError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
