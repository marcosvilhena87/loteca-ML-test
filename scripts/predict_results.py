import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def predict(input_file, model_file, output_file):
    """Generate predictions for future games and write them to CSV.

    Parameters
    ----------
    input_file : str
        CSV file with upcoming games and odds or probabilities.
    model_file : str
        Unused legacy parameter kept for API compatibility.
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
            odds_cols = ['Odds 1', 'Odds X', 'Odds 2']
            if all(col in df.columns for col in odds_cols):
                logging.info("Colunas de probabilidade ausentes. Calculando a partir das odds...")
                df['P(1)'] = 1 / df['Odds 1']
                df['P(X)'] = 1 / df['Odds X']
                df['P(2)'] = 1 / df['Odds 2']

                prob_sum = df['P(1)'] + df['P(X)'] + df['P(2)']
                df['P(1)'] /= prob_sum
                df['P(X)'] /= prob_sum
                df['P(2)'] /= prob_sum
                logging.info("Probabilidades calculadas e normalizadas a partir das odds do bookmaker.")
            else:
                raise ValueError(
                    f"As colunas de odds {odds_cols} são necessárias para calcular as probabilidades no arquivo {input_file}."
                )
        else:
            logging.info("Probabilidades já presentes no arquivo. Usando valores fornecidos.")

        # Reconfirmando que as colunas de probabilidade agora estão presentes
        required_columns = ['P(1)', 'P(X)', 'P(2)']

        # Utilizando diretamente as probabilidades do bookmaker para decisões
        logging.info("Usando as probabilidades do bookmaker para gerar apostas...")
        bookmaker_probs = df[required_columns].to_numpy(dtype=float)

        # Reforçando a normalização para somar 1
        prob_sum = bookmaker_probs.sum(axis=1, keepdims=True)
        if np.any(prob_sum == 0):
            raise ValueError("Probabilidades do bookmaker não podem somar 0.")
        normalized_probs = bookmaker_probs / prob_sum
        if not np.all(np.isfinite(normalized_probs)):
            raise ValueError("Probabilidades do bookmaker resultaram em valores não numéricos.")

        # Registrar as probabilidades usadas no output
        df['Probabilidade (1)'] = np.round(normalized_probs[:, 0], 5)
        df['Probabilidade (X)'] = np.round(normalized_probs[:, 1], 5)
        df['Probabilidade (2)'] = np.round(normalized_probs[:, 2], 5)

        class_labels = ['1', 'X', '2']

        # Identificação do favorito e do segundo favorito em cada jogo
        top2_indices = np.argsort(normalized_probs, axis=1)[:, ::-1][:, :2]
        top2_probs = np.take_along_axis(normalized_probs, top2_indices, axis=1)
        top2_labels = [[class_labels[i] for i in row] for row in top2_indices]

        p_max = top2_probs[:, 0]
        second_best = top2_probs[:, 1]
        df['ProbFavorito'] = np.round(p_max, 5)
        df['ProbSegundo'] = np.round(second_best, 5)
        df['Top2'] = [", ".join(labels) for labels in top2_labels]

        # Regra 1: secos mantêm o favorito quando ele é forte
        seco_indices = top2_indices[:, 0]
        df['Seco'] = [class_labels[i] for i in seco_indices]

        # Adicionando um valor pequeno para evitar problemas com log(0)
        epsilon = 1e-10
        adjusted_probabilities = np.clip(normalized_probs, epsilon, 1.0)

        # Mantemos a entropia para inspeção, embora a seleção siga as novas regras
        logging.info("Calculando entropia (com base no bookmaker) para registrar incerteza...")
        df['Entropia'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)

        # Calculando o "gap" entre as duas maiores probabilidades
        df['Gap'] = p_max - second_best

        # Regra 2: selecionar duplos apenas em jogos equilibrados (gap <= 0.12)
        n_duplos = 5
        mask_equilibrado = (df['Gap'] <= 0.12) & (p_max < 0.55)
        candidatos_duplo = df[mask_equilibrado].sort_values(by='Gap')
        if len(candidatos_duplo) < n_duplos:
            logging.info(
                "Menos de %s jogos com gap <= 0.12. Completando com menores gaps restantes.",
                n_duplos,
            )
            faltantes = n_duplos - len(candidatos_duplo)
            restantes = df[~df.index.isin(candidatos_duplo.index)].sort_values(by='Gap')
            candidatos_duplo = pd.concat([candidatos_duplo, restantes.head(faltantes)])
        jogos_duplos_idxs = candidatos_duplo.head(min(n_duplos, len(df))).index

        expected_games = 14
        if len(df) != expected_games:
            logging.warning(
                f"Quantidade de jogos diferente do esperado ({expected_games}). Encontrados {len(df)} registros."
            )

        # Regra 3: antipulverização controlada (1 contrário opcional)
        df['Aposta'] = df['Seco']
        if len(jogos_duplos_idxs) > 0:
            logging.info("Aplicando duplos a %s jogos conforme regra de gap.", len(jogos_duplos_idxs))
        for idx in jogos_duplos_idxs:
            df.loc[idx, 'Aposta'] = df.loc[idx, 'Top2']

        candidatos_contrario = df.loc[
            (~df.index.isin(jogos_duplos_idxs))
            & (p_max >= 0.50)
            & (p_max <= 0.58)
            & (second_best >= 0.30)
        ]

        if not candidatos_contrario.empty:
            escolhido = candidatos_contrario.sort_values(by='ProbSegundo', ascending=False)
            idx_contrario = escolhido.index[0]
            df.loc[idx_contrario, 'Aposta'] = top2_labels[idx_contrario][1]
            logging.info(
                "Aplicando antipulverização no jogo %s (favorito %.2f, segundo %.2f).",
                idx_contrario,
                p_max[idx_contrario],
                second_best[idx_contrario],
            )

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
