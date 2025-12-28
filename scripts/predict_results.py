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
            else:
                raise ValueError(
                    f"As colunas de odds {odds_cols} são necessárias para calcular as probabilidades no arquivo {input_file}."
                )

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

        # Seco = classe com maior probabilidade do bookmaker
        class_labels = ['1', 'X', '2']
        seco_indices = normalized_probs.argmax(axis=1)
        df['Seco'] = [class_labels[i] for i in seco_indices]

        # Adicionando um valor pequeno para evitar problemas com log(0)
        epsilon = 1e-10
        adjusted_probabilities = np.clip(normalized_probs, epsilon, 1.0)

        # Calculando a entropia com as probabilidades ajustadas
        logging.info("Calculando entropia (com base no bookmaker) para determinar os jogos mais incertos...")
        df['Entropia'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)

        # Labels mais prováveis para cada jogo
        top2_indices = np.argsort(adjusted_probabilities, axis=1)[:, -2:][:, ::-1]
        top2_labels = []
        for idxs in top2_indices:
            seen = set()
            ordered_labels = []
            for idx in idxs:
                label = class_labels[idx]
                if label not in seen:
                    ordered_labels.append(label)
                    seen.add(label)

            if len(ordered_labels) < 2:
                fallback_label = next(label for label in class_labels if label not in seen)
                ordered_labels.append(fallback_label)

            top2_labels.append(ordered_labels)

        df['Top2'] = [", ".join(labels) for labels in top2_labels]

        # Calculando o "gap" entre as duas maiores probabilidades
        prob_sorted = np.sort(adjusted_probabilities, axis=1)[:, ::-1]
        df['Gap'] = prob_sorted[:, 0] - prob_sorted[:, 1]

        # Identificar os jogos mais incertos (menor gap, desempate por maior entropia)
        n_duplos = 5
        jogos_ordenados = df.sort_values(by=['Gap', 'Entropia'], ascending=[True, False])
        rank_duplo = pd.Series(range(1, len(jogos_ordenados) + 1), index=jogos_ordenados.index)
        df['RankDuplo'] = rank_duplo.reindex(df.index)
        jogos_duplos_idxs = jogos_ordenados.head(min(n_duplos, len(df))).index
        if len(df) < n_duplos:
            logging.warning(
                "Menos jogos do que o esperado. Aplicando duplos em todos: "
                f"{len(df)} jogos disponíveis para {n_duplos} duplos."
            )
        else:
            logging.info(f"Aplicando exatamente {n_duplos} duplos com base no gap.")

        expected_games = 14
        if len(df) != expected_games:
            logging.warning(
                f"Quantidade de jogos diferente do esperado ({expected_games}). Encontrados {len(df)} registros."
            )
        logging.info(
            "Índices dos jogos selecionados para duplos (gap + entropia): "
            f"{jogos_duplos_idxs.tolist()}"
        )

        if len(jogos_duplos_idxs) != n_duplos:
            logging.warning(
                f"Total de duplos gerados: {len(jogos_duplos_idxs)} (configurado: {n_duplos})."
            )
        logging.info(
            "Distribuição planejada: %s secos e %s duplos.",
            len(df) - len(jogos_duplos_idxs),
            len(jogos_duplos_idxs)
        )

        # Gerar a coluna "Aposta"
        logging.info("Gerando a coluna de aposta com 9 secos e 5 duplos...")
        df['Aposta'] = df['Seco']  # Copia as apostas secas inicialmente

        # Escolhendo os "duplos" para os 5 jogos mais incertos
        for idx in jogos_duplos_idxs:
            df.loc[idx, 'Aposta'] = df.loc[idx, 'Top2']

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
