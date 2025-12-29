import logging
import os

import numpy as np
import pandas as pd
from joblib import load

from .pulverization import calculate_concurso_features

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

        # Persistir as probabilidades normalizadas para reuso
        df['P(1)'] = normalized_probs[:, 0]
        df['P(X)'] = normalized_probs[:, 1]
        df['P(2)'] = normalized_probs[:, 2]

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
        concurso_features = calculate_concurso_features(df[['P(1)', 'P(X)', 'P(2)']])
        aggregated_metrics = {
            "mean_prob_favorito": float(concurso_features["mean_pmax"]),
            "mean_entropy": float(concurso_features["mean_entropy"]),
            "gap_std": float(concurso_features["std_gap"]),
            "mean_gap": float(concurso_features["mean_gap"]),
        }
        logging.info("Métricas agregadas do concurso: %s", aggregated_metrics)

        models_dir = os.path.dirname(model_file) or ""

        def _load_bundle(path):
            if not os.path.exists(path):
                logging.warning("Modelo de pulverização não encontrado em %s", path)
                return None
            return load(path)

        ganhadores_bundle = _load_bundle(os.path.join(models_dir, "pulverization_ganhadores.joblib"))
        rateio_bundle = _load_bundle(os.path.join(models_dir, "pulverization_rateio.joblib"))

        pred_ganhadores14 = None
        pred_rateio14 = None
        if ganhadores_bundle is not None:
            feature_frame = pd.DataFrame([concurso_features])[ganhadores_bundle["feature_names"]]
            pred_log = ganhadores_bundle["model"].predict(feature_frame)[0]
            pred_ganhadores14 = float(np.expm1(pred_log))
            logging.info("Pulverização prevista (ganhadores14): %.2f", pred_ganhadores14)

        if rateio_bundle is not None:
            feature_frame = pd.DataFrame([concurso_features])[rateio_bundle["feature_names"]]
            pred_log_rateio = rateio_bundle["model"].predict(feature_frame)[0]
            pred_rateio14 = float(np.exp(pred_log_rateio) - 1)
            logging.info("Rateio previsto (14 acertos): R$ %.2f", pred_rateio14)

        if pred_ganhadores14 is None:
            logging.warning("Modelo de pulverização não carregado; usando configuração padrão.")

        def _policy_from_pulverization(pred_ganhadores, pred_rateio):
            base_policy = {
                "n_duplos": 5,
                "gap_threshold": 0.12,
                "min_second_best_duplo": 0.25,
                "max_contrarios": 1,
                "contrario_range": (0.50, 0.58),
                "min_second_best": 0.30,
            }

            if pred_ganhadores is None:
                return base_policy

            rateio_baixo = pred_rateio is None or pred_rateio <= 1_000_000
            muitos_ganhadores = pred_ganhadores >= 50
            varios_ganhadores = pred_ganhadores >= 30

            if pred_ganhadores >= 50:
                base_policy.update({
                    "n_duplos": 6,
                    "gap_threshold": 0.14,
                    "min_second_best": 0.28,
                })
            elif pred_ganhadores <= 10:
                base_policy.update({
                    "gap_threshold": 0.10,
                    "max_contrarios": 0,
                    "contrario_range": (0.0, 0.0),
                    "min_second_best": 1.0,
                })

            if (muitos_ganhadores or varios_ganhadores) and rateio_baixo:
                if muitos_ganhadores:
                    base_policy.update({
                        "max_contrarios": 2,
                        "contrario_range": (0.48, 0.62),
                    })
                else:
                    base_policy.update({
                        "max_contrarios": 1,
                        "contrario_range": (0.50, 0.60),
                    })
            else:
                base_policy.update({
                    "max_contrarios": 0,
                    "contrario_range": (0.0, 0.0),
                })

            return base_policy

        policy = _policy_from_pulverization(pred_ganhadores14, pred_rateio14)

        def _apply_metric_overrides(policy_settings, metrics):
            """Tweak the policy with contest-level risk signals."""
            n_duplos_local = policy_settings["n_duplos"]
            max_contrarios_local = policy_settings["max_contrarios"]
            ajustes = []
            freio_pulverizacao = False

            entropy = metrics["mean_entropy"]
            gap_std = metrics["gap_std"]
            mean_pmax = metrics["mean_prob_favorito"]

            if entropy >= 1.05:
                n_duplos_local += 1
                ajustes.append("entropia_alta")
            elif entropy <= 0.95:
                max_contrarios_local = max(0, max_contrarios_local - 1)
                ajustes.append("entropia_baixa")

            if gap_std >= 0.10:
                n_duplos_local += 1
                max_contrarios_local = max(0, max_contrarios_local - 1)
                ajustes.append("desvio_gap_alto")

            favorite_cap_threshold = 0.62
            if mean_pmax > favorite_cap_threshold:
                n_duplos_local = max(0, n_duplos_local - 1)
                max_contrarios_local = 0
                freio_pulverizacao = True
                ajustes.append("teto_favoritos")

            return {
                "n_duplos": int(n_duplos_local),
                "max_contrarios": int(max_contrarios_local),
                "ajustes": ajustes,
                "freio_pulverizacao": freio_pulverizacao,
            }

        override = _apply_metric_overrides(policy, aggregated_metrics)
        policy["n_duplos"] = override["n_duplos"]
        policy["max_contrarios"] = override["max_contrarios"]

        n_duplos = policy["n_duplos"]
        min_second_best_duplo = policy["min_second_best_duplo"]
        absolute_gap_limit = 0.18

        logging.info(
            "Política final após sinais de risco: n_duplos=%s, max_contrarios=%s, ajustes=%s",
            n_duplos,
            policy["max_contrarios"],
            ", ".join(override["ajustes"]) if override["ajustes"] else "nenhum",
        )

        mask_equilibrado = (
            (df['Gap'] <= policy["gap_threshold"]) &
            (df['ProbSegundo'] >= min_second_best_duplo) &
            (p_max < 0.55)
        )
        candidatos_duplo = df[mask_equilibrado].sort_values(by='Gap')
        if len(candidatos_duplo) < n_duplos:
            logging.info(
                "Menos de %s jogos com gap <= %.2f e segundo favorito >= %.2f."
                " Buscando gaps adicionais abaixo de %.2f.",
                n_duplos,
                policy["gap_threshold"],
                min_second_best_duplo,
                absolute_gap_limit,
            )
            faltantes = n_duplos - len(candidatos_duplo)
            restantes = df[~df.index.isin(candidatos_duplo.index)]
            restantes = restantes[
                (restantes['Gap'] <= absolute_gap_limit)
                & (restantes['ProbSegundo'] >= min_second_best_duplo)
                & (restantes['ProbFavorito'] < 0.55)
            ].sort_values(by='Gap')
            adicionais = restantes.head(faltantes)
            candidatos_duplo = pd.concat([candidatos_duplo, adicionais])
        jogos_duplos_idxs = candidatos_duplo.head(min(n_duplos, len(df))).index

        expected_games = 14
        if len(df) != expected_games:
            logging.warning(
                f"Quantidade de jogos diferente do esperado ({expected_games}). Encontrados {len(df)} registros."
            )

        # Regra 3: antipulverização controlada (1 contrário opcional)
        df['Aposta'] = df['Seco']
        df['ContrarioAplicado'] = False
        df['Motivo'] = ""
        if len(jogos_duplos_idxs) > 0:
            logging.info("Aplicando duplos a %s jogos conforme regra de gap.", len(jogos_duplos_idxs))
        for idx in jogos_duplos_idxs:
            df.loc[idx, 'Aposta'] = df.loc[idx, 'Top2']

        candidatos_contrario = df.loc[
            (~df.index.isin(jogos_duplos_idxs))
            & (p_max >= policy["contrario_range"][0])
            & (p_max <= policy["contrario_range"][1])
            & (second_best >= policy["min_second_best"])
        ]

        if policy["max_contrarios"] > 0 and not candidatos_contrario.empty:
            escolhidos = candidatos_contrario.sort_values(
                by=['Gap', 'ProbSegundo'], ascending=[True, False]
            ).head(
                policy["max_contrarios"]
            )
            for idx_contrario in escolhidos.index:
                df.loc[idx_contrario, 'Aposta'] = top2_labels[idx_contrario][1]
                df.loc[idx_contrario, 'ContrarioAplicado'] = True
                df.loc[idx_contrario, 'Motivo'] = (
                    "Antipulverizacao: Gap {:.3f}; favorito {:.2f}; segundo {:.2f}".format(
                        df.loc[idx_contrario, 'Gap'],
                        p_max[idx_contrario],
                        second_best[idx_contrario],
                    )
                )
                logging.info(
                    "Aplicando antipulverização no jogo %s (favorito %.2f, segundo %.2f).",
                    idx_contrario,
                    p_max[idx_contrario],
                    second_best[idx_contrario],
                )

        df['PredGanhadores14'] = pred_ganhadores14
        df['PredRateio14'] = pred_rateio14
        df['MetricMeanProbFavorito'] = aggregated_metrics["mean_prob_favorito"]
        df['MetricMeanEntropy'] = aggregated_metrics["mean_entropy"]
        df['MetricGapStd'] = aggregated_metrics["gap_std"]
        df['MetricMeanGap'] = aggregated_metrics["mean_gap"]
        df['FreioPulverizacao'] = override["freio_pulverizacao"]
        df['AjustesPulverizacao'] = ", ".join(override["ajustes"]) if override["ajustes"] else "nenhum"
        df['DuplosPlanejados'] = n_duplos
        df['MaxContrariosPlanejado'] = policy["max_contrarios"]

        if override["freio_pulverizacao"]:
            logging.info(
                "Freio de pulverização acionado: média favoritos=%.3f, duplos=%s, contrários zerados.",
                aggregated_metrics["mean_prob_favorito"],
                n_duplos,
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
