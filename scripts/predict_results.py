import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def _normalize_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure probability columns exist and are normalized.

    If only odds are provided, they are converted to probabilities that sum to 1.
    """

    required_prob_cols = ['P(1)', 'P(X)', 'P(2)']
    odds_cols = ['Odds 1', 'Odds X', 'Odds 2']

    if not all(col in df.columns for col in required_prob_cols):
        if all(col in df.columns for col in odds_cols):
            logging.info("Colunas de probabilidade ausentes. Calculando a partir das odds...")
            df['P(1)'] = 1 / df['Odds 1']
            df['P(X)'] = 1 / df['Odds X']
            df['P(2)'] = 1 / df['Odds 2']
        else:
            raise ValueError(
                f"As colunas de odds {odds_cols} são necessárias para calcular as probabilidades no arquivo informado."
            )

    prob_sum = df['P(1)'] + df['P(X)'] + df['P(2)']
    df['P(1)'] /= prob_sum
    df['P(X)'] /= prob_sum
    df['P(2)'] /= prob_sum
    return df


def _compute_match_metrics(prob_array: np.ndarray) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return p_max, gap and zebra values for each match."""

    p_max = pd.Series(prob_array.max(axis=1))
    sorted_probs = np.sort(prob_array, axis=1)[:, ::-1]
    gap = pd.Series(sorted_probs[:, 0] - sorted_probs[:, 1])
    zebra = 1 - p_max
    return p_max, gap, zebra


def _choose_duplos(df: pd.DataFrame, prob_array: np.ndarray, classes: np.ndarray) -> list[int]:
    """Select the five best games for duplos using the custom score."""

    p_max, gap, zebra = _compute_match_metrics(prob_array)
    df['p_max'] = p_max
    df['Gap'] = gap
    df['Zebra'] = zebra
    gap_component = (1 - (gap / 0.25)).clip(0, 1)
    df['Score Duplo'] = 0.75 * zebra + 0.25 * gap_component

    ordered = df.sort_values(
        by=['Score Duplo', 'Entropia', 'Zebra'],
        ascending=False
    )
    top_duplos = ordered.head(5).index.tolist()
    logging.info(f"Índices selecionados para duplos pelo score personalizado: {top_duplos}")
    return top_duplos


def _build_duplo(prob_row: np.ndarray, classes: np.ndarray) -> str:
    """Return the two most probable outcomes for a match."""

    mais_provaveis = prob_row.argsort()[-2:][::-1]
    opcoes_duplas = classes[mais_provaveis]
    return ", ".join(opcoes_duplas)


def _choose_seco(prob_row: np.ndarray, p_max: float, gap: float, classes: np.ndarray,
                 p_x: float) -> str:
    """Apply heuristic rules to select a single outcome (seco)."""

    sorted_indices = prob_row.argsort()[::-1]
    top1, top2 = sorted_indices[:2]

    # Empate inteligente
    if classes[top1] == 'X' and p_max <= 0.45:
        return 'X'
    if classes[top2] == 'X' and gap <= 0.06:
        return 'X'

    candidates: list[tuple[float, str]] = []

    def add_candidate(idx: int):
        prob = prob_row[idx]
        penalty = 0.25 * p_max + 0.25 * gap
        score = prob - penalty
        candidates.append((score, classes[idx]))

    if p_max >= 0.62:
        add_candidate(top1)
    elif p_max <= 0.50 and gap <= 0.08:
        add_candidate(top2)
        add_candidate(top1)
    else:
        add_candidate(top1)
        add_candidate(top2)

    # Reforçar X como opção diferenciada quando for 2ª maior probabilidade.
    if classes[top2] == 'X' and p_x >= 0.28:
        add_candidate(top2)

    best_score, best_choice = max(candidates, key=lambda item: item[0])
    logging.debug(f"Escolha de seco baseada no score {best_score:.4f} -> {best_choice}")
    return best_choice


def _limit_favoritos(df: pd.DataFrame, apostas: list[str], duplo_idxs: list[int], classes: np.ndarray,
                     prob_array: np.ndarray) -> list[str]:
    """Reduz a quantidade de favoritos travados quando ultrapassar o limite."""

    favoritos_idxs = []
    for idx, aposta in enumerate(apostas):
        if idx in duplo_idxs:
            continue
        prob_row = prob_array[idx]
        favorito_idx = prob_row.argmax()
        favorito = classes[favorito_idx]
        if df.loc[idx, 'p_max'] >= 0.62 and aposta == favorito:
            favoritos_idxs.append(idx)

    excesso = max(0, len(favoritos_idxs) - 6)
    if excesso > 0:
        logging.info(f"Reduzindo {excesso} favoritos travados para diversificar o volante.")
        candidatos_troca = (
            df.loc[favoritos_idxs, ['Gap', 'p_max']]
            .assign(idx=favoritos_idxs)
            .sort_values(by=['Gap', 'p_max'])
            .head(excesso)['idx']
        )

        for idx in candidatos_troca:
            sorted_indices = prob_array[idx].argsort()[::-1]
            alternativa = classes[sorted_indices[1]]
            apostas[idx] = alternativa
    return apostas


def _garantir_empates(df: pd.DataFrame, apostas: list[str], duplo_idxs: list[int],
                      prob_array: np.ndarray, classes: np.ndarray, minimo: int = 2) -> list[str]:
    """Garante pelo menos ``minimo`` jogos com X entre as escolhas."""

    def tem_x(aposta: str) -> bool:
        return 'X' in aposta.split(', ')

    empates_atual = sum(tem_x(a) for a in apostas)
    if empates_atual >= minimo:
        return apostas

    candidatos = df.assign(idx=df.index)
    candidatos['top2'] = [
        'X' in classes[prob_array[int(i)].argsort()[-2:]]
        for i in candidatos['idx']
    ]
    candidatos = (
        candidatos
        .query("`P(X)` >= 0.28 and top2 == True")
        .sort_values('P(X)', ascending=False)
    )

    for _, row in candidatos.iterrows():
        idx = int(row['idx'])
        if tem_x(apostas[idx]):
            continue

        if idx in duplo_idxs:
            apostas[idx] = _build_duplo(prob_array[idx], classes)
        else:
            sorted_indices = prob_array[idx].argsort()[::-1]
            if classes[sorted_indices[1]] != 'X':
                continue
            apostas[idx] = 'X'

        empates_atual += 1
        if empates_atual >= minimo:
            break

    if empates_atual < minimo:
        fallback = (
            df.assign(idx=df.index)
            .sort_values('P(X)', ascending=False)
        )
        for _, row in fallback.iterrows():
            idx = int(row['idx'])
            if idx in duplo_idxs or tem_x(apostas[idx]):
                continue

            sorted_indices = prob_array[idx].argsort()[::-1]
            if classes[sorted_indices[1]] != 'X':
                continue

            apostas[idx] = 'X'
            empates_atual += 1

            if empates_atual >= minimo:
                break

    return apostas


def _aplicar_diferencial_controlado(df: pd.DataFrame, apostas: list[str], duplo_idxs: list[int],
                                    prob_array: np.ndarray, classes: np.ndarray,
                                    max_viradas: int = 2) -> list[str]:
    """Aplica 1-2 viradas em jogos equilibrados para evitar pulverização."""

    def pode_virar(idx: int) -> bool:
        return (
            0.52 <= df.loc[idx, 'p_max'] <= 0.62
            and df.loc[idx, 'Gap'] <= 0.10
            and idx not in duplo_idxs
        )

    candidatos = [idx for idx in range(len(df)) if pode_virar(idx)]
    candidatos.sort(key=lambda i: (-df.loc[i, 'Zebra'], -df.loc[i, 'Entropia']))

    viradas_feitas = 0
    for idx in candidatos:
        if viradas_feitas >= max_viradas:
            break

        prob_row = prob_array[idx]
        sorted_indices = prob_row.argsort()[::-1]
        top1, top2 = sorted_indices[:2]
        if apostas[idx] != classes[top1]:
            continue

        if classes[top2] == 'X' or classes[top2] != classes[top1]:
            apostas[idx] = classes[top2]
            viradas_feitas += 1

    return apostas


def predict(input_file, output_file):
    """Generate enriched baseline predictions for future games and write them to CSV."""
    try:
        logging.info("Carregando dados dos jogos futuros...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        df = _normalize_probabilities(df)
        logging.info("Probabilidades carregadas e normalizadas.")

        prob_array = df[['P(1)', 'P(X)', 'P(2)']].to_numpy()
        classes = np.array(['1', 'X', '2'])

        df['Probabilidade (1)'] = np.round(prob_array[:, 0], 5)
        df['Probabilidade (X)'] = np.round(prob_array[:, 1], 5)
        df['Probabilidade (2)'] = np.round(prob_array[:, 2], 5)
        df['Seco'] = classes[prob_array.argmax(axis=1)]

        epsilon = 1e-10
        adjusted_probabilities = prob_array + epsilon
        df['Entropia'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)

        p_max, gap, zebra = _compute_match_metrics(prob_array)
        df['p_max'] = p_max
        df['Gap'] = gap
        df['Zebra'] = zebra

        duplo_idxs = _choose_duplos(df, prob_array, classes)

        logging.info("Gerando a coluna de aposta com 9 secos e 5 duplos baseados nas novas regras...")
        apostas = [''] * len(df)

        for idx in range(len(df)):
            if idx in duplo_idxs:
                apostas[idx] = _build_duplo(prob_array[idx], classes)
            else:
                apostas[idx] = _choose_seco(
                    prob_row=prob_array[idx],
                    p_max=df.loc[idx, 'p_max'],
                    gap=df.loc[idx, 'Gap'],
                    classes=classes,
                    p_x=df.loc[idx, 'P(X)']
                )

        apostas = _limit_favoritos(df, apostas, duplo_idxs, classes, prob_array)
        apostas = _aplicar_diferencial_controlado(df, apostas, duplo_idxs, prob_array, classes)
        apostas = _garantir_empates(df, apostas, duplo_idxs, prob_array, classes)

        df['Aposta'] = apostas

        logging.info(f"Salvando predições no arquivo {output_file}...")
        df.to_csv(output_file, sep=';', index=False)
        logging.info(f"Previsões salvas com sucesso em {output_file}!")

    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo não encontrado - {e}")
    except ValueError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
