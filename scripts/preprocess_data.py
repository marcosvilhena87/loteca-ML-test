import logging
import pandas as pd

from scripts.features import (
    FEATURE_COLUMNS,
    ODDS_COLUMNS,
    PROB_COLUMNS,
    add_engineered_features,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def _compute_result(row):
    if row.get('[1]') == 1:
        return '1'
    if row.get('[x]') == 1:
        return 'X'
    if row.get('[2]') == 1:
        return '2'
    return None


def _validate_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas com odds inválidas ou ausentes."""

    missing_odds = [col for col in ODDS_COLUMNS if col not in df.columns]
    if missing_odds:
        raise KeyError(f"Colunas de odds ausentes: {missing_odds}")

    invalid_mask = df[ODDS_COLUMNS].isnull().any(axis=1) | (df[ODDS_COLUMNS] <= 1).any(axis=1)
    if invalid_mask.any():
        logging.warning("Removendo %d linhas com odds inválidas ou nulas.", invalid_mask.sum())
        df = df.loc[~invalid_mask].copy()
    return df


def _validate_probabilities(df: pd.DataFrame, tol: float = 1e-3) -> pd.DataFrame:
    """Garante que as probabilidades somam aproximadamente 1."""

    prob_sum = df[PROB_COLUMNS].sum(axis=1)
    invalid_mask = prob_sum.isnull() | (prob_sum < 1 - tol) | (prob_sum > 1 + tol)
    if invalid_mask.any():
        logging.warning(
            "Removendo %d linhas com probabilidades fora do intervalo esperado.",
            invalid_mask.sum(),
        )
        df = df.loc[~invalid_mask].copy()
    return df


def _validate_one_hot_results(df: pd.DataFrame) -> pd.DataFrame:
    """Confere se [1]/[x]/[2] formam one-hot válido e remove linhas ruins."""

    required = ['[1]', '[x]', '[2]']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError("As colunas '[1]', '[x]' e '[2]' são necessárias para calcular o resultado.")

    values_ok = df[required].apply(lambda col: col.isin([0, 1])).all(axis=1)
    row_sum = df[required].sum(axis=1)
    one_hot_ok = row_sum == 1
    valid_mask = values_ok & one_hot_ok
    if (~valid_mask).any():
        logging.warning("Removendo %d linhas com labels inválidos (one-hot incorreto).", (~valid_mask).sum())
        df = df.loc[valid_mask].copy()
    return df


def process(input_file, output_file):
    """Limpa e enriquece o histórico com novas features."""
    try:
        logging.info("Carregando dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        if not all(col in df.columns for col in PROB_COLUMNS):
            logging.info("Validando colunas de odds...")
            df = _validate_odds(df)

        logging.info("Gerando features derivadas...")
        df = add_engineered_features(df)

        logging.info("Validando probabilidades normalizadas...")
        df = _validate_probabilities(df)

        logging.info("Validando labels one-hot...")
        df = _validate_one_hot_results(df)

        logging.info("Determinando o resultado real de cada jogo...")
        df['Resultado'] = df.apply(_compute_result, axis=1)

        logging.info("Removendo linhas inválidas...")
        df = df.dropna(subset=['Resultado'] + FEATURE_COLUMNS)

        logging.info(f"Salvando o arquivo processado em {output_file}...")
        df.to_csv(output_file, index=False, sep=';', decimal='.')
        logging.info("Processamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except KeyError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
