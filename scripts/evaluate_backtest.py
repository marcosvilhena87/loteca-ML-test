import logging
from typing import Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def _buscar_valor_cartao(valor_path: str, secos: int, duplos: int, triplos: int) -> Tuple[int, float]:
    valores = pd.read_csv(valor_path, delimiter=';', decimal='.')
    valores.columns = valores.columns.str.replace('\ufeff', '', regex=False).str.strip()
    valores = valores.rename(columns={'Nº de Apostas': 'Num_de_Apostas', 'Valor': 'Valor_da_Aposta'})

    linha = valores[
        (valores['Secos'] == secos)
        & (valores['Duplos'] == duplos)
        & (valores['Triplos'] == triplos)
    ]
    if linha.empty:
        raise ValueError("Combinação de secos/duplos/triplos não encontrada no valor_cartao.csv")

    registro = linha.iloc[0]
    return int(registro['Num_de_Apostas']), float(registro['Valor_da_Aposta'])


def _recuperar_rateios(rateio_path: str, concurso: int) -> Tuple[float, float]:
    rateios = pd.read_csv(rateio_path, delimiter=';', decimal='.')
    rateios.columns = rateios.columns.str.replace('\ufeff', '', regex=False).str.strip()

    linha = rateios[rateios['Concurso'] == concurso]
    if linha.empty:
        raise ValueError(f"Concurso {concurso} não encontrado em concurso_rateio.csv")

    registro = linha.iloc[0]
    return float(registro['Rateio 14 Acertos']), float(registro['Rateio 13 Acertos'])


def avaliar_roi(predictions_file: str, rateio_file: str, valor_cartao_file: str,
                secos: int, duplos: int, triplos: int) -> dict:
    """Calcula um ROI simplificado comparando acertos vs custo do cartão."""
    df = pd.read_csv(predictions_file, delimiter=';', decimal='.')
    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()

    if not {'Aposta', 'Resultado', 'Concurso'}.issubset(df.columns):
        raise ValueError("O arquivo de predições precisa conter 'Aposta', 'Resultado' e 'Concurso'.")

    concurso = int(df['Concurso'].iloc[0])
    premio_14, premio_13 = _recuperar_rateios(rateio_file, concurso)
    num_apostas, custo_cartao = _buscar_valor_cartao(valor_cartao_file, secos, duplos, triplos)

    acertos = df.apply(lambda row: str(row['Resultado']) in str(row['Aposta']), axis=1).sum()
    logging.info(f"Total de jogos acertados: {acertos}/14")

    retorno = 0.0
    if acertos == 14:
        retorno = premio_14
    elif acertos == 13:
        retorno = premio_13

    roi = (retorno - custo_cartao) / custo_cartao
    resumo = {
        'concurso': concurso,
        'acertos': acertos,
        'retorno': retorno,
        'custo_cartao': custo_cartao,
        'roi': roi,
        'num_apostas': num_apostas,
    }
    logging.info(f"ROI calculado: {roi:.3f} (retorno {retorno} para custo {custo_cartao})")
    return resumo
