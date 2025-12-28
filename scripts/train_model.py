import json
import logging
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def train(input_file, report_file):
    """Evaluate the market baseline and persist metrics.

    Parameters
    ----------
    input_file : str
        CSV file containing training features and labels.
    report_file : str
        Path to save the evaluation metrics as JSON.

    Returns
    -------
        None
        The baseline metrics are written to disk.
    """
    try:
        logging.info("Carregando os dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        required_columns = ['P(1)', 'P(X)', 'P(2)', 'Resultado']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"As seguintes colunas necessárias estão ausentes: {missing_columns}")

        logging.info("Calculando métricas do baseline (argmax das probabilidades implícitas)...")
        prob_columns = ['P(1)', 'P(X)', 'P(2)']
        prob_df = df[prob_columns]
        baseline_mapping = {'P(1)': '1', 'P(X)': 'X', 'P(2)': '2'}

        baseline_pred = prob_df.idxmax(axis=1).map(baseline_mapping)
        accuracy = (baseline_pred == df['Resultado']).mean()
        logging.info(f"Acurácia do baseline: {accuracy:.4f}")

        label_order = ['1', '2', 'X']
        prob_df_for_loss = df[['P(1)', 'P(2)', 'P(X)']]
        baseline_log_loss = log_loss(df['Resultado'], prob_df_for_loss, labels=label_order)
        logging.info(f"Log loss (utilizando as probabilidades implícitas): {baseline_log_loss:.4f}")

        class_brier_scores = {}
        for class_label in ['1', 'X', '2']:
            class_true = (df['Resultado'] == class_label).astype(int)
            class_brier = brier_score_loss(class_true, prob_df[f'P({class_label})'])
            class_brier_scores[class_label] = class_brier
            logging.info(f"Brier score para a classe {class_label}: {class_brier:.4f}")

        metrics = {
            "accuracy": accuracy,
            "log_loss": baseline_log_loss,
            "brier_score": class_brier_scores,
        }

        logging.info(f"Salvando métricas do baseline em {report_file}...")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        logging.info("Avaliação concluída com sucesso!")
    
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except ValueError as e:
        logging.error(f"Erro nos dados: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
