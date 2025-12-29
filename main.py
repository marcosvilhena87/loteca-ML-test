import logging
import os

from scripts import predict, process, train, train_pulverization_models

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Configurações iniciais
PATHS = {
    "raw_data": "data/raw/",
    "processed_data": "data/processed/",
    "models": "models/",
    "output": "output/",
}

# Função para criar as pastas necessárias
def ensure_directories_exist(paths):
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

# Função principal do pipeline
def run_pipeline():
    ensure_directories_exist(PATHS)

    # Passo 1: Processamento dos dados
    logging.info("[1/5] Processando dados...")
    raw_data_file = os.path.join(PATHS["raw_data"], "concursos_anteriores.csv")
    processed_data_file = os.path.join(PATHS["processed_data"], "loteca_treinamento.csv")
    process(raw_data_file, processed_data_file)
    logging.info(f"Dados processados salvos em {processed_data_file}")

    # Passo 2: Treinamento do modelo
    logging.info("[2/5] Treinando modelo...")
    model_file = os.path.join(PATHS["models"], "final_model.pkl")
    train(processed_data_file, model_file)
    logging.info(f"Modelo treinado e salvo em {model_file}")

    # Passo 3: Treino de modelos de pulverização por concurso
    logging.info("[3/5] Treinando modelos de pulverização...")
    rateio_file = os.path.join(PATHS["raw_data"], "concurso_rateio.csv")
    train_pulverization_models(processed_data_file, rateio_file, PATHS["models"])

    # Passo 4: Predição de resultados futuros
    logging.info("[4/5] Gerando predições...")
    future_games_file = os.path.join(PATHS["raw_data"], "proximo_concurso.csv")
    predictions_file = os.path.join(PATHS["output"], "predictions.csv")
    predict(future_games_file, model_file, predictions_file)
    logging.info(f"Predições salvas em {predictions_file}")

    # Passo 5: Conclusão
    logging.info("[5/5] Pipeline concluído com sucesso!")

# Execução do pipeline
if __name__ == "__main__":
    run_pipeline()
