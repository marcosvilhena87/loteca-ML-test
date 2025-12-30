# Loteca Machine Learning Pipeline

This project predicts results for the Brazilian "Loteca" football lottery. The repository contains a small data pipeline that cleans historical matches, trains a model and generates predictions for upcoming games.

## Requirements

- **Python**: 3.8 or newer

Install dependencies by running:

```bash
pip install -r requirements.txt
```

Run this command before executing the pipeline or running the tests.

## Data preparation

Place your CSV files in `data/raw/`:

- `concursos_anteriores.csv` – historical games with columns such as `Mandante`, `Visitante`, outcome flags (`[1]`, `[x]`, `[2]`) and odds.
- `proximo_concurso.csv` – upcoming games with the same odds columns.

Running the pipeline will create `data/processed/loteca_treinamento.csv` and store trained models in `models/`.

## Running the full pipeline

Execute the following from the repository root:

```bash
python main.py
```

The script processes the raw data, trains a `RandomForestClassifier` and writes predictions to `output/predictions.csv`.

## Running individual steps

Functions for each stage live inside `scripts/`. You can import and call them directly:

```bash
python - <<'PY'
from scripts import process, train, predict

process(
    'data/raw/concursos_anteriores.csv',
    'data/processed/loteca_treinamento.csv')
train(
    'data/processed/loteca_treinamento.csv',
    'models/final_model.pkl',
    None)
predict(
    'data/raw/proximo_concurso.csv',
    'models/final_model.pkl',
    None,
    'output/predictions.csv')
PY
```

This executes the same pipeline steps individually.

## Running tests

The repository includes a few unit tests. Before running them, install all
project dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest
```

## Interpreting model vs. market performance

Recent experiments show that the calibrated model (isotonic regression) is better calibrated than the market but does not outperform market odds for single-result picks. Use the outputs accordingly:

- **Single (seco) picks:** follow the market baseline via the `Seco_Mercado` column, which is the argmax of the implied probabilities from odds.
- **Double picks:** rely on the model to decide where to open doubles. The prediction step exposes `Entropia`, `Pmax_Modelo` and `Score_Duplo` to rank candidates and fills `Duplo_Modelo` with the two outcomes with the highest model probabilities.
- **Triple picks:** the three most uncertain matches (highest entropy/lowest `Pmax_Modelo`) are fully covered in `Triplo_Modelo`, ensuring coverage where the model and odds show little separation.
- **Anti-spray filters:** keep the existing safeguards to avoid over-distributing doubles; they are aligned with the model’s calibrated probabilities.

This division of labor lets the market guide deterministic picks while the model highlights the best opportunities to broaden coverage. The `Aposta` column applies this mix, starting from `Seco_Mercado`, assigning three triplos, and replacing five other matches with the `Duplo_Modelo` combinations selected via `Score_Duplo`.

## License

This project is licensed under the [MIT License](LICENSE).
