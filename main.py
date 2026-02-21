import subprocess
import sys


def run(cmd):
    print("Executando:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run([sys.executable, "scripts/preprocess_data.py"])
    run([sys.executable, "scripts/train_model.py"])
    run([sys.executable, "scripts/predict_results.py"])
    print("Pipeline concluído. Veja output/predictions.csv")
