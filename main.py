import argparse
import subprocess
import sys


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Pipeline Loteca")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--run-backtest", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    run([py, "-m", "scripts.preprocess_data", "--log-level", args.log_level])
    run([py, "-m", "scripts.train_model", "--log-level", args.log_level])
    run([py, "-m", "scripts.predict_results", "--log-level", args.log_level])
    if args.run_backtest:
        run([py, "-m", "scripts.backtest_walk_forward", "--log-level", args.log_level])
    print("\nPipeline concluído. Arquivos em output/ e models/.")


if __name__ == "__main__":
    main()
