import argparse
import os
import sys
import mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-info-path", default="model_info.txt")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("ERROR: MLFLOW_TRACKING_URI is not set.")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)

    if not os.path.exists(args.model_info_path):
        print(f"ERROR: {args.model_info_path} not found.")
        sys.exit(1)

    with open(args.model_info_path, "r", encoding="utf-8") as f:
        run_id = f.read().strip()

    if not run_id:
        print("ERROR: model_info.txt is empty.")
        sys.exit(1)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")

    if accuracy is None:
        print(f"ERROR: accuracy metric not found for run {run_id}.")
        sys.exit(1)

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {args.threshold:.2f}")

    if accuracy < args.threshold:
        print("RESULT: FAILED threshold check.")
        sys.exit(1)

    print("RESULT: PASSED threshold check.")


if __name__ == "__main__":
    main()