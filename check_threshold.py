import os
import sys

import mlflow


THRESHOLD = 0.85


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI environment variable is not set")

    mlflow.set_tracking_uri(tracking_uri)

    with open("model_info.txt", "r", encoding="utf-8") as f:
        run_id = f.read().strip()

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    metric_value = run.data.metrics.get("accuracy")
    if metric_value is None:
        raise RuntimeError(f"Run {run_id} does not contain an 'accuracy' metric")

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {metric_value:.4f}")
    print(f"Threshold: {THRESHOLD:.2f}")

    if metric_value < THRESHOLD:
        print("Accuracy is below threshold. Failing deployment.")
        sys.exit(1)

    print("Accuracy meets threshold. Deployment can proceed.")


if __name__ == "__main__":
    main()
