import argparse
import os
import sys
import mlflow
import glob
import yaml


def find_accuracy_in_filesystem(run_id):
    mlruns_path = "mlruns"
    
    if not os.path.exists(mlruns_path):
        return None
    
    for root, dirs, files in os.walk(mlruns_path):
        if "metrics" in root and run_id in root:
            for file in files:
                if file.endswith('.yaml'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            data = yaml.safe_load(f)
                            if data.get('key') == 'accuracy':
                                return float(data.get('value'))
                    except:
                        pass
    
    meta_pattern = f"{mlruns_path}/**/{run_id}/meta.yaml"
    meta_files = glob.glob(meta_pattern, recursive=True)
    
    for meta_file in meta_files:
        try:
            with open(meta_file, 'r') as f:
                data = yaml.safe_load(f)
                if 'metrics' in data and 'accuracy' in data['metrics']:
                    return float(data['metrics']['accuracy']['value'])
        except:
            pass
    
    meta_files_all = glob.glob(f"{mlruns_path}/**/meta.yaml", recursive=True)
    for meta_file in meta_files_all:
        try:
            with open(meta_file, 'r') as f:
                data = yaml.safe_load(f)
                if data.get('run_id') == run_id:
                    if 'metrics' in data and 'accuracy' in data['metrics']:
                        return float(data['metrics']['accuracy']['value'])
        except:
            pass
    
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-info-path", default="model_info.txt")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("ERROR: MLFLOW_TRACKING_URI is not set.")
        sys.exit(1)

    if not os.path.exists(args.model_info_path):
        print(f"ERROR: {args.model_info_path} not found.")
        sys.exit(1)

    with open(args.model_info_path, "r", encoding="utf-8") as f:
        run_id = f.read().strip()

    if not run_id:
        print("ERROR: model_info.txt is empty.")
        sys.exit(1)

    print(f"Run ID: {run_id}")
    print(f"Tracking URI: {tracking_uri}")
    
    accuracy = None
    
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        accuracy = run.data.metrics.get("accuracy")
        if accuracy is not None:
            print("Found accuracy via MLflow API")
    except Exception as e:
        print(f"MLflow API error: {e}")
        print("Trying filesystem fallback...")
    
    if accuracy is None:
        accuracy = find_accuracy_in_filesystem(run_id)
        if accuracy is not None:
            print("Found accuracy via filesystem search")
    
    if accuracy is None:
        print(f"ERROR: accuracy metric not found for run {run_id}.")
        print("\nAvailable runs in mlruns:")
        mlruns_path = "mlruns"
        if os.path.exists(mlruns_path):
            meta_files = glob.glob(f"{mlruns_path}/**/meta.yaml", recursive=True)
            for mf in meta_files[:5]:
                try:
                    with open(mf, 'r') as f:
                        data = yaml.safe_load(f)
                        print(f"  - {data.get('run_id')}")
                except:
                    pass
        sys.exit(1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {args.threshold:.2f}")

    if accuracy < args.threshold:
        print("RESULT: FAILED threshold check.")
        sys.exit(1)

    print("RESULT: PASSED threshold check.")


if __name__ == "__main__":
    main()