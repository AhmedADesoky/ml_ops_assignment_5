import argparse
import os
import sys
import glob
import yaml
import mlflow


def find_accuracy_in_filesystem(run_id):
    mlruns_path = "mlruns"
    
    if not os.path.exists(mlruns_path):
        return None
    
    print(f"Searching for run {run_id} in {mlruns_path}")
    
    # Method 1: Look for metrics file directly
    metrics_pattern = f"{mlruns_path}/**/metrics/accuracy.yaml"
    metric_files = glob.glob(metrics_pattern, recursive=True)
    
    for metric_file in metric_files:
        try:
            with open(metric_file, 'r') as f:
                data = yaml.safe_load(f)
                if data.get('key') == 'accuracy':
                    return float(data.get('value'))
        except:
            pass
    
    # Method 2: Look for run directory and find meta.yaml
    run_dirs = glob.glob(f"{mlruns_path}/**/{run_id}", recursive=True)
    
    for run_dir in run_dirs:
        meta_file = os.path.join(run_dir, "meta.yaml")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'metrics' in data and 'accuracy' in data['metrics']:
                        return float(data['metrics']['accuracy']['value'])
            except:
                pass
        
        # Look for metrics in metrics directory
        metrics_dir = os.path.join(run_dir, "metrics")
        if os.path.exists(metrics_dir):
            for metric_file in os.listdir(metrics_dir):
                if metric_file.endswith('.yaml'):
                    metric_path = os.path.join(metrics_dir, metric_file)
                    try:
                        with open(metric_path, 'r') as f:
                            data = yaml.safe_load(f)
                            if data.get('key') == 'accuracy':
                                return float(data.get('value'))
                    except:
                        pass
    
    # Method 3: Search all meta.yaml files
    meta_files = glob.glob(f"{mlruns_path}/**/meta.yaml", recursive=True)
    
    for meta_file in meta_files:
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
    
    # Try MLflow API first
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
    
    # If MLflow API fails, try filesystem
    if accuracy is None:
        accuracy = find_accuracy_in_filesystem(run_id)
        if accuracy is not None:
            print("Found accuracy via filesystem search")
    
    if accuracy is None:
        print(f"ERROR: accuracy metric not found for run {run_id}")
        print("\nDebug: Searching for run files...")
        
        # Debug: Show what's in mlruns
        mlruns_path = "mlruns"
        if os.path.exists(mlruns_path):
            print(f"\nContents of {mlruns_path}:")
            for root, dirs, files in os.walk(mlruns_path):
                level = root.replace(mlruns_path, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files
                    print(f'{subindent}{file}')
        
        sys.exit(1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {args.threshold:.2f}")

    if accuracy < args.threshold:
        print("RESULT: FAILED threshold check.")
        sys.exit(1)

    print("RESULT: PASSED threshold check.")


if __name__ == "__main__":
    main()