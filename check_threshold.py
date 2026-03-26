import os
import sys
import glob
import yaml

THRESHOLD = 0.85

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
    
    return None

def main():
    try:
        with open("model_info.txt", "r", encoding="utf-8") as f:
            run_id = f.read().strip()
        print(f"Run ID: {run_id}")
    except FileNotFoundError:
        print("Error: model_info.txt not found")
        sys.exit(1)
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            accuracy = run.data.metrics.get("accuracy")
            
            if accuracy is not None:
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Threshold: {THRESHOLD}")
                
                if accuracy >= THRESHOLD:
                    print("Deployment approved")
                    sys.exit(0)
                else:
                    print("Deployment rejected - accuracy below threshold")
                    sys.exit(1)
        except Exception as e:
            print(f"MLflow API error: {e}")
    
    accuracy = find_accuracy_in_filesystem(run_id)
    
    if accuracy is None:
        print(f"Error: Could not find accuracy for run {run_id}")
        sys.exit(1)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {THRESHOLD}")
    
    if accuracy >= THRESHOLD:
        print("Deployment approved")
        sys.exit(0)
    else:
        print("Deployment rejected - accuracy below threshold")
        sys.exit(1)

if __name__ == "__main__":
    main()