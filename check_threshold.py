import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

THRESHOLD = 0.85

def main() -> None:
    # Get tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI environment variable is not set")
    
    print(f"MLflow Tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Read run ID from file
    try:
        with open("model_info.txt", "r", encoding="utf-8") as f:
            run_id = f.read().strip()
        print(f"Run ID from file: {run_id}")
    except FileNotFoundError:
        print("Error: model_info.txt not found")
        sys.exit(1)
    
    # Try to find the run
    client = MlflowClient()
    
    try:
        # First try to get the run directly
        run = client.get_run(run_id)
        print(f"Found run: {run_id}")
    except Exception as e:
        print(f"Could not find run {run_id} directly: {e}")
        
        # If direct lookup fails, try to search for the run
        print("Attempting to search for the run...")
        try:
            # List all experiments
            experiments = client.search_experiments()
            found = False
            
            for exp in experiments:
                # Search for runs in this experiment
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string=f"run_id = '{run_id}'"
                )
                
                if len(runs) > 0:
                    print(f"Found run in experiment: {exp.name}")
                    # Get the run details
                    run = client.get_run(run_id)
                    found = True
                    break
            
            if not found:
                # Try to list all runs as a fallback
                print("Run not found in any experiment. Listing recent runs:")
                for exp in experiments[:2]:  # Check first 2 experiments
                    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=5)
                    if len(runs) > 0:
                        print(f"Experiment: {exp.name}")
                        for idx, row in runs.iterrows():
                            print(f"  - Run ID: {row['run_id']}, Accuracy: {row.get('metrics.accuracy', 'N/A')}")
                
                raise RuntimeError(f"Run {run_id} not found in MLflow tracking store")
                
        except Exception as search_error:
            print(f"Error searching for run: {search_error}")
            raise RuntimeError(f"Cannot find run {run_id}")
    
    # Get accuracy metric
    metric_value = run.data.metrics.get("accuracy")
    if metric_value is None:
        # Check for alternative metric names
        metric_value = run.data.metrics.get("test_accuracy")
        if metric_value is None:
            print(f"Available metrics: {list(run.data.metrics.keys())}")
            raise RuntimeError(f"Run {run_id} does not contain an 'accuracy' metric")
    
    print(f"Run ID: {run_id}")
    print(f"Accuracy: {metric_value:.4f}")
    print(f"Threshold: {THRESHOLD:.2f}")
    
    if metric_value < THRESHOLD:
        print(f"Accuracy ({metric_value:.4f}) is below threshold ({THRESHOLD:.2f}). Failing deployment.")
        sys.exit(1)
    
    print(f"Accuracy ({metric_value:.4f}) meets threshold ({THRESHOLD:.2f}). Deployment can proceed.")
    sys.exit(0)

if __name__ == "__main__":
    main()