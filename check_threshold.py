import argparse
import os
import sys
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-info-path", default="model_info.txt")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    # Read run ID
    if not os.path.exists(args.model_info_path):
        print(f"ERROR: {args.model_info_path} not found.")
        sys.exit(1)

    with open(args.model_info_path, "r", encoding="utf-8") as f:
        run_id = f.read().strip()

    if not run_id:
        print("ERROR: model_info.txt is empty.")
        sys.exit(1)

    print(f"Run ID: {run_id}")
    
    # Find the accuracy file
    accuracy = None
    
    # Method 1: Direct path pattern
    pattern = f"mlruns/*/{run_id}/metrics/accuracy"
    accuracy_files = glob.glob(pattern, recursive=True)
    
    if accuracy_files:
        try:
            with open(accuracy_files[0], 'r') as f:
                accuracy = float(f.read().strip())
            print(f"Found accuracy via direct pattern: {accuracy_files[0]}")
        except Exception as e:
            print(f"Error reading accuracy file: {e}")
    
    # Method 2: Walk through directories if pattern didn't work
    if accuracy is None and os.path.exists("mlruns"):
        for root, dirs, files in os.walk("mlruns"):
            if run_id in root and "metrics" in root:
                for file in files:
                    if file == "accuracy":
                        accuracy_file = os.path.join(root, file)
                        try:
                            with open(accuracy_file, 'r') as f:
                                accuracy = float(f.read().strip())
                            print(f"Found accuracy via walk: {accuracy_file}")
                            break
                        except:
                            pass
                if accuracy:
                    break
    
    if accuracy is None:
        print(f"ERROR: Could not find accuracy for run {run_id}")
        print("\nAvailable runs in mlruns:")
        if os.path.exists("mlruns"):
            for root, dirs, files in os.walk("mlruns"):
                if "meta.yaml" in files:
                    meta_file = os.path.join(root, "meta.yaml")
                    try:
                        import yaml
                        with open(meta_file, 'r') as f:
                            data = yaml.safe_load(f)
                            rid = data.get('run_id', 'unknown')
                            print(f"  - {rid[:8]}...")
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