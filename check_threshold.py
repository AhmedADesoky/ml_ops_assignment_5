import argparse
import os
import sys
import glob


def read_accuracy_file(file_path):
    """Read accuracy from MLflow metrics file (handles both formats)"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        print(f"File has {len(lines)} lines")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            print(f"Processing line: {repr(line)}")
            
            # Try format: timestamp value step
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # Try the second part as the value
                    value = float(parts[1])
                    if 0 <= value <= 1:
                        return value
                except:
                    pass
            
            # Try format: timestamp value (2 parts)
            if len(parts) == 2:
                try:
                    value = float(parts[1])
                    if 0 <= value <= 1:
                        return value
                except:
                    pass
            
            # Try format: value
            try:
                value = float(line)
                if 0 <= value <= 1:
                    return value
            except:
                pass
            
            # Try each part in the line
            for part in parts:
                try:
                    value = float(part)
                    if 0 <= value <= 1:
                        return value
                except:
                    pass
        
        return None
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


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
    accuracy_pattern = f"mlruns/*/{run_id}/metrics/accuracy"
    accuracy_files = glob.glob(accuracy_pattern, recursive=True)
    
    if not accuracy_files:
        print(f"ERROR: Could not find accuracy file for run {run_id}")
        print("\nSearching in all accuracy files...")
        all_accuracy = glob.glob("mlruns/*/*/metrics/accuracy", recursive=True)
        for acc in all_accuracy:
            print(f"  Found: {acc}")
        sys.exit(1)
    
    # Read and parse accuracy
    accuracy = read_accuracy_file(accuracy_files[0])
    
    if accuracy is None:
        print(f"ERROR: Could not extract accuracy from file")
        print(f"File: {accuracy_files[0]}")
        with open(accuracy_files[0], 'r') as f:
            print(f"Full content:\n{f.read()}")
        sys.exit(1)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {args.threshold:.2f}")

    if accuracy < args.threshold:
        print("RESULT: FAILED threshold check.")
        sys.exit(1)

    print("RESULT: PASSED threshold check.")


if __name__ == "__main__":
    main()