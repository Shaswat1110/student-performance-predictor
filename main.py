import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n{'='*50}")
    print(f"Running {script_path}...")
    print(f"{'='*50}")
    
    # Use python executable from current environment
    result = subprocess.run([sys.executable, script_path], cwd="src")
    if result.returncode != 0:
        print(f"Error running {script_path}. Exiting.")
        sys.exit(1)
        
    print(f"Successfully completed {script_path}.\n")

if __name__ == "__main__":
    print("Starting Student Performance ML Pipeline...\n")
    
    # Define steps
    scripts = [
        "data_preprocessing.py",
        "feature_engineering.py",
        "train.py",
        "evaluate.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    print("Pipeline execution complete!")
    print("To run the web app, execute: streamlit run app.py")
