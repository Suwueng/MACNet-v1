import os
import subprocess
import sys
import json

# =============================================================================
# Configuration Source
# =============================================================================
# Specify the experiment key from ExperimentSetting.json to run
TARGET_EXPERIMENT = "Exp5_AccretionConvNet"  # Change this to the desired experiment key

def load_config(exp_key):
    """
    Load configuration from ExperimentSetting.json and flatten it.
    """
    # Path to the JSON configuration file (Located in .config folder, sibling to Experiment folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(current_dir, "..", ".config", "ExperimentSetting.json"))
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON configuration file: {e}")
        sys.exit(1)
        
    if exp_key not in full_config:
        print(f"Error: Experiment key '{exp_key}' not found in {config_path}")
        print(f"Available keys: {list(full_config.keys())}")
        sys.exit(1)
        
    exp_config = full_config[exp_key]
    
    # Flatten the configuration
    # 展平配置
    flat_config = {"exp_name": exp_key} # Use the key as the experiment name
    
    for section, params in exp_config.items():
        if isinstance(params, dict):
            for key, value in params.items():
                if key == "description":
                    continue
                # Handle specific key mappings if the script arguments differ from JSON keys
                if key == "cache_file":
                    flat_config["cache_dir"] = value
                else:
                    flat_config[key] = value
        else:
             # Description or other top level simple types
             if section != "description": 
                 flat_config[section] = params
                 
    return flat_config

# =============================================================================
# Execution Script
# 执行脚本
# =============================================================================

def main():
    # Load Configuration
    print(f"Loading configuration for: {TARGET_EXPERIMENT}")
    config = load_config(TARGET_EXPERIMENT)

    # Locate the training script in the same directory
    # 定位同一目录下的训练脚本
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "train.py")

    if not os.path.exists(script_path):
        print(f"Error: Could not find {script_path}")
        return

    # Build the command
    cmd = [sys.executable, script_path]

    for key, value in config.items():
        # Handle boolean flags (store_true arguments)
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        # Handle None values (skip)
        elif value is None:
            continue
        # Handle list/tuple (e.g., stage_depths)
        elif isinstance(value, (list, tuple)):
            cmd.append(f"--{key}")
            cmd.append(",".join(str(item) for item in value))
        # Handle standard key-value arguments
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    # Print the command for verification
    print("=" * 60)
    print("Running Experiment with config:")
    print("=" * 60)
    for k, v in config.items():
        print(f"{k:<20}: {v}")
    print("-" * 60)
    
    # Run
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Process exited with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")

if __name__ == "__main__":
    main()
