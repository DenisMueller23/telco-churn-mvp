import json
import random
import numpy as np
from pathlib import Path

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    
def save_threshold(threshold: float, path: str = "models/threshold.json"):
    """Save the threshold value to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"threshold": threshold}, f)
    print(f"✓ Threshold saved: {threshold:.4f} ->{path}")
    
    
def load_threshold(path: str="models/threshold.json")->float:
    """Load threshold form JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data