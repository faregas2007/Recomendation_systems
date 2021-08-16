import logging
import sys, os
from pathlib import Path

import mlflow


# Directories
base_dir = Path(os.getcwd()).parent.absolute()
config_dir = Path(base_dir, "config")
logs_dir = Path(base_dir, "logs")
data_dir = Path(base_dir, "data")
model_dir = Path(base_dir, "model")
stores_dir = Path(base_dir, "stores")

# Local stores

# Create dirs
logs_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
stores_dir.mkdir(parents=True, exist_ok=True)
feature_store.mkdir(parents=True, exist_ok=True)
model_registry.mkdir(parents=True, exist_ok=True)


# mlflow model registry (for uri/mlflow_id tracking)
mlflow.set_tracking_uri("file://"+str(model_registry.absolute()))

