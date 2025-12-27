"""
paths.py
Centralized path definitions for the project.
"""

import os

# Base directories
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Dataset storage
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

# Plots directory
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# Config files
QLORA_CFG_FILE_PATH = os.path.join(CODE_DIR, "config.yaml")

# Ensure directories exist
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
