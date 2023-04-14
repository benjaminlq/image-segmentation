import torch
import os

MAIN_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(MAIN_PATH, "data")
ARTIFACT_PATH = os.path.join(MAIN_PATH, "artifacts")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = os.cpu_count()
