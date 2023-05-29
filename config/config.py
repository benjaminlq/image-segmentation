import os 

MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(MAIN_DIR, "data")
CKPT_DIR = os.path.join(MAIN_DIR, "ckpt")

NUM_WORKERS = os.cpu_count()