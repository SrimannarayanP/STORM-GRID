# config.py


from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


DATA_DIR = BASE_DIR/'data'
CHECKPOINTS_DIR = BASE_DIR/'checkpoints'

RAW_DIR = DATA_DIR/'raw'
UNSTRUCTURED_DIR = DATA_DIR/'raw_unstructured'
PROCESSED_DIR = DATA_DIR/'processed'
TRAINING_TENSORS_DIR = DATA_DIR/'training_tensors'
BLACKLIST_PATH = DATA_DIR/'blacklist.txt'


for directory in (RAW_DIR, UNSTRUCTURED_DIR, PROCESSED_DIR, TRAINING_TENSORS_DIR, CHECKPOINTS_DIR):
    directory.mkdir(parents = True, exist_ok = True)


# Spatial Domain (Bay of Bengal)
LON_MIN, LON_MAX = 84.0, 88.0
LAT_MIN, LAT_MAX = 18.0, 22.0
GRID_RES = 256

# Physical Scaling Consts
MAX_TOPO = 4000.0
MAX_WIND = 50.0
MAX_SURGE = 10.0
MAX_VEL = 5.0