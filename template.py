import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "sumo_files/cross.net.xml",
    "sumo_files/cross.rou.xml",
    "sumo_files/cross.sumocfg",
    "models/",
    "config.py",
    "environment.py",
    "agent.py",
    "train.py",
    "evaluate.py",
    "tensorflow_gpu_test.ipynb",
    "visualization.py"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for file: {filename}")
        
    if (not os.path.isfile(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Created empty file: {filepath}")

    else:
        logging.info(f"File already exists: {filepath}")

