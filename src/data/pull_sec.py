# Session Setup
# -------------

# Boilerplate Modules
import os
import sys
import json

from dotenv import load_dotenv

import time

from tqdm import tqdm

# Config Files
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# Modelus to Build Corpus from Corpora Scrapped from the Web
from sec_edgar_downloader import Downloader

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg : DictConfig):

    repo_root = get_original_cwd()
    sec_folder = os.path.normpath(os.path.join(repo_root, cfg.sec_folder))
    
    os.makedirs(sec_folder, exist_ok=True)
    print(f"SEC filings will be saved to: {sec_folder}")

    # Setting up an user-agent
    dl = Downloader(
        "RZs_RAG_Project", 
        "rzambrano@gmail.com",
        download_folder=sec_folder
        )

    # Pulling 10-K Records
    dl.get("10-K", "JPM", limit=2)
    dl.get("10-K", "COF", limit=2)

if __name__ == "__main__":
    main()
