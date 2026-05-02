
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
import wikipediaapi

# Setting up an user-agent
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='rics-rag-project/1.0 (rzambrano@gmail.com)'
)

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg : DictConfig):

    # Workaround for hydra's changing directory
    repo_root = get_original_cwd()
    wiki_folder = os.path.normpath(os.path.join(repo_root, cfg.wiki_folder))
    os.makedirs(wiki_folder, exist_ok=True)
    print(f"Saving to: {wiki_folder}")

    wikipedia_docs = []

    for title in tqdm(cfg.target_topics, desc="Pulling from Wikipedia...", colour='green'):
        page = wiki.page(title)
        if page.exists():
            wikipedia_docs.append({"title": title, "text": page.text})
            print(f"Ok... {title} ({len(page.text)} chars)")
        else:
            print(f"Oh No!... {title} — not found")
            continue # To skip saving below
        print("-" * 50)
    
        # Save results to a local JSON file for the vector ingestion step
        
        filename = title.lower().replace(" ", "_").replace("/", "_").replace("(","").replace(")","") + ".json"
        full_filepath = os.path.join(wiki_folder, filename)

        with open(full_filepath, "w", encoding="utf-8") as f:
            json.dump(page.text, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()