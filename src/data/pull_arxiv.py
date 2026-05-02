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
import arxiv

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg : DictConfig):

    # Setting up an user-agent
    arxiv_client = arxiv.Client(
        page_size=cfg.data.page_size,
        delay_seconds=cfg.data.delay_seconds,
        num_retries=cfg.data.num_retries
    )

    # Workaround for hydra's changing directory
    repo_root = get_original_cwd()
    arxiv_folder = os.path.normpath(os.path.join(repo_root, cfg.arxiv_folder))
    os.makedirs(arxiv_folder, exist_ok=True)
    print(f"Saving to: {arxiv_folder}")

    max_arxiv_results = cfg.data.page_size
    

    for title in tqdm(cfg.target_topics, desc="Pulling from Arxiv...", colour='green'):

        query = title.replace("_", " ").replace("(","").replace(")","").lower()
        arxiv_result = arxiv.Search(query=query, max_results=max_arxiv_results)
        
        topic_results = []

        for result in arxiv_client.results(arxiv_result):

            doc_info = {
                "title": result.title,
                "summary": result.summary,
                "url": result.entry_id,
                "published": result.published.isoformat()
            }
            topic_results.append(doc_info)
            print(f"Ok... {result.title} ({len(result.summary)} chars)")

            if topic_results:

                filename = title.lower().replace(" ", "_").replace("/", "_").replace("(","").replace(")","") + ".json"
                full_filepath = os.path.join(arxiv_folder, filename)

                with open(full_filepath, "w", encoding="utf-8") as f:
                    json.dump(topic_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
