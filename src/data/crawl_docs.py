"""
Crawl corpora to ingest into the knowledge vector DB
"""

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
from omegaconf import DictConfig, OmegaConf

# Modules to Build Corpus from Corpora Scrapped from the Web
from firecrawl import FirecrawlApp

# Load variables from .env into the environment
load_dotenv()

# Initializing client
firecrawl_api_key = os.getenv("FIRECRAWL_KEY")

# Initialize the app
app = FirecrawlApp(api_key=firecrawl_api_key)

# Helper functions
from datetime import datetime, date

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()  # converts to "2024-01-15T10:30:00"
        return str(obj)  # fallback: convert anything else to string
    
def ingest_docs(target_urls: list, path: str, limit: int = 10):

    os.makedirs(path, exist_ok=True)
    print(f"Saving to: {os.path.abspath(path)}")

    for url in tqdm(target_urls, desc="Crawling...", colour='green'):
            
        print(f"Currently crawling: {url}")

        try:
            response = app.crawl(
                    url, 
                    limit=limit,
                    scrape_options={
                        'formats': [
                            'markdown',
                            { 'type': 'json', 'schema': { 'type': 'object', 'properties': { 'title': { 'type': 'string' } } } }
                        ]
                    }
            )

            # CrawlResponse is a Pydantic model — serialize it
            data = response.model_dump()
            
            # Save results to a local JSON file for the vector ingestion step
            filename = url.split("//")[1].replace("/", "_") + ".json"
            full_filepath = os.path.join(path, filename)

            with open(full_filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, cls=SafeEncoder)

            n_pages = len(data.get("data", []))
            print(f"  Saved {n_pages} pages → {full_filepath}")
        
        except Exception as e:
            print(f"Oh No!... ERROR on {url}: {e}")
            continue

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg : DictConfig):

    path = cfg.data.target_folder
    urls = list(cfg.data.target_urls)
    limit = cfg.data.get("limit", 10)
    ingest_docs(urls, path, limit)
    
if __name__ == "__main__":
    main()