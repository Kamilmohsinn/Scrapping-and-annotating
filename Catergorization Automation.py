import os
import json
import csv
import time
import signal
import sys
from google.colab import files
from transformers import pipeline
from typing import List, Dict
import torch
from torch.amp import autocast, GradScaler

# Category list and configuration setup
CATEGORIES = [
    "Deep Learning",
    "Computer Vision",
    "Reinforcement Learning",
    "Natural Language Processing",
    "Optimization"
]

API_CALL_DELAY = 1  # Delay in seconds between API requests

class PaperProcessor:
    def __init__(self):
        self.model = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', device=0)  # Use GPU
        self.scaler = GradScaler('cuda')

    def categorize_papers(self, papers: List[Dict[str, str]]) -> List[str]:
        valid_papers = []
        for paper in papers:
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            if title and abstract:
                valid_papers.append(f"Title: {title}\nAbstract: {abstract}")
            else:
                print(f"Skipping paper due to missing 'title' or 'abstract': {paper}")
        
        if valid_papers:
            with autocast('cuda'):
                classification_results = self.model(valid_papers, CATEGORIES, batch_size=16)
            return [result['labels'][0] for result in classification_results]
        return []

def process_csv(file_path: str, processor: PaperProcessor):
    updated_data = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames + ['category'] if 'category' not in reader.fieldnames else reader.fieldnames
        papers = [row for row in reader]
    
    try:
        categories = processor.categorize_papers(papers)
        for row, category in zip(papers, categories):
            try:
                row['category'] = category
                updated_data.append(row)
            except Exception as e:
                print(f"Error processing row {row}: {e}")
                continue
    except KeyboardInterrupt:
        print("Process interrupted! Saving progress...") 
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in updated_data:
                writer.writerow({key: row.get(key, '') for key in fieldnames})
        return updated_data

    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in updated_data:
            writer.writerow({key: row.get(key, '') for key in fieldnames})

def process_json(file_path: str, processor: PaperProcessor):
    with open(file_path, 'r', encoding='utf-8') as file:
        papers = json.load(file)
    
    try:
        categories = processor.categorize_papers(papers)
        for paper, category in zip(papers, categories):
            try:
                paper['category'] = category
            except Exception as e:
                print(f"Error processing paper {paper}: {e}")
                continue
    except KeyboardInterrupt:
        print("Process interrupted! Saving progress...") 
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(papers, file, indent=4)
        return papers
    
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(papers, file, indent=4)

def main():
    processor = PaperProcessor()
    
    print("Upload your CSV and/or JSON files.")
    uploaded_files = files.upload()
    
    for filename in uploaded_files.keys():
        if filename.endswith('.csv'):
            print(f"Processing CSV: {filename}")
            process_csv(filename, processor)
    
    for filename in uploaded_files.keys():
        if filename.endswith('.json'):
            print(f"Processing JSON: {filename}")
            process_json(filename, processor)
    
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    json_files = [f for f in os.listdir() if f.endswith('.json')]
    
    if not csv_files and not json_files:
        print("No CSV or JSON files to download!")
        return
    
    archive_name = 'annotated_files.zip'
    os.system(f'zip -r {archive_name} ./*.csv ./*.json')
    
    if os.path.exists(archive_name):
        print(f"Zip completed. Downloading {archive_name}...")
        files.download(archive_name)
    else:
        print(f"Error: {archive_name} not created.")

def handle_interrupt(sig, frame):
    print("Interrupt signal received! Saving progress...")
    archive_name = 'annotated_files.zip'
    os.system(f'zip -r {archive_name} ./*.csv ./*.json')
    
    if os.path.exists(archive_name):
        files.download(archive_name)
    else:
        print(f"Error: {archive_name} not created.")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

if __name__ == "__main__":
    main()
