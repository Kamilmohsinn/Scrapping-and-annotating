# Scrapping-and-annotating
README
Overview
This repository contains one scraper for downloading NeurIPS conference papers and their metadata:


A Python scraper for papers from 1987 to 2023.


Python Scraper
Python 3 installed. You can download it from here.

Required Python libraries: requests, beautifulsoup4, urllib3, re, json, csv, shutil, threading, concurrent.futures, google.colab.

Instructions

Python Scraper
Set Up Environment:

Ensure you have Python 3 installed.

Install the required libraries:


pip install requests beautifulsoup4 urllib3 re json csv google.colab
Run the Script:

Save the Python scraper code to a file named scraper.py.

Open a terminal/command prompt.

Navigate to the directory containing scraper.py.

Run the Python script:

python scraper.py
Running the Scripts

Python Scraper
The Python scraper allows you to specify up to 5 years between 1987 and 2023 to scrape NeurIPS papers. It will download the PDFs and save the metadata in both JSON and CSV formats. After completing the downloads, the script will compress the directories and prompt you to download the archives.

