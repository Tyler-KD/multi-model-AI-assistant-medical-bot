import sys
import json
import logging
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add project root to path if needed
sys.path.append(str(Path(__file__).parent.parent))

# Import your components
from agents.rag_agent import MedicalRAG
from config import Config

import argparse

# Initialize parser
parser = argparse.ArgumentParser(description="Process some command-line arguments.")

# Add arguments
parser.add_argument("--file", type=str, required=False, help="Enter file path to ingest")
parser.add_argument("--dir", type=str, required=False, help="Enter directory path of files to ingest")

# Parse arguments
args = parser.parse_args()

# Load configuration
config = Config()

rag = MedicalRAG(config)

# document ingestion
def data_ingestion():

    if args.file:
        # Define path to file
        file_path = args.file
        # Process and ingest the file
        result = rag.ingest_file(file_path)
    elif args.dir:
        # Define path to dir
        dir_path = args.dir
        # Process and ingest the files
        result = rag.ingest_directory(dir_path)

    print("Ingestion result:", json.dumps(result, indent=2))

    return result["success"]

# Run tests
if __name__ == "__main__":
   
    print("\nIngesting document(s)...")

    ingestion_success = data_ingestion()
    
    if ingestion_success:
        print("\nSuccessfully ingested the documents.")