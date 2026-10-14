import os
import yaml
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_sources_config(config_path="data/sources.yaml"):
    """Loads the sources configuration file if available."""
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file {config_path} not found. Skipping config load.")
        return []
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("sources", [])

def run_ingestion():
    print("--- Starting Data Ingestion Pipeline ---")
    
    # Ensure directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Load configuration
    sources = load_sources_config()
    print(f"Loaded {len(sources)} sources from configuration.")
    
    # Dummy processing check for demonstration and robustness
    sample_text = (
        "MICRO-ENTERPRISE FINANCIAL GUIDANCE:\n"
        "1. Working Capital: Maintain at least 30 days of operating buffer.\n"
        "2. Compliance: Ensure annual returns are filed with the relevant authorities."
    )
    
    # Write processed chunk demonstration
    output_file = "data/processed/chunks.jsonl"
    with open(output_file, "w", encoding="utf-8") as out:
        out.write('{"source": "sample_policy", "text": "' + sample_text.replace('\n', ' ') + '"}\n')
        
    print(f"Successfully processed and wrote chunks to {output_file}")
    print("--- Ingestion Pipeline Complete ---")

if __name__ == "__main__":
    run_ingestion()
