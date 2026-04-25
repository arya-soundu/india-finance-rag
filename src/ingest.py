import os
import time
import logging
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

# Domain-specific collections
CHROMA_PATH = "./finrag_db"
COLLECTION_COMPANY = "company_data"
COLLECTION_SEBI = "sebi_regulations"
COLLECTION_RBI = "rbi_policies"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ingest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Target companies for analysis
INDIAN_TICKERS = [
    'RELIANCE.NS', 
    'TCS.NS', 
    'INFY.NS', 
    'HDFCBANK.NS', 
    'ICICIBANK.NS'
]

# Local data directories (for manually downloaded files)
DATA_DIRS = {
    'sebi': 'src/data/sebi',
    'rbi': 'src/data/rbi',
    'tax': 'src/data/tax'
}

# ============================================================
# UTILITIES
# ============================================================

def chunk_documents(docs: List[Dict], chunk_size: int = 1000, overlap: int = 100) -> List[Dict]:
    """
    Splits long documents into smaller, overlapping chunks for better semantic search.
    """
    logger.info(f"Chunking {len(docs)} document(s)...")
    chunks = []
    
    for doc in docs:
        text = doc['text']
        metadata = doc['metadata']
        
        # Simple character-based splitting
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            chunk_meta = metadata.copy()
            chunk_meta['chunk_id'] = f"{metadata.get('source', 'local')}_{start}"
            
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_meta
            })
            
            start += (chunk_size - overlap)
            
    return chunks

# ============================================================
# DATA FETCHERS
# ============================================================

def fetch_indian_company_info(ticker: str) -> Optional[dict]:
    """
    Fetches fundamental data and recent news for a company using yFinance.
    yFinance is a standard API and compliance-safe.
    """
    try:
        logger.info(f"[yFinance] Fetching {ticker}...")
        stock = yf.Ticker(ticker)
        info = stock.info
        
        company_name = info.get('longName', ticker)
        business_summary = info.get('longBusinessSummary', 'No summary available.')
        
        lines = [
            f"Company: {company_name}",
            f"Sector: {info.get('sector', 'N/A')}",
            f"Industry: {info.get('industry', 'N/A')}",
            f"Business Summary: {business_summary}",
            "\nKey Financial Stats:",
            f"- Market Cap: {info.get('marketCap', 'N/A')}",
            f"- PE Ratio: {info.get('trailingPE', 'N/A')}",
            f"- Dividend Yield: {info.get('dividendYield', 'N/A')}",
        ]
        
        # Fetch recent news snippets
        news = stock.news
        if news:
            lines.append("\nRecent News Snippets:")
            for item in news[:3]:
                title = item.get('title', 'No Title')
                publisher = item.get('publisher', 'N/A')
                lines.append(f"- {title} ({publisher})")
        else:
            lines.append("News not available")

        text = "\n".join(lines)
        logger.info(f"[yFinance] OK - {company_name} — {len(text):,} chars")

        return {
            'text': text,
            'metadata': {
                'source': f"yfinance_{ticker}",
                'company': company_name,
                'ticker': ticker,
                'filing_type': 'company_profile',
                'data_source': 'yFinance API'
            }
        }
    except Exception as e:
        logger.error(f"[yFinance] Error fetching {ticker}: {e}")
        return None

def fetch_local_documents(domain: str) -> List[Dict]:
    """
    Reads PDFs from local directories.
    This replaces automated scraping for SEBI and RBI to ensure compliance.
    Users should download their own PDFs and place them in the folders.
    """
    dir_path = DATA_DIRS.get(domain)
    if not dir_path or not os.path.exists(dir_path):
        return []
    
    docs = []
    import fitz # PyMuPDF
    
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(dir_path, filename)
            try:
                logger.info(f"[LOCAL] Processing {domain.upper()} file: {filename}")
                pdf = fitz.open(full_path)
                text = ""
                for page in pdf:
                    text += page.get_text()
                pdf.close()
                
                if len(text) > 100:
                    docs.append({
                        'text': text,
                        'metadata': {
                            'source': filename,
                            'company': domain.upper(),
                            'filing_type': f'{domain}_local_document',
                            'data_source': 'Local User PDF'
                        }
                    })
            except Exception as e:
                logger.error(f"[LOCAL] Error processing {filename}: {e}")
                
    return docs

# ============================================================
# CHROMADB STORAGE
# ============================================================

def store_in_chromadb(chunks: List[Dict], chroma_path: str, collection_name: str):
    """
    Stores chunks in a domain-specific ChromaDB collection.
    """
    if not chunks:
        logger.warning(f"No chunks to store for {collection_name}")
        return

    # Use the same embedding model as the retriever
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Initialize Chroma Persistent Client
    collection = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings,
        collection_name=collection_name
    )

    all_texts = [c['text'] for c in chunks]
    all_metas = [c['metadata'] for c in chunks]
    
    # Batch processing to avoid overhead
    batch_size = 100
    total_added = 0
    total_skipped = 0
    
    logger.info(f"Storing {len(chunks)} chunks in collection '{collection_name}'...")
    
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        batch_metas = all_metas[i:i+batch_size]
        
        # In a real app, you'd check for duplicates by Source ID here
        collection.add_texts(
            texts=batch_texts,
            metadatas=batch_metas
        )
        total_added += len(batch_texts)
        logger.info(f"Batch {i//batch_size + 1}: Added {len(batch_texts)} chunks")

    logger.info(f"STORAGE COMPLETE for {collection_name}. Total chunks: {total_added}")

# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("============================================================")
    logger.info("INDIA FINANCE RAG - CLEAN INGESTION PIPELINE")
    logger.info("Compliance Mode: Automated SEBI/RBI scraping disabled.")
    logger.info("============================================================")

    # Ensure directories exist
    for d in DATA_DIRS.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 1. Company Data (Safe API)
    company_docs = []
    for ticker in INDIAN_TICKERS:
        doc = fetch_indian_company_info(ticker)
        if doc:
            company_docs.append(doc)
        time.sleep(1)

    # 2. Local Documents (Compliance-Safe)
    sebi_docs = fetch_local_documents('sebi')
    rbi_docs = fetch_local_documents('rbi')
    tax_docs = fetch_local_documents('tax')

    # 3. Process & Store
    partitions = [
        (company_docs, COLLECTION_COMPANY, "Company Data"),
        (sebi_docs,    COLLECTION_SEBI,    "SEBI Regulations"),
        (rbi_docs,     COLLECTION_RBI,     "RBI Policies"),
        (tax_docs,     "tax_regulations",  "Income Tax"),
    ]

    for docs, coll_name, label in partitions:
        if docs:
            logger.info(f"\nProcessing {label}...")
            chunks = chunk_documents(docs)
            store_in_chromadb(chunks, CHROMA_PATH, coll_name)
        else:
            logger.info(f"\nSkipping {label} (No data source available)")

    logger.info("\nINGESTION FINISHED.")
    logger.info("Next: Use python -m streamlit run src/app.py to run the application.")

if __name__ == "__main__":
    main()
