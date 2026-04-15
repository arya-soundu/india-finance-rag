# src/ingest.py
# ============================================================
# INDIA FINANCE RAG — DOCUMENT INGESTION PIPELINE
# Covers: Week 2 (fetch + chunk) + Week 3 (embed + store)
# ============================================================
#
# WHAT THIS FILE DOES (in order):
#   1. Fetches Indian company data from yFinance (NSE stocks)
#   2. Scrapes SEBI regulation pages from sebi.gov.in
#   3. Downloads and extracts text from BSE annual report PDFs
#   4. Cuts all text into 512-token chunks with 50-token overlap
#   5. Converts each chunk to a 384-number embedding vector
#   6. Stores everything in ChromaDB for fast semantic search
#
# WHY EACH STEP EXISTS:
#   Step 1-3 = Build the knowledge base (the "library")
#   Step 4   = Cut books into readable pages
#   Step 5   = Translate each page into a searchable format
#   Step 6   = File every page on the right shelf
#
# CONCEPT — What is an embedding?
#   Text: "Reliance Industries revenue grew 8% this year"
#   Embedding: [0.23, -0.87, 0.45, 0.12, ... 384 numbers]
#   Similar meaning = similar numbers = found together in search
# ============================================================

import os
import time
import hashlib
import logging
import re

import requests
from bs4 import BeautifulSoup
import pdfplumber

from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb
from sentence_transformers import SentenceTransformer

import yfinance as yf
import pandas as pd

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

INDIAN_TICKERS = [
    'RELIANCE.NS',
    'TCS.NS',
    'INFY.NS',
    'HDFCBANK.NS',
    'ICICIBANK.NS',
]

COMPANY_NAMES = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS':      'Tata Consultancy Services',
    'INFY.NS':     'Infosys',
    'HDFCBANK.NS': 'HDFC Bank',
    'ICICIBANK.NS':'ICICI Bank',
}

# BSE scrip codes — needed to search BSE filing pages
BSE_CODES = {
    'RELIANCE.NS': '500325',
    'TCS.NS':      '532540',
    'INFY.NS':     '500209',
    'HDFCBANK.NS': '500180',
    'ICICIBANK.NS':'532174',
}

# SEBI documents — publicly available HTML pages
SEBI_DOCUMENTS = [
    {
        'url': 'https://www.sebi.gov.in/legal/regulations/sep-2015/sebi-listing-obligations-and-disclosure-requirements-regulations-2015_30220.html',
        'name': 'SEBI LODR Regulations 2015',
        'type': 'sebi_regulation',
    },
    {
        'url': 'https://www.sebi.gov.in/legal/regulations/jan-2015/sebi-prohibition-of-insider-trading-regulations-2015_27683.html',
        'name': 'SEBI Insider Trading Regulations 2015',
        'type': 'sebi_regulation',
    },
    {
        'url': 'https://www.sebi.gov.in/legal/regulations/may-2018/sebi-issue-of-capital-and-disclosure-requirements-regulations-2018_38548.html',
        'name': 'SEBI ICDR Regulations 2018',
        'type': 'sebi_regulation',
    },
]

# Chunking settings
# 512 tokens = ~380 words — good balance of context vs precision
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50   # Shared tokens between adjacent chunks

# Embedding model — free, local, strong on financial text
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB path — overridden for Colab in main()
CHROMA_PATH = "./finrag_db"

# Collection name inside ChromaDB
COLLECTION_NAME = "india_finance"

# ── Logging ────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ingest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# SECTION A — DATA FETCHERS
# Three sources: yFinance, SEBI website, BSE PDFs
# ============================================================

def fetch_indian_company_info(ticker: str) -> dict:
    """
    WEEK 2 — SOURCE 1: yFinance
    ────────────────────────────
    Fetches company overview, key financial metrics,
    and recent news headlines for an NSE-listed stock.

    yFinance gives us free access to:
    - Business description (what the company does)
    - Financial ratios (P/E, EPS, market cap)
    - 52-week price range
    - Recent news headlines

    The .NS suffix tells yFinance to look on NSE (National Stock Exchange).

    Returns: dict with 'text' and 'metadata' keys, or None on failure
    """
    try:
        logger.info(f"[yFinance] Fetching {ticker}...")
        stock = yf.Ticker(ticker)
        info  = stock.info

        # Validate we got real data back
        if not info or 'longName' not in info:
            logger.warning(f"No yFinance data for {ticker}")
            return None

        company_name = info.get('longName', COMPANY_NAMES.get(ticker, ticker))

        # ── Build a rich text summary ──────────────────────
        lines = [
            f"COMPANY PROFILE: {company_name}",
            f"NSE Ticker: {ticker}",
            f"Sector: {info.get('sector', 'N/A')}",
            f"Industry: {info.get('industry', 'N/A')}",
            f"Website: {info.get('website', 'N/A')}",
            "",
            "BUSINESS DESCRIPTION:",
            info.get('longBusinessSummary', 'Not available'),
            "",
            "KEY FINANCIAL METRICS (latest available):",
        ]

        # Market cap in crores (Indian standard unit)
        # 1 crore = 10 million
        mcap = info.get('marketCap')
        if mcap:
            lines.append(f"Market Capitalisation: ₹{mcap/10_000_000:,.0f} Crores")

        lines += [
            f"P/E Ratio (TTM):      {info.get('trailingPE', 'N/A')}",
            f"EPS (TTM):            ₹{info.get('trailingEps', 'N/A')}",
            f"Revenue (TTM):        ₹{info.get('totalRevenue', 'N/A')}",
            f"Profit Margin:        {info.get('profitMargins', 'N/A')}",
            f"Return on Equity:     {info.get('returnOnEquity', 'N/A')}",
            f"Debt to Equity:       {info.get('debtToEquity', 'N/A')}",
            f"Current Ratio:        {info.get('currentRatio', 'N/A')}",
            f"52-Week High:         ₹{info.get('fiftyTwoWeekHigh', 'N/A')}",
            f"52-Week Low:          ₹{info.get('fiftyTwoWeekLow', 'N/A')}",
            f"Dividend Yield:       {info.get('dividendYield', 'N/A')}",
            f"Beta:                 {info.get('beta', 'N/A')}",
            "",
            "RECENT NEWS HEADLINES:",
        ]

        # Add up to 5 recent headlines
        try:
            for item in (stock.news or [])[:5]:
                lines.append(f"- {item.get('title', '')}")
        except Exception:
            lines.append("News not available")

        text = "\n".join(lines)
        logger.info(f"[yFinance] ✅ {company_name} — {len(text):,} chars")

        return {
            'text': text,
            'metadata': {
                'ticker':      ticker,
                'company':     company_name,
                'filing_type': 'company_overview',
                'filing_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'source':      f'NSE via yFinance — {ticker}',
                'data_source': 'yFinance',
            }
        }

    except Exception as e:
        logger.error(f"[yFinance] Failed {ticker}: {e}")
        return None


def fetch_sebi_document(doc_info: dict) -> dict:
    """
    WEEK 2 — SOURCE 2: SEBI Website
    ─────────────────────────────────
    Scrapes a SEBI regulation page from sebi.gov.in.

    Process:
    1. HTTP GET request with browser headers (avoids bot detection)
    2. BeautifulSoup parses the HTML
    3. Remove clutter (nav, scripts, footers)
    4. Extract clean text

    BeautifulSoup is a Python library that reads HTML like a tree
    and lets you pick specific parts (like just the article text).

    Returns: dict with 'text' and 'metadata', or None on failure
    """
    try:
        logger.info(f"[SEBI] Fetching: {doc_info['name']}...")

        # Mimic a real browser to avoid being blocked
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9',
        }

        # timeout=30 — give up after 30 seconds if no response
        response = requests.get(doc_info['url'], headers=headers, timeout=30)
        response.raise_for_status()  # Raises exception for 4xx/5xx errors

        # Parse HTML
        soup = BeautifulSoup(response.content, 'lxml')

        # Remove elements that add noise to our text
        for tag in soup(['script', 'style', 'nav', 'header',
                         'footer', 'aside', 'form', 'button']):
            tag.decompose()

        # Extract clean text
        # separator='\n' preserves paragraph structure
        text = soup.get_text(separator='\n', strip=True)

        # Clean up excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Add document name at the top for context
        text = f"DOCUMENT: {doc_info['name']}\nSOURCE: SEBI India\n\n{text}"

        if len(text) < 300:
            logger.warning(f"[SEBI] Too short, skipping: {doc_info['name']}")
            return None

        logger.info(f"[SEBI] ✅ {doc_info['name']} — {len(text):,} chars")

        return {
            'text': text,
            'metadata': {
                'ticker':        'N/A',
                'company':       'SEBI',
                'filing_type':   doc_info['type'],
                'filing_date':   pd.Timestamp.now().strftime('%Y-%m-%d'),
                'source':        doc_info['url'],
                'document_name': doc_info['name'],
                'data_source':   'SEBI India',
            }
        }

    except requests.Timeout:
        logger.error(f"[SEBI] Timeout: {doc_info['name']}")
        return None
    except Exception as e:
        logger.error(f"[SEBI] Failed {doc_info['name']}: {e}")
        return None


def extract_pdf_from_url(pdf_url: str, company: str, ticker: str) -> dict:
    """
    WEEK 2 — SOURCE 3 HELPER: PDF Text Extraction
    ───────────────────────────────────────────────
    Downloads a PDF from a URL and extracts text + tables.

    pdfplumber is better than PyPDF2 for financial PDFs because:
    - Handles multi-column layouts (annual reports often use 2 columns)
    - Extracts tables as structured text (important for financial statements)
    - Better handling of special characters and Indian rupee symbol ₹

    We process max 60 pages to keep runtime reasonable.

    Returns: dict with 'text' and 'metadata', or None on failure
    """
    try:
        logger.info(f"[PDF] Downloading: {pdf_url[:70]}...")

        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36'
            )
        }

        # Download PDF — timeout=120 for large annual reports
        response = requests.get(pdf_url, headers=headers, timeout=120)
        response.raise_for_status()

        # Save to temp file (pdfplumber needs a file path)
        temp_path = '/tmp/temp_report.pdf'
        with open(temp_path, 'wb') as f:
            f.write(response.content)

        # Extract text using pdfplumber
        all_text = []

        with pdfplumber.open(temp_path) as pdf:
            total_pages = len(pdf.pages)
            pages_to_read = min(60, total_pages)
            logger.info(f"[PDF] Processing {pages_to_read}/{total_pages} pages")

            for page_num in range(pages_to_read):
                page = pdf.pages[page_num]

                # Extract plain text from page
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    all_text.append(f"\n[Page {page_num+1}]\n{page_text}")

                # Extract tables separately
                # Financial statements (P&L, Balance Sheet) are tables
                tables = page.extract_tables()
                for table in (tables or []):
                    for row in (table or []):
                        if row:
                            # Convert None cells to empty string
                            row_clean = [str(c).strip() if c else '' for c in row]
                            row_text = ' | '.join(r for r in row_clean if r)
                            if len(row_text) > 10:
                                all_text.append(row_text)

        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        full_text = '\n'.join(all_text)

        if len(full_text) < 200:
            logger.warning("[PDF] Extracted text too short")
            return None

        logger.info(f"[PDF] ✅ Extracted {len(full_text):,} chars from {company}")

        return {
            'text': full_text,
            'metadata': {
                'ticker':      ticker,
                'company':     company,
                'filing_type': 'annual_report_pdf',
                'filing_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'source':      pdf_url,
                'data_source': 'BSE India PDF',
            }
        }

    except Exception as e:
        logger.error(f"[PDF] Failed for {company}: {e}")
        return None


def fetch_bse_annual_reports(ticker: str) -> list:
    """
    WEEK 2 — SOURCE 3: BSE Annual Reports
    ───────────────────────────────────────
    Searches BSE India for annual report PDF links for a company
    and extracts text from them.

    BSE filing search URL pattern:
    https://www.bseindia.com/corporates/ann.html?scripcd=XXXXX&ancat=Annual+Report

    Returns: list of document dicts (may be empty if BSE blocks us)
    """
    documents = []
    scrip_code = BSE_CODES.get(ticker)
    company_name = COMPANY_NAMES.get(ticker, ticker)

    if not scrip_code:
        logger.warning(f"[BSE] No scrip code for {ticker}")
        return documents

    try:
        logger.info(f"[BSE] Searching annual reports for {company_name}...")

        url = (
            f"https://www.bseindia.com/corporates/ann.html"
            f"?scripcd={scrip_code}&ancat=Annual+Report"
        )

        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            'Referer': 'https://www.bseindia.com/',
        }

        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code != 200:
            logger.warning(f"[BSE] Got status {response.status_code} for {ticker}")
            return documents

        soup = BeautifulSoup(response.content, 'lxml')

        # Find all PDF links on the page
        pdf_links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Look for links containing PDF and recent years
            if (href.lower().endswith('.pdf') and
                    any(yr in href for yr in ['2024', '2023', '2022', '2021'])):
                # Make absolute URL if relative
                if not href.startswith('http'):
                    href = 'https://www.bseindia.com' + href
                pdf_links.append(href)

        logger.info(f"[BSE] Found {len(pdf_links)} PDF links for {company_name}")

        # Process up to 2 annual reports per company
        for pdf_url in pdf_links[:2]:
            doc = extract_pdf_from_url(pdf_url, company_name, ticker)
            if doc:
                documents.append(doc)
            time.sleep(3)  # Be polite — don't hammer BSE servers

    except Exception as e:
        logger.error(f"[BSE] Failed for {ticker}: {e}")

    return documents


# ============================================================
# SECTION B — CHUNKING (Week 2)
# Cut long documents into pieces the LLM can process
# ============================================================

def chunk_documents(documents: list) -> list:
    """
    WEEK 2 — CHUNKING
    ──────────────────
    Splits full documents into smaller chunks.

    WHY WE CHUNK:
    - LLMs have a context window limit (max text they can read at once)
    - Smaller chunks = more precise retrieval
    - 512 tokens ≈ 380 words ≈ one topic / one financial concept

    HOW OVERLAP WORKS:
    Chunk 1: tokens 1-512
    Chunk 2: tokens 463-974   (starts 50 tokens before chunk 1 ended)
    Chunk 3: tokens 925-1436
    This ensures sentences aren't cut in half between chunks.

    Each chunk gets:
    - A unique ID (MD5 hash of text content)
    - All original document metadata
    - A context_header describing where it came from
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "।", ". ", " ", ""]
        # "।" = Devanagari full stop for Hindi content in some docs
    )

    all_chunks = []

    for doc in documents:
        if not doc or not doc.get('text'):
            continue

        raw_chunks = splitter.split_text(doc['text'])
        company = doc['metadata'].get('company', 'Unknown')
        ftype   = doc['metadata'].get('filing_type', 'document')

        logger.info(f"Chunking: {company} ({ftype}) → {len(raw_chunks)} chunks")

        for i, text in enumerate(raw_chunks):
            # Skip trivially short chunks (page numbers, blank lines, headers)
            if len(text.strip()) < 50:
                continue

            # Unique ID — same text always produces same ID (no duplicates)
            chunk_id = hashlib.md5(text.encode('utf-8')).hexdigest()

            chunk = {
                'id':   chunk_id,
                'text': text,
                'metadata': {
                    **doc['metadata'],
                    'chunk_index':   i,
                    'chunk_total':   len(raw_chunks),
                    # context_header is prepended when sending to LLM
                    # so the model always knows which document it's reading
                    'context_header': (
                        f"Source: {company} | "
                        f"Type: {ftype} | "
                        f"Date: {doc['metadata'].get('filing_date','N/A')} | "
                        f"Chunk {i+1} of {len(raw_chunks)}"
                    ),
                }
            }
            all_chunks.append(chunk)

    logger.info(f"Total chunks ready: {len(all_chunks)}")
    return all_chunks


# ============================================================
# SECTION C — EMBEDDING + STORAGE (Week 3)
# Convert chunks to numbers and store in ChromaDB
# ============================================================

def store_in_chromadb(chunks: list, chroma_path: str) -> chromadb.Collection:
    """
    WEEK 3 — EMBEDDING + CHROMADB STORAGE
    ────────────────────────────────────────
    Takes text chunks and:
    1. Converts each to a 384-dimensional embedding vector
    2. Stores text + embedding + metadata in ChromaDB

    WHAT IS AN EMBEDDING VECTOR?
    The model reads "SEBI mandates quarterly disclosures"
    and outputs: [0.23, -0.87, 0.45, ...] (384 numbers)
    These numbers encode the MEANING of the sentence.
    "Quarterly reporting requirements" would produce very similar numbers
    even though no words match — that's semantic search.

    BATCH PROCESSING:
    We process 100 chunks at a time to avoid:
    - RAM overflow on free Colab (12GB limit)
    - Losing all progress if something crashes

    DEDUPLICATION:
    We check if each chunk ID already exists before adding.
    Re-running ingest.py won't create duplicates.

    Returns the ChromaDB collection object for verification.
    """
    # Load embedding model (downloads ~90MB on first run)
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    logger.info("First run downloads ~90MB — please wait...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("✅ Embedding model loaded")

    # Connect to persistent ChromaDB
    # PersistentClient = data saved to disk, survives restarts
    os.makedirs(chroma_path, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)

    # Get or create collection
    # A collection = a named group of documents (like a table in SQL)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
        # cosine similarity = measures angle between vectors
        # better than euclidean distance for text embeddings
    )

    existing = collection.count()
    logger.info(f"ChromaDB connected. Existing chunks: {existing}")

    # Process in batches of 100
    BATCH_SIZE = 100
    total_added   = 0
    total_skipped = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

        ids       = [c['id']       for c in batch]
        texts     = [c['text']     for c in batch]
        metadatas = [c['metadata'] for c in batch]

        # Check which IDs already exist (deduplication)
        try:
            existing_result = collection.get(ids=ids, include=[])
            existing_ids = set(existing_result['ids'])
        except Exception:
            existing_ids = set()

        # Keep only chunks not already in the DB
        new_items = [
            (id_, txt, meta)
            for id_, txt, meta in zip(ids, texts, metadatas)
            if id_ not in existing_ids
        ]

        skipped = len(batch) - len(new_items)
        total_skipped += skipped

        if not new_items:
            logger.info(f"Batch {batch_num}/{total_batches} — all {skipped} already exist, skipping")
            continue

        new_ids, new_texts, new_metas = zip(*new_items)

        # ── THE EMBEDDING STEP ──────────────────────────────
        # This converts each text string to 384 numbers
        # show_progress_bar=False keeps output clean
        embeddings = model.encode(
            list(new_texts),
            show_progress_bar=False,
            batch_size=32,
            normalize_embeddings=True   # Normalise for cosine similarity
        ).tolist()

        # Add to ChromaDB
        collection.add(
            ids=list(new_ids),
            documents=list(new_texts),
            embeddings=embeddings,
            metadatas=list(new_metas)
        )

        total_added += len(new_items)
        logger.info(
            f"Batch {batch_num}/{total_batches} — "
            f"Added: {len(new_items)}, Skipped: {skipped}"
        )

    logger.info(
        f"\n✅ STORAGE COMPLETE"
        f"\n   New chunks added:  {total_added}"
        f"\n   Duplicates skipped:{total_skipped}"
        f"\n   Total in DB:       {collection.count()}"
    )

    return collection


# ============================================================
# MAIN — Runs the full pipeline
# ============================================================

def main():
    """
    Full ingestion pipeline orchestrator.
    Detects Colab vs local and sets paths accordingly.
    """
    # Detect environment
    try:
        import google.colab
        chroma_path = "/content/drive/MyDrive/Finance_RAG/finrag_db"
        os.makedirs("/content/drive/MyDrive/Finance_RAG/logs", exist_ok=True)
        in_colab = True
        logger.info("Running on Google Colab — using Drive paths")
    except ImportError:
        chroma_path = CHROMA_PATH
        in_colab = False
        logger.info("Running locally")

    os.makedirs(chroma_path, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    logger.info("=" * 60)
    logger.info("INDIA FINANCE RAG — INGESTION PIPELINE")
    logger.info(f"Tickers: {INDIAN_TICKERS}")
    logger.info("=" * 60)

    all_documents = []

    # ── 1. yFinance company data ─────────────────────────
    logger.info("\n📊 Fetching Indian company data (yFinance)...")
    for ticker in INDIAN_TICKERS:
        doc = fetch_indian_company_info(ticker)
        if doc:
            all_documents.append(doc)
        time.sleep(1)

    # ── 2. SEBI regulations ──────────────────────────────
    logger.info("\n📋 Fetching SEBI regulations...")
    for sebi_doc in SEBI_DOCUMENTS:
        doc = fetch_sebi_document(sebi_doc)
        if doc:
            all_documents.append(doc)
        time.sleep(2)

    # ── 3. BSE annual reports (best effort) ──────────────
    logger.info("\n📄 Fetching BSE annual reports (may fail if BSE blocks)...")
    for ticker in INDIAN_TICKERS[:2]:   # Start with 2 to test
        bse_docs = fetch_bse_annual_reports(ticker)
        all_documents.extend(bse_docs)
        time.sleep(3)

    logger.info(f"\nTotal documents collected: {len(all_documents)}")

    if not all_documents:
        logger.error("No documents fetched — check internet connection")
        return

    # ── 4. Chunk all documents ───────────────────────────
    logger.info("\n✂️  Chunking documents...")
    all_chunks = chunk_documents(all_documents)

    # ── 5. Embed and store ───────────────────────────────
    logger.info("\n💾 Embedding and storing in ChromaDB...")
    collection = store_in_chromadb(all_chunks, chroma_path)

    logger.info(f"\n🎉 INGESTION COMPLETE! DB has {collection.count()} chunks")
    logger.info("Next: run retriever.py to test searching")


if __name__ == "__main__":
    main()
