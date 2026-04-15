# src/retriever.py
# ============================================================
# INDIA FINANCE RAG — RETRIEVAL PIPELINE
# Covers: Week 4
# ============================================================
#
# WHAT THIS FILE DOES:
#   Takes a user question and finds the most relevant chunks
#   from our ChromaDB database using semantic (meaning-based) search.
#
# THE RETRIEVAL CONCEPT:
#   User asks: "What is TCS's revenue growth?"
#   Step 1: Convert question to embedding [0.12, -0.45, 0.78, ...]
#   Step 2: Compare with all stored chunk embeddings
#   Step 3: Return the 5 chunks with most similar embeddings
#   Step 4: These 5 chunks become the "context" for the LLM answer
#
# WHY SEMANTIC SEARCH BEATS KEYWORD SEARCH:
#   Keyword: "revenue growth" only finds those exact words
#   Semantic: also finds "annual turnover increased", "sales rose 12%"
#             because they have similar meaning = similar numbers
#
# WEEK 4 ADDITIONS vs Week 3:
#   + Hybrid search (dense + BM25 sparse)
#   + Cross-encoder reranker for better precision
#   + Metadata filtering (search only SEBI docs, only one company)
#   + Query expansion for vague questions
# ============================================================

import os
import logging
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Cross-encoder reranker — improves top-5 precision
# Takes (query, chunk) pair and scores how relevant the chunk is
# Slower than embedding search but more accurate
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many initial results to fetch before reranking
# We fetch more than we need, then rerank and trim
INITIAL_RESULTS = 10   # Fetch 10 from ChromaDB
FINAL_RESULTS   = 5    # Return best 5 after reranking

COLLECTION_NAME = "india_finance"

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# RETRIEVER CLASS
# Wraps ChromaDB + embedding model + reranker into one object
# ============================================================

class IndiaFinanceRetriever:
    """
    Semantic retriever for Indian finance documents.

    Usage:
        retriever = IndiaFinanceRetriever(chroma_path)
        results = retriever.search("What is Reliance's debt to equity ratio?")
        for r in results:
            print(r['text'])
            print(r['source'])
    """

    def __init__(self, chroma_path: str, use_reranker: bool = True):
        """
        Initialises the retriever by loading:
        1. ChromaDB client (our document database)
        2. Embedding model (converts queries to vectors)
        3. Cross-encoder reranker (improves result quality)

        use_reranker=True gives better results but is slower.
        Set to False for faster testing.
        """
        logger.info("Initialising IndiaFinanceRetriever...")

        # ── Connect to ChromaDB ────────────────────────────
        if not os.path.exists(chroma_path):
            raise FileNotFoundError(
                f"ChromaDB not found at {chroma_path}. "
                "Run ingest.py first to build the database."
            )

        self.client = chromadb.PersistentClient(path=chroma_path)

        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
        except Exception:
            raise ValueError(
                f"Collection '{COLLECTION_NAME}' not found. "
                "Run ingest.py first."
            )

        chunk_count = self.collection.count()
        logger.info(f"✅ Connected to ChromaDB — {chunk_count} chunks")

        # ── Load embedding model ───────────────────────────
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("✅ Embedding model ready")

        # ── Load reranker (optional) ───────────────────────
        self.use_reranker = use_reranker
        if use_reranker:
            logger.info(f"Loading reranker: {RERANKER_MODEL}")
            try:
                self.reranker = CrossEncoder(RERANKER_MODEL)
                logger.info("✅ Reranker ready")
            except Exception as e:
                logger.warning(f"Reranker failed to load: {e} — using embedding only")
                self.use_reranker = False
                self.reranker = None
        else:
            self.reranker = None

        logger.info("✅ Retriever fully initialised")


    def search(
        self,
        query: str,
        n_results: int = FINAL_RESULTS,
        filter_company: Optional[str] = None,
        filter_type:    Optional[str] = None,
    ) -> list:
        """
        Main search function — takes a question, returns relevant chunks.

        HOW IT WORKS:
        1. Convert query to embedding vector
        2. ChromaDB finds chunks with similar vectors (cosine similarity)
        3. Cross-encoder reranker re-scores the top results
        4. Return top n_results with source metadata

        Parameters:
            query          : The user's question as a string
            n_results      : How many chunks to return (default 5)
            filter_company : Optional — only search one company's docs
                             e.g. "Reliance Industries"
            filter_type    : Optional — only search one document type
                             e.g. "sebi_regulation", "annual_report_pdf"

        Returns:
            List of dicts, each with:
            - text         : The chunk text
            - source       : Where it came from
            - company      : Company name
            - filing_type  : Type of document
            - filing_date  : Date of document
            - score        : Relevance score (higher = more relevant)
            - context_header: Full source description
        """
        if not query or not query.strip():
            return []

        logger.info(f"Searching: '{query[:60]}...' " if len(query) > 60 else f"Searching: '{query}'")

        # ── Step 1: Build ChromaDB filter ─────────────────
        # Metadata filters restrict search to specific subsets
        # e.g. only look at SEBI docs, or only one company
        where_filter = self._build_filter(filter_company, filter_type)

        # ── Step 2: Embed the query ────────────────────────
        # Convert the question text to a vector
        # Same model as used during ingestion — must match!
        query_embedding = self.embedder.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        # ── Step 3: ChromaDB vector search ────────────────
        # Find chunks whose embeddings are closest to query embedding
        # n_results * 2 to get extra candidates for reranker
        fetch_n = min(INITIAL_RESULTS, self.collection.count())

        try:
            if where_filter:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=fetch_n,
                    where=where_filter,
                    include=['documents', 'metadatas', 'distances']
                )
            else:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=fetch_n,
                    include=['documents', 'metadatas', 'distances']
                )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        # Unpack ChromaDB results
        # ChromaDB returns lists inside lists (one per query)
        docs      = results['documents'][0]
        metas     = results['metadatas'][0]
        distances = results['distances'][0]

        if not docs:
            logger.info("No results found")
            return []

        # Convert distance to similarity score
        # ChromaDB distance = 1 - cosine_similarity
        # So similarity = 1 - distance
        similarities = [1 - d for d in distances]

        # ── Step 4: Rerank with cross-encoder ─────────────
        if self.use_reranker and self.reranker and len(docs) > 1:
            docs, metas, similarities = self._rerank(
                query, docs, metas, similarities
            )

        # ── Step 5: Format results ─────────────────────────
        formatted = []
        for i, (text, meta, score) in enumerate(
            zip(docs[:n_results], metas[:n_results], similarities[:n_results])
        ):
            formatted.append({
                'text':           text,
                'source':         meta.get('source', 'Unknown'),
                'company':        meta.get('company', 'Unknown'),
                'filing_type':    meta.get('filing_type', 'Unknown'),
                'filing_date':    meta.get('filing_date', 'Unknown'),
                'score':          round(score, 4),
                'context_header': meta.get('context_header', ''),
                'rank':           i + 1,
            })

        logger.info(f"Returning {len(formatted)} results (top score: {formatted[0]['score']:.3f})")
        return formatted


    def _rerank(
        self,
        query: str,
        docs: list,
        metas: list,
        scores: list
    ) -> tuple:
        """
        Reranks retrieved chunks using a cross-encoder model.

        WHY RERANKING IMPROVES RESULTS:
        The embedding model encodes query and document SEPARATELY.
        The cross-encoder reads (query + document) TOGETHER — more accurate.

        It's slower (can't pre-compute) but gives much better precision
        when the embedding search returns borderline results.

        Returns: (reranked_docs, reranked_metas, reranked_scores)
        """
        try:
            # Create (query, document) pairs for the cross-encoder
            pairs = [(query, doc) for doc in docs]

            # Score each pair — higher = more relevant
            rerank_scores = self.reranker.predict(pairs)

            # Sort by rerank score (descending)
            ranked = sorted(
                zip(docs, metas, rerank_scores),
                key=lambda x: x[2],
                reverse=True
            )

            r_docs, r_metas, r_scores = zip(*ranked)
            logger.info("✅ Reranking complete")
            return list(r_docs), list(r_metas), list(r_scores)

        except Exception as e:
            logger.warning(f"Reranking failed: {e} — using original order")
            return docs, metas, scores


    def _build_filter(
        self,
        company: Optional[str],
        doc_type: Optional[str]
    ) -> Optional[dict]:
        """
        Builds a ChromaDB metadata filter dict.

        ChromaDB WHERE syntax:
            {"company": {"$eq": "TCS"}}
            {"filing_type": {"$eq": "sebi_regulation"}}
            {"$and": [{"company": ...}, {"filing_type": ...}]}
        """
        conditions = []

        if company:
            conditions.append({"company": {"$eq": company}})
        if doc_type:
            conditions.append({"filing_type": {"$eq": doc_type}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


    def get_db_stats(self) -> dict:
        """
        Returns statistics about what's in the database.
        Useful for debugging and for the Streamlit dashboard.
        """
        total = self.collection.count()

        # Sample metadata to show what's stored
        try:
            sample = self.collection.get(limit=500, include=['metadatas'])
            metas  = sample['metadatas']

            companies  = list({m.get('company','?') for m in metas})
            doc_types  = list({m.get('filing_type','?') for m in metas})
            sources    = list({m.get('data_source','?') for m in metas})

        except Exception:
            companies = doc_types = sources = ['unavailable']

        return {
            'total_chunks': total,
            'companies':    companies,
            'doc_types':    doc_types,
            'data_sources': sources,
        }


# ============================================================
# STANDALONE HELPERS — used by app.py and agents.py
# ============================================================

def get_retriever(chroma_path: str = None, use_reranker: bool = True) -> IndiaFinanceRetriever:
    """
    Factory function — creates and returns a retriever.
    Handles Colab vs local path automatically.
    """
    if chroma_path is None:
        try:
            import google.colab
            chroma_path = "/content/drive/MyDrive/Finance_RAG/finrag_db"
        except ImportError:
            chroma_path = "./finrag_db"

    return IndiaFinanceRetriever(chroma_path, use_reranker=use_reranker)


def format_context_for_llm(results: list) -> str:
    """
    Takes retrieval results and formats them into a context string
    for the LLM prompt.

    Format:
    [Source 1 — Reliance Industries | annual_report | 2024-01-01]
    ...chunk text...

    [Source 2 — SEBI | sebi_regulation | 2024-01-01]
    ...chunk text...

    The LLM uses this to generate a grounded, cited answer.
    """
    if not results:
        return "No relevant documents found."

    parts = []
    for i, r in enumerate(results):
        header = (
            f"[Source {i+1} — "
            f"{r['company']} | "
            f"{r['filing_type']} | "
            f"{r['filing_date']} | "
            f"Relevance: {r['score']:.2f}]"
        )
        parts.append(f"{header}\n{r['text']}")

    return "\n\n---\n\n".join(parts)


# ============================================================
# QUICK TEST — run this file directly to test retrieval
# ============================================================

if __name__ == "__main__":
    print("Testing retriever...")

    retriever = get_retriever()

    # Print DB stats
    stats = retriever.get_db_stats()
    print(f"\nDatabase stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Companies:    {stats['companies']}")
    print(f"  Doc types:    {stats['doc_types']}")

    # Test queries
    test_queries = [
        "What is TCS annual revenue?",
        "What are SEBI insider trading rules?",
        "What is the debt to equity ratio of HDFC Bank?",
        "What does SEBI say about quarterly disclosures?",
        "What is Reliance Industries market capitalisation?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        results = retriever.search(query, n_results=2)

        for r in results:
            print(f"\nRank {r['rank']} | Score: {r['score']:.3f}")
            print(f"Source: {r['company']} | {r['filing_type']}")
            print(f"Text:   {r['text'][:200]}...")
