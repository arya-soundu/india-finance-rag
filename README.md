# 🇮🇳 India Finance RAG Chatbot

An AI-powered chatbot that answers questions grounded in Indian financial
documents — SEBI regulations, RBI policies, and BSE company annual reports.

## Features
- Answers questions from real SEBI/RBI/BSE documents — no hallucination
- Fetches live NSE/BSE stock prices via yFinance
- Cites exact source document for every answer
- Router Agent → Calculator Agent → Web Search Agent pipeline
- Full RAGAS evaluation with faithfulness and relevancy scores
- 100% free stack — no paid APIs required

## Tech Stack
| Layer | Tool | Cost |
|-------|------|------|
| LLM | Groq API — Llama3-8b | Free |
| Vector DB | ChromaDB | Free |
| Embeddings | all-MiniLM-L6-v2 | Free |
| Stock Data | yFinance NSE/BSE | Free |
| Regulations | SEBI + RBI scraper | Free |
| UI | Streamlit | Free |
| Deployment | Hugging Face Spaces | Free |

## Indian Companies Covered
- Reliance Industries (RELIANCE.NS)
- TCS (TCS.NS)
- Infosys (INFY.NS)
- HDFC Bank (HDFCBANK.NS)
- ICICI Bank (ICICIBANK.NS)

## Regulatory Documents
- SEBI LODR Regulations 2015
- SEBI Insider Trading Regulations 2015
- RBI Monetary Policy documents
- BSE Annual Reports (PDF extraction)

## Colab Setup
```
1. git clone https://github.com/YOUR_USERNAME/india-finance-rag.git
2. pip install -r requirements.txt
3. Add API keys to Colab Secrets
4. Run India_Finance_RAG_Master.ipynb cells in order
```

## Project Structure
```
india_finance_rag/
├── src/
│   ├── ingest.py       Week 2+3: Fetch docs, chunk, embed, store
│   ├── retriever.py    Week 4:   Search ChromaDB by meaning
│   ├── agents.py       Week 6:   Router + Calculator + Web Search agents
│   ├── app.py          Week 7:   Streamlit chat UI
│   └── evaluate.py     Week 8:   RAGAS evaluation pipeline
├── India_Finance_RAG_Master.ipynb
├── .env.example
├── requirements.txt
└── README.md
```
