# 🇮🇳 Vitta-Mitra: Your Agentic Financial Friend

**Empowering Indian investors through verified regulatory intelligence and real-time market awareness.**

This project is a state-of-the-art **Agentic RAG (Retrieval-Augmented Generation)** system designed to bridge the gap between complex financial regulations and everyday market activity. It serves as a comprehensive literacy tool that connects the "Rules of the Game" (SEBI/RBI/Tax) with the "Scoreboard" (Live Market Data).

---

## 🌟 The Financial Literacy Mission
Financial literacy in India is often hindered by jargon-heavy regulatory documents and fragmented data. This chatbot solves this by providing a **Unified Financial Brain** that can:
1.  **Explain the Rules**: Deep-dive into SEBI, RBI, and Income Tax Master Directions.
2.  **Monitor the Market**: Fetch live prices and news for top Indian companies.
3.  **Perform the Math**: Execute high-precision financial calculations without LLM hallucinations.
4.  **Connect the Dots**: Explain how a change in an RBI Repo rate or a SEBI regulation directly affects a stock's performance or your personal tax liability.

---

## 🚀 Key Features

### 🏛️ 4-Domain Partitioned Knowledge Base
Unlike standard RAG systems that mix all data into one database, we use a **Domain-Partitioned Architecture** to prevent "Semantic Pollution":
- **Company Domain**: Real-time fundamentals and news via yFinance API.
- **SEBI Domain**: Markets, IPOs, Insider Trading, and Listing regulations.
- **RBI Domain**: Banking, KYC, Monetary Policy, and Digital Payments.
- **Income Tax Domain**: Tax slabs, TDS, and Deductions (80C/80D).

### 🤖 Agentic Multi-Tool Orchestration
The system uses an **LLM Router Agent** (Llama3) to intelligently coordinate:
- **RAG Specialist**: Searches the partitioned local vector database for legal facts.
- **Web Search Agent**: Uses Tavily API to fetch live news and bridge knowledge gaps.
- **Calculator Agent**: A dedicated Python logic tool for error-free financial math.

### 🛡️ Compliance & Ethics First
- **Zero Bot-Scraping**: Respects SEBI/RBI anti-bot measures by utilizing a secure manual ingestion pipeline.
- **Source Transparency**: Cites the exact document name and chunk for every answer to ensure auditability.

---

## 🛠️ Tech Stack
- **Orchestration**: LangChain
- **LLM Engine**: Groq (Llama3-70b) for sub-second reasoning speed.
- **Vector Database**: ChromaDB (Semantic Search)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Live Data**: yFinance API & Tavily AI Search
- **UI**: Streamlit (Professional Dark Mode)

---

## 🎓 Project Context & Mission
In an era where my generation is increasingly participating in the markets but remains fundamentally under-literate in financial regulations and logic, this project was born out of a necessity to bridge that gap. We live in a world of information overload, but verified, actionable intelligence is rare. This chatbot serves as a dedicated, **easy-to-use** medium to democratize financial education—taking the daunting, 500-page "Master Directions" of the RBI, CBDT & SEBI and turning them into a simple, **conversational mentor**. By providing an **intuitive understanding** of complex jargon and grounding every answer in verified regulatory sources, we aim to transform passive curiosity into active, literate financial empowerment.
