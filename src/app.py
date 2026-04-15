# src/app.py
# ============================================================
# INDIA FINANCE RAG — STREAMLIT CHAT UI
# Covers: Week 7
# ============================================================
#
# WHAT THIS FILE DOES:
#   Builds the full chat interface users interact with.
#
# FEATURES:
#   - Chat input with streaming-style responses
#   - Source citation panel (expandable per answer)
#   - Live NSE stock price ticker (top of page)
#   - Confidence score displayed per answer
#   - Agent type indicator (RAG / Calculator / Web Search)
#   - Conversation history (persists during session)
#   - DB stats sidebar (what documents are loaded)
#
# HOW TO RUN:
#   streamlit run src/app.py
#   Then open http://localhost:8501 in your browser
#
# HOW TO RUN ON COLAB:
#   See the notebook — uses pyngrok to tunnel the port
# ============================================================

import os
import sys
import time
import logging

import streamlit as st
import yfinance as yf

# Add parent dir to path so we can import from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import process_query
from src.retriever import get_retriever

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# PAGE CONFIGURATION
# Must be the first Streamlit call in the file
# ============================================================

st.set_page_config(
    page_title="India Finance RAG",
    page_icon="🇮🇳",
    layout="wide",             # Use full width
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS — Makes it look professional
# ============================================================

st.markdown("""
<style>
    /* Main chat container */
    .main { padding: 0rem 1rem; }

    /* User message bubble */
    .user-message {
        background-color: #1a73e8;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 75%;
        float: right;
        clear: both;
    }

    /* Bot message bubble */
    .bot-message {
        background-color: #f1f3f4;
        color: #202124;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        float: left;
        clear: both;
    }

    /* Source citation box */
    .source-box {
        background-color: #e8f0fe;
        border-left: 4px solid #1a73e8;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85em;
    }

    /* Confidence indicator */
    .confidence-high  { color: #0f9d58; font-weight: bold; }
    .confidence-med   { color: #f4b400; font-weight: bold; }
    .confidence-low   { color: #db4437; font-weight: bold; }

    /* Agent badge */
    .agent-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: bold;
        margin-left: 8px;
    }
    .badge-rag        { background: #e8f5e9; color: #2e7d32; }
    .badge-calculator { background: #fff3e0; color: #e65100; }
    .badge-websearch  { background: #e3f2fd; color: #1565c0; }
    .badge-direct     { background: #f3e5f5; color: #6a1b9a; }

    /* Stock ticker strip */
    .ticker-strip {
        background: #1a1a2e;
        color: #00d4aa;
        padding: 6px 12px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def load_retriever():
    """
    Loads the retriever ONCE and caches it.

    @st.cache_resource means:
    - First call: creates the retriever (takes ~30 seconds)
    - All subsequent calls: returns the cached version instantly
    - Without this, every message reload would re-load the model!
    """
    try:
        # Detect Colab vs local
        try:
            import google.colab
            chroma_path = "/content/drive/MyDrive/Finance_RAG/finrag_db"
        except ImportError:
            chroma_path = "./finrag_db"

        retriever = get_retriever(chroma_path=chroma_path, use_reranker=True)
        return retriever, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=300)   # Cache for 5 minutes — prices change
def get_live_prices() -> dict:
    """
    Fetches current NSE stock prices for our 5 companies.
    Cached for 5 minutes to avoid hammering yFinance.
    """
    tickers = {
        'RELIANCE.NS': 'Reliance',
        'TCS.NS':      'TCS',
        'INFY.NS':     'Infosys',
        'HDFCBANK.NS': 'HDFC Bank',
        'ICICIBANK.NS':'ICICI Bank',
    }

    prices = {}
    for ticker, name in tickers.items():
        try:
            stock = yf.Ticker(ticker)
            info  = stock.info
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            change_pct = info.get('regularMarketChangePercent', 0)
            if price:
                prices[name] = {
                    'price':  price,
                    'change': round(change_pct, 2)
                }
        except Exception:
            pass

    return prices


def format_confidence_badge(confidence: float) -> str:
    """Returns an HTML confidence badge with colour coding."""
    if confidence >= 0.75:
        css_class = "confidence-high"
        label = f"High ({confidence:.0%})"
    elif confidence >= 0.50:
        css_class = "confidence-med"
        label = f"Medium ({confidence:.0%})"
    else:
        css_class = "confidence-low"
        label = f"Low ({confidence:.0%})"
    return f'<span class="{css_class}">● {label}</span>'


def format_agent_badge(agent: str) -> str:
    """Returns a coloured badge showing which agent answered."""
    labels = {
        'rag':        ('RAG Search',   'badge-rag'),
        'calculator': ('Calculator',   'badge-calculator'),
        'websearch':  ('Web Search',   'badge-websearch'),
        'direct':     ('Direct',       'badge-direct'),
    }
    label, css = labels.get(agent, ('Unknown', 'badge-rag'))
    return f'<span class="agent-badge {css}">{label}</span>'


# ============================================================
# SESSION STATE INITIALISATION
# Streamlit re-runs the whole script on every interaction.
# st.session_state persists data across re-runs.
# ============================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []
    # Add a welcome message on first load
    st.session_state.messages.append({
        'role':       'assistant',
        'content':    (
            "Namaste! 🇮🇳 I'm your India Finance RAG assistant.\n\n"
            "I can answer questions about:\n"
            "- **Indian companies**: Reliance, TCS, Infosys, HDFC Bank, ICICI Bank\n"
            "- **SEBI regulations**: LODR, Insider Trading, ICDR\n"
            "- **Financial calculations**: P/E ratio, CAGR, margins\n"
            "- **Live market data**: Current prices, recent news\n\n"
            "Ask me anything about Indian finance!"
        ),
        'agent':      'direct',
        'confidence': 1.0,
        'sources':    [],
    })


# ============================================================
# SIDEBAR — Settings and DB stats
# ============================================================

with st.sidebar:
    st.title("🇮🇳 India Finance RAG")
    st.caption("Powered by Groq Llama3 + ChromaDB")

    st.divider()

    # Load retriever
    retriever, error = load_retriever()

    if error:
        st.error(f"⚠️ Database not ready: {error}")
        st.info("Run ingest.py first to build the database.")
        st.stop()
    else:
        st.success("✅ Database connected")

    # DB Statistics
    st.subheader("📊 Database Info")
    try:
        stats = retriever.get_db_stats()
        st.metric("Total Chunks", stats['total_chunks'])

        with st.expander("Companies covered"):
            for c in stats['companies']:
                if c != 'SEBI' and c != 'RBI' and c != 'N/A':
                    st.write(f"• {c}")

        with st.expander("Document types"):
            for dt in stats['doc_types']:
                st.write(f"• {dt}")
    except Exception:
        st.warning("Could not load DB stats")

    st.divider()

    # Search settings
    st.subheader("⚙️ Settings")

    filter_company = st.selectbox(
        "Filter by company",
        ["All companies", "Reliance Industries", "Tata Consultancy Services",
         "Infosys", "HDFC Bank", "ICICI Bank", "SEBI", "RBI"],
        help="Restrict search to one company's documents"
    )

    filter_type = st.selectbox(
        "Filter by document type",
        ["All types", "company_overview", "sebi_regulation",
         "annual_report_pdf", "rbi_policy"],
        help="Restrict search to one document type"
    )

    use_web_search = st.toggle(
        "Enable Web Search",
        value=True,
        help="Allow agent to fetch live news (uses Tavily API)"
    )

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Sample questions
    st.subheader("💡 Try asking:")
    sample_questions = [
        "What is TCS's revenue?",
        "Explain SEBI LODR regulations",
        "Calculate P/E if EPS is ₹80 and price is ₹3200",
        "What are HDFC Bank's key financial metrics?",
        "What does SEBI say about insider trading?",
        "Compare Infosys and TCS business models",
    ]

    for q in sample_questions:
        if st.button(q, use_container_width=True, key=f"sample_{q[:20]}"):
            # Inject the question as if the user typed it
            st.session_state['inject_query'] = q
            st.rerun()


# ============================================================
# MAIN CHAT AREA
# ============================================================

# Live price ticker strip at the top
st.subheader("📈 Live NSE Prices")
prices = get_live_prices()
if prices:
    cols = st.columns(len(prices))
    for col, (name, data) in zip(cols, prices.items()):
        change_color = "green" if data['change'] >= 0 else "red"
        arrow = "▲" if data['change'] >= 0 else "▼"
        col.metric(
            label=name,
            value=f"₹{data['price']:,.0f}",
            delta=f"{arrow} {abs(data['change']):.2f}%"
        )
else:
    st.info("Live prices unavailable (markets may be closed)")

st.divider()

# ── Render conversation history ────────────────────────────
st.subheader("💬 Chat")

for msg in st.session_state.messages:

    if msg['role'] == 'user':
        # User message — right aligned
        with st.chat_message("user", avatar="👤"):
            st.write(msg['content'])

    else:
        # Assistant message — left aligned
        with st.chat_message("assistant", avatar="🇮🇳"):

            # Answer text
            st.markdown(msg['content'])

            # Metadata row (confidence + agent badge)
            if msg.get('confidence') is not None and msg.get('agent'):
                confidence_html = format_confidence_badge(msg['confidence'])
                agent_html      = format_agent_badge(msg['agent'])
                st.markdown(
                    f"Confidence: {confidence_html} &nbsp; Agent: {agent_html}",
                    unsafe_allow_html=True
                )

            # Source citations (expandable)
            if msg.get('sources'):
                with st.expander(f"📄 View {len(msg['sources'])} sources"):
                    for i, src in enumerate(msg['sources']):
                        st.markdown(f"""
<div class="source-box">
<b>Source {i+1}</b> — {src.get('company','?')} 
| {src.get('filing_type','?')} 
| {src.get('filing_date','?')}
| Relevance: {src.get('score', 0):.2f}<br>
<small>{src.get('text','')[:250]}...</small>
</div>
""", unsafe_allow_html=True)


# ── Handle injected sample questions ──────────────────────
if 'inject_query' in st.session_state:
    user_query = st.session_state.pop('inject_query')
    # Process it (same as if user typed it)
    st.session_state.messages.append({
        'role':    'user',
        'content': user_query
    })
    # Will be processed below


# ── Chat input box ─────────────────────────────────────────
user_input = st.chat_input(
    "Ask about Indian finance, stocks, SEBI regulations...",
    key="chat_input"
)

# Use either typed input or injected question
query_to_process = user_input

# Process new query
if query_to_process:
    # Add user message to history
    st.session_state.messages.append({
        'role':    'user',
        'content': query_to_process
    })

    # Show thinking spinner while processing
    with st.spinner("🔍 Searching documents and generating answer..."):

        # Build filters from sidebar settings
        company_filter = (
            None if filter_company == "All companies"
            else filter_company
        )
        type_filter = (
            None if filter_type == "All types"
            else filter_type
        )

        # Override web search setting
        if not use_web_search:
            os.environ['TAVILY_API_KEY'] = ''

        # Process the query through our agent pipeline
        try:
            result = process_query(query_to_process, retriever)
            answer     = result.get('answer', 'No answer generated')
            sources    = result.get('sources', [])
            confidence = result.get('confidence', 0.0)
            agent_used = result.get('agent_used', 'rag')

        except Exception as e:
            answer     = f"⚠️ Error processing query: {e}\n\nPlease try again."
            sources    = []
            confidence = 0.0
            agent_used = 'rag'

    # Add assistant response to history
    st.session_state.messages.append({
        'role':       'assistant',
        'content':    answer,
        'sources':    sources,
        'confidence': confidence,
        'agent':      agent_used,
    })

    # Rerun to display the new messages
    st.rerun()
