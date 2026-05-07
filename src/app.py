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
    page_title="Vitta-Mitra: Your Financial Friend",
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
        color: #202124;  /* Explicit dark text for readability */
        border-left: 4px solid #1a73e8;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 4px 8px 8px 4px;
        font-size: 0.9em;
        line-height: 1.4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
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
        # detectar Colab vs local
        if 'google.colab' in sys.modules:
            chroma_path = "/content/drive/MyDrive/Finance_RAG/finrag_db"
        else:
            chroma_path = None  # Let get_retriever handle the default absolute path

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
            "Namaste! 🇮🇳 I am **Vitta-Mitra**, your intelligent financial companion.\n\n"
            "I'm here to help you navigate the complex world of Indian finance using **verified facts** from official sources.\n\n"
            "I can assist you with:\n"
            "- **Regulation Decoding**: SEBI (LODR/PIT) and RBI Master Directions.\n"
            "- **Tax Planning**: Slabs, TDS, and 80C deductions.\n"
            "- **Market Reality**: Live prices and performance for top Indian companies.\n"
            "- **Analyst Math**: Accurate financial ratios and growth calculations.\n\n"
            "What can we learn together today?"
        ),
        'agent':      'direct',
        'confidence': 1.0,
        'sources':    [],
    })


# ============================================================
# SIDEBAR — Settings and DB stats
# ============================================================

with st.sidebar:
    # ── Brand Identity ─────────────────────────────
    st.image("C:/Users/sriso/.gemini/antigravity/brain/5f7bc8af-69ec-4a9b-a1a6-8e279993a0a6/vitta_mitra_growth_logo_1777125272114.png")
    st.title("Vitta-Mitra 🇮🇳")
    st.markdown("""
    <div style="margin-top: -15px; margin-bottom: 20px;">
        <i>vitta (Finance) + mitra (Friend)</i><br>
        <b>Bridging the Literacy Gap.</b>
    </div>
    """, unsafe_allow_html=True)
    
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
         "Infosys", "HDFC Bank", "ICICI Bank", "SEBI", "RBI", "Income Tax"],
        help="Restrict search to one domain's documents"
    )

    filter_type = st.selectbox(
        "Filter by document type",
        ["All types", "company_overview", "sebi_regulation",
         "annual_report_pdf", "rbi_policy", "tax_regulation"],
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
        "What are the income tax slabs for FY 2024-25?",
        "Calculate P/E if EPS is ₹80 and price is ₹3200",
        "What are HDFC Bank's key financial metrics?",
        "What does SEBI say about insider trading?",
        "What are the 80C tax deduction limits?",
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
                        # Check if it's a web source or a local document
                        is_web = src.get('url') or src.get('url_link')
                        link_url = src.get('url') or src.get('url_link')
                        
                        source_title = f"Source {i+1}"
                        if is_web:
                            source_header = f'<a href="{link_url}" target="_blank">🌐 {source_title} (Click to visit)</a>'
                            source_meta = f"Web Search Result | {src.get('title','?')}"
                        else:
                            source_header = f"📄 {source_title}"
                            source_meta = f"{src.get('company','?')} | {src.get('filing_type','?')} | {src.get('filing_date','?')}"

                        st.markdown(f"""
<div class="source-box">
<b>{source_header}</b><br>
<small>{source_meta}</small><br>
| Relevance: {src.get('score', 0):.0%}<br>
<small>{src.get('text','')[:250]}...</small>
</div>
""", unsafe_allow_html=True)


# ── Handle injected sample questions ──────────────────────
if 'inject_query' in st.session_state:
    query_to_process = st.session_state.pop('inject_query')
else:
    # ── Chat input box ─────────────────────────────────────────
    query_to_process = st.chat_input(
        "Ask about Indian finance, stocks, SEBI regulations...",
        key="chat_input"
    )

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

        # Process the query through our agent pipeline
        try:
            result = process_query(
                query_to_process, 
                retriever, 
                use_web_search=use_web_search
            )
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
