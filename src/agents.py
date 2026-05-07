# src/agents.py
# ============================================================
# INDIA FINANCE RAG — LLM + AGENTS
# Covers: Week 5 (LLM integration) + Week 6 (3 agents)
# ============================================================
#
# WHAT THIS FILE CONTAINS:
#
#   WEEK 5 — LLM Integration:
#   ─────────────────────────
#   - Groq API connection (free Llama3)
#   - System prompt with financial analyst persona
#   - Structured output schema (answer + citations + confidence)
#   - Confidence-gated hedging (tells user when it's uncertain)
#   - Legal disclaimer injection for investment queries
#
#   WEEK 6 — Three Agents:
#   ──────────────────────
#   Agent 1 — Router Agent:
#     Reads the user query and decides which tool to use.
#     Routes to: RAG search / Calculator / Web Search / Direct LLM
#
#   Agent 2 — Calculator Agent:
#     Handles numerical questions (P/E ratio, CAGR, margins).
#     Extracts numbers from retrieved text, writes Python, executes it.
#
#   Agent 3 — Web Search Agent:
#     Fetches live financial news when corpus is stale or
#     query contains "today", "latest", "current price".
#     Uses Tavily API (free tier).
#
# HOW THEY WORK TOGETHER:
#   User question
#       ↓
#   Router Agent (classifies intent)
#       ↓           ↓           ↓
#   RAG Search  Calculator  Web Search
#       ↓           ↓           ↓
#       └─────── LLM generates answer ──────┘
#                   ↓
#   Structured response (answer + sources + confidence + disclaimer)
# ============================================================

import os
import re
import json
import math
import logging
import subprocess
import tempfile
import sys
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()

# Add project root to path for standalone execution
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import get_retriever, format_context_for_llm


# ============================================================
# CONFIGURATION
# ============================================================

# Groq model — free, fast, good quality
GROQ_MODEL = "llama-3.1-8b-instant"

# Temperature controls randomness
# 0 = deterministic (same question = same answer) — good for facts
# 1 = creative (varies each time) — not what we want for finance
LLM_TEMPERATURE = 0

# Confidence thresholds for hedging
# When retrieval similarity score is high → answer directly
# When medium → add "based on available data" caveat
# When low → tell user the answer isn't well-supported
HIGH_CONFIDENCE  = 0.75
LOW_CONFIDENCE   = 0.50

# Keywords that trigger the Web Search Agent
LIVE_DATA_KEYWORDS = [
    'today', 'current', 'latest', 'now', 'live',
    'price', 'share price', 'stock price', 'market cap today',
    'recent news', 'breaking', 'this week', 'this month'
]

# Keywords that trigger the Calculator Agent
CALC_KEYWORDS = [
    'calculate', 'compute', 'cagr', 'growth rate', 'ratio',
    'margin', 'percentage', 'how much', 'total', 'average',
    'p/e', 'pe ratio', 'eps', 'return on', 'debt to equity',
    'earnings per share', 'book value', 'yield'
]

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# WEEK 5 — LLM SETUP + CORE RAG CHAIN
# ============================================================

def get_llm() -> ChatGroq:
    """
    Creates and returns a Groq LLM connection.

    Groq runs Llama3 on their servers for free.
    No GPU needed on your end — the model runs remotely.
    Free tier: 30 requests/minute, 14,400/day.

    ChatGroq is LangChain's wrapper that makes Groq behave
    like any other LangChain LLM (easy to swap providers later).
    """
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Add it to .env or Colab Secrets."
        )

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=api_key,
        max_tokens=1024,      # Max tokens in the response
    )


# ── System Prompt ──────────────────────────────────────────
# This is the "identity" of our chatbot.
# It tells the LLM how to behave, what format to use,
# and what rules to follow. Crucial for consistent behaviour.

SYSTEM_PROMPT = """You are an expert Indian financial analyst assistant.
You answer questions about Indian companies listed on NSE/BSE, SEBI regulations, 
RBI policies, and Indian financial markets.

STRICT RULES YOU MUST FOLLOW:
1. ONLY answer based on the retrieved context provided to you.
2. ALWAYS cite your source using the format [Source N] where N is the source number.
3. If the context does not contain enough information and YOU HAVE NOT used web search, say:
   "I don't have sufficient information in my documents to answer this accurately."
   If you ARE using web data because documents were insufficient, start with:
   "I couldn't find specific details in my documents, but according to recent financial sources..."
4. NEVER make up financial figures, ratios, or regulatory rules.
5. ALWAYS add this disclaimer for investment-related questions:
   "⚠️ This is for informational purposes only and not financial advice. 
    Consult a SEBI-registered financial advisor before investing."
6. Use Indian financial terminology: crores/lakhs (not millions/billions), 
   ₹ symbol, NSE/BSE (not NYSE), SEBI (not SEC).
7. When discussing regulations, always mention the regulation name and year.

OUTPUT FORMAT:
Answer the question clearly and concisely.
End with: Sources: [list the source numbers you used]
"""


def normalize_confidence(score: float) -> float:
    """
    Normalizes raw AI logit scores into a human-readable 0-1 range.
    Uses a Sigmoid function: 1 / (1 + exp(-score/3))
    
    Why?
    - Raw scores from the reranker can range from -20 to +20.
    - -10 (bad)     → ~3.5%
    - 0   (neutral) → 50.0%
    - +5  (good)    → ~84.1%
    - +10 (perfect) → ~96.4%
    """
    try:
        # Standard Sigmoid with temperature scaling (T=3)
        return 1 / (1 + math.exp(-score / 3))
    except OverflowError:
        return 1.0 if score > 0 else 0.0


def build_rag_prompt(query: str, context: str, confidence: float) -> str:
    """
    WEEK 5 — Builds the complete prompt sent to the LLM.

    Combines:
    - The retrieved context (relevant chunks from ChromaDB)
    - The user's question
    - A confidence-based instruction

    CONFIDENCE-GATED HEDGING:
    High (>0.75): "Answer directly and confidently"
    Medium (0.5-0.75): "Note the information is from filed documents"
    Low (<0.5): "Explicitly state limited information available"

    This prevents the LLM from confidently giving wrong answers
    when we couldn't find good supporting documents.
    """
    # Choose hedging instruction based on confidence
    if confidence >= HIGH_CONFIDENCE:
        confidence_instruction = (
            "The retrieved context is highly relevant. "
            "Answer confidently and directly."
        )
    elif confidence >= LOW_CONFIDENCE:
        confidence_instruction = (
            "The retrieved context is moderately relevant. "
            "Answer but note: 'Based on available filing data as of [date].'"
        )
    else:
        confidence_instruction = (
            "The retrieved context has low relevance to this query. "
            "State clearly that you have limited information and "
            "recommend the user check official SEBI/BSE sources directly."
        )

    return f"""RETRIEVED CONTEXT:
{context}

---

USER QUESTION: {query}

INSTRUCTION: {confidence_instruction}

Please answer the question based strictly on the context above."""


def rag_answer(
    query: str,
    retriever,
    domain: str = "company",
    filter_company: Optional[str] = None,
    filter_type: Optional[str] = None
) -> dict:
    """
    WEEK 5 — Core RAG chain.

    Pipeline:
    1. Retrieve relevant chunks from ChromaDB
    2. Calculate average confidence score
    3. Build prompt with context + confidence instruction
    4. Send to Groq LLM
    5. Return structured response

    Returns dict with:
        answer     : LLM's response text
        sources    : List of source metadata dicts
        confidence : Average retrieval score (0-1)
        query      : Original question
    """
    logger.info(f"RAG answer for: {query[:60]} (Domain: {domain})")

    # Step 1: Retrieve relevant chunks
    results = retriever.search(
        query,
        collection_name=domain,
        n_results=5,
        filter_company=filter_company,
        filter_type=filter_type
    )

    if not results:
        return {
            'answer':     "I couldn't find relevant information in my documents. "
                         "Please check SEBI (sebi.gov.in) or BSE (bseindia.com) directly.",
            'sources':    [],
            'confidence': 0.0,
            'query':      query,
        }

    # Step 2: Calculate confidence
    raw_avg_score = sum(r['score'] for r in results) / len(results)
    avg_confidence = normalize_confidence(raw_avg_score)

    # Step 3: Format context for the LLM
    context = format_context_for_llm(results)

    # Step 4: Build prompt
    prompt = build_rag_prompt(query, context, avg_confidence)

    # Step 5: Call Groq LLM
    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        answer_text = response.content

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        answer_text = f"LLM error: {e}. Please try again."

    # Check if we need a financial disclaimer
    investment_keywords = ['invest', 'buy', 'sell', 'portfolio', 'should i', 'recommend']
    needs_disclaimer = any(kw in query.lower() for kw in investment_keywords)

    if needs_disclaimer and '⚠️' not in answer_text:
        answer_text += (
            "\n\n⚠️ This is for informational purposes only and not financial advice. "
            "Consult a SEBI-registered financial advisor before making investment decisions."
        )

    # Normalize individual source scores for the UI
    for r in results:
        r['score'] = normalize_confidence(r['score'])

    return {
        'answer':     answer_text,
        'sources':    results,
        'confidence': round(avg_confidence, 3),
        'query':      query,
    }


# ============================================================
# WEEK 6 — AGENT 1: ROUTER AGENT
# ============================================================

def router_agent(query: str) -> dict:
    """
    WEEK 6 — Agent 1: Router (UPGRADED)
    ─────────────────────────
    Classifies the user's query into:
    1. Route  (rag, calculator, websearch, direct)
    2. Domain (company, sebi, rbi)

    Returns: dict with 'route' and 'domain'
    """
    query_lower = query.lower().strip()
    logger.info(f"Router: classifying '{query_lower[:50]}'")

    # ── Fast path for greetings ───────────────────────────
    direct_patterns = ['hello', 'hi', 'what can you do', 'help', 'who are you']
    if any(p in query_lower for p in direct_patterns):
        return {"route": "direct", "domain": "company"}

    # ── LLM Classification ───────────────────────────────
    router_prompt = f"""Target Categories:
- route: [rag, calculator, websearch, direct]
- domain: [company, sebi, rbi, tax]

Definitions:
    - 'domain': The financial area. MUST be one of:
        * 'company' (Stock prices, company news, revenue, profits)
        * 'sebi' (Market regulations, insider trading, listing rules, mutual fund laws)
        * 'rbi' (Banking rules, interest rates, repo rates, KYC, digital payments)
        * 'tax' (Income tax slabs, TDS, 80C deductions, capital gains tax, tax filing)
    - 'query': The search query...

Instructions:
1. If the query asks for calculations (P/E, growth, total), route is 'calculator'.
2. If it asks for live/latest news or current price, route is 'websearch'.
3. Default route for financial questions is 'rag'.

USER QUERY: {query}

RESPONSE FORMAT:
{{"route": "...", "domain": "..."}}
"""
    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content="You are a routing agent. Respond ONLY with JSON."),
            HumanMessage(content=router_prompt)
        ])
        # Clean the response in case LLM adds markdown
        clean_json = re.sub(r'```json\s*|\s*```', '', response.content).strip()
        classification = json.loads(clean_json)
        logger.info(f"Router → {classification}")
        return classification
    except Exception as e:
        logger.warning(f"LLM routing failed: {e}. Falling back to default.")
        return {"route": "rag", "domain": "company"}


# ============================================================
# WEEK 6 — AGENT 2: CALCULATOR AGENT
# ============================================================

def calculator_agent(query: str, retriever, domain: str = "company") -> dict:
    """
    WEEK 6 — Agent 2: Calculator
    """
    logger.info(f"Calculator agent: {query[:60]} (Domain: {domain})")

    # Step 1: Get relevant context
    results = retriever.search(query, collection_name=domain, n_results=5)
    context = format_context_for_llm(results)
    avg_confidence = sum(r['score'] for r in results) / len(results) if results else 0

    # Step 2: Ask LLM to extract numbers and write calculation code
    calc_prompt = f"""You are a financial calculator assistant for Indian markets.

RETRIEVED FINANCIAL DATA:
{context}

USER QUESTION: {query}

TASK:
1. Extract the relevant numbers from the context above.
2. Write Python code to calculate the answer.
3. Use Indian units: crores (1 crore = 10,000,000) where appropriate.
4. Your response must be in this EXACT format:

NUMBERS_EXTRACTED:
[list each number you found and what it represents]

PYTHON_CODE:
```python
# Your calculation here
# Use variables with clear names
# Print the final answer with ₹ or % as appropriate
result = ...
print(f"Answer: {{result}}")
```

EXPLANATION:
[Brief explanation of what you calculated and why]

If you cannot find the required numbers in the context, write:
NUMBERS_EXTRACTED: Not found in documents
PYTHON_CODE: None
EXPLANATION: The required numerical data is not in the retrieved documents."""

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content="You are a precise financial calculator. Follow the format exactly."),
            HumanMessage(content=calc_prompt)
        ])
        llm_response = response.content

    except Exception as e:
        logger.error(f"Calculator LLM failed: {e}")
        return {
            'answer':     f"Calculator error: {e}",
            'code':       None,
            'result':     None,
            'sources':    results,
            'confidence': avg_confidence,
            'query':      query,
        }

    # Step 3: Extract and execute the Python code
    code_result = None
    extracted_code = None

    # Find Python code block in the response (handle py/python and variations)
    code_match = re.search(r'```(?:python|py)?\s*\n(.*?)\s*```', llm_response, re.DOTALL)

    if code_match:
        extracted_code = code_match.group(1).strip()

        if extracted_code and extracted_code != 'None':
            code_result = _execute_python_safely(extracted_code)

    # Step 4: Build final answer
    if code_result and code_result.get('success'):
        answer = (
            f"{llm_response}\n\n"
            f"**Calculated Result:** {code_result['output']}"
        )
    elif code_result and not code_result.get('success'):
        answer = (
            f"{llm_response}\n\n"
            f"⚠️ Calculation error: {code_result.get('error', 'Unknown error')}"
        )
    else:
        answer = llm_response

    return {
        'answer':     answer,
        'code':       extracted_code,
        'result':     code_result,
        'sources':    results,
        'confidence': round(avg_confidence, 3),
        'query':      query,
    }


def _execute_python_safely(code: str) -> dict:
    """
    Executes Python code in a sandboxed subprocess.

    WHY A SUBPROCESS?
    If the LLM writes buggy code (infinite loop, division by zero),
    running it in our main process would crash the app.
    A subprocess is isolated — if it crashes, we just get an error message.

    timeout=10: Any calculation should complete in under 10 seconds.
    If it doesn't, something is wrong (probably an infinite loop).

    Returns: {'success': bool, 'output': str, 'error': str}
    """
    try:
        # Write code to a temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(code)
            temp_path = f.name

        # Run it in a subprocess with timeout
        # sys.executable ensures we use this same Python interpreter
        proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=10          # Kill after 10 seconds
        )

        # Cleanup temp file
        os.unlink(temp_path)

        if proc.returncode == 0:
            return {
                'success': True,
                'output':  proc.stdout.strip(),
                'error':   None,
            }
        else:
            return {
                'success': False,
                'output':  None,
                'error':   proc.stderr.strip()[:200],   # Limit error length
            }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output':  None,
            'error':   "Calculation timed out (>10 seconds)",
        }
    except Exception as e:
        return {
            'success': False,
            'output':  None,
            'error':   str(e),
        }


# ============================================================
# WEEK 6 — AGENT 3: WEB SEARCH AGENT
# ============================================================

def web_search_agent(query: str, retriever, domain: str = "company", is_fallback: bool = False) -> dict:
    """
    WEEK 6 — Agent 3: Web Search
    ──────────────────────────────
    Fetches live financial news and data when:
    - Query asks for current/live data ("today's price", "latest news")
    - Retrieval confidence is low (our corpus doesn't cover this)
    - Query contains time-sensitive keywords

    Uses Tavily API (free tier: 1000 searches/month).
    Tavily is designed for LLMs — returns clean text, not HTML.

    Falls back to RAG-only if Tavily key is missing.
    """
    logger.info(f"Web search agent: {query[:60]} (Domain: {domain})")

    tavily_key = os.environ.get('TAVILY_API_KEY')

    web_context = ""
    web_sources = []

    # ── Try Tavily web search ──────────────────────────────
    if tavily_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)

            # Add "India" context to queries for relevance
            search_query = f"{query} India NSE BSE stock market"

            search_results = client.search(
                query=search_query,
                max_results=3,
                search_depth="basic"    # "advanced" uses more credits
            )

            # Extract text from Tavily results
            web_parts = []
            for i, result in enumerate(search_results.get('results', [])):
                title   = result.get('title', 'No title')
                content = result.get('content', '')
                url     = result.get('url', '')

                web_parts.append(
                    f"[Web Source {i+1}: {title}]\n{content}"
                )
                web_sources.append({
                    'title': title,
                    'url':   url,
                    'text':  content,
                    'type':  'web_search'
                })

            web_context = "\n\n".join(web_parts)
            logger.info(f"Tavily returned {len(web_sources)} results")

        except ImportError:
            logger.warning("tavily-python not installed. Run: pip install tavily-python")
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")

    # ── Also get RAG context ───────────────────────────────
    rag_results = retriever.search(query, collection_name=domain, n_results=3)
    rag_context = format_context_for_llm(rag_results)
    
    # Normalize confidence
    raw_avg_score = sum(r['score'] for r in rag_results) / len(rag_results) if rag_results else 0
    avg_confidence = normalize_confidence(raw_avg_score)

    # Normalize individual scores for the UI
    for r in rag_results:
        r['score'] = normalize_confidence(r['score'])

    # ── Combine web + RAG context ─────────────────────────
    if web_context:
        combined_context = (
            f"LIVE WEB DATA (fetched now):\n{web_context}"
            f"\n\n---\n\nDOCUMENT DATABASE:\n{rag_context}"
        )
    else:
        combined_context = f"DOCUMENT DATABASE:\n{rag_context}"
        logger.info("No web data — using RAG only")

    # Determine the correct instruction based on fallback state
    fallback_prefix = (
        "If using web data as a fallback because documents were insufficient, "
        "start exactly with: \"I couldn't find specific details in my documents, "
        "but according to recent financial sources...\""
    ) if is_fallback else 'If using web data, note it as "According to recent sources..."'

    # ── Generate answer ────────────────────────────────────
    prompt = f"""You are an Indian financial analyst with access to both 
live web data and a document database.

{combined_context}

USER QUESTION: {query}

Answer using the most recent information available. 
{fallback_prefix}
If using document data, cite [Source N].
Always add date context so user knows how recent the info is."""

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        answer_text = response.content
    except Exception as e:
        answer_text = f"LLM error: {e}"

    return {
        'answer':      answer_text,
        'sources':     rag_results,
        'web_sources': web_sources,
        'confidence':  round(avg_confidence, 3),
        'query':       query,
        'used_web':    bool(web_context),
    }


# ============================================================
# MAIN ORCHESTRATOR — Combines all agents
# ============================================================

def process_query(query: str, retriever=None, use_web_search: bool = True) -> dict:
    """
    Main entry point called by app.py.

    1. Router decides which agent to use
    2. Correct agent processes the query
    3. Returns standardised response dict

    Response dict always has:
        answer      : The text answer
        sources     : List of source metadata
        confidence  : Retrieval confidence score
        query       : Original question
        agent_used  : Which agent handled it
    """
    if retriever is None:
        retriever = get_retriever()

    # Step 1: Route the query
    classification = router_agent(query)
    route  = classification.get('route', 'rag')
    domain = classification.get('domain', 'company')
    logger.info(f"Query routed to: {route} | Domain: {domain}")

    # Step 2: Process with the appropriate agent
    if route == "calculator":
        result = calculator_agent(query, retriever, domain=domain)

    elif route == "websearch" and use_web_search:
        result = web_search_agent(query, retriever, domain=domain)

    elif route == "websearch" and not use_web_search:
        # Fallback to RAG if web search is disabled but routed
        logger.info("Web search disabled — falling back to RAG")
        result = rag_answer(query, retriever, domain=domain)
        result['agent_used'] = 'rag'
        return result

    elif route == "direct":
        # Simple direct LLM response — no retrieval
        try:
            llm = get_llm()
            response = llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=query)
            ])
            result = {
                'answer':     response.content,
                'sources':    [],
                'confidence': 1.0,
                'query':      query,
            }
        except Exception as e:
            result = {
                'answer':     f"Error: {e}",
                'sources':    [],
                'confidence': 0.0,
                'query':      query,
            }

    else:
        # Default: RAG
        result = rag_answer(query, retriever, domain=domain)
        
        # ── Fallback to Web Search ──────────────────────────
        # If RAG confidence is low and web search is enabled
        if result.get('confidence', 0) < 0.35 and use_web_search:
            logger.info(f"RAG confidence low ({result['confidence']}) — falling back to Web Search")
            web_result = web_search_agent(query, retriever, domain=domain, is_fallback=True)
            
            # Use web result but keep the original query info
            result = web_result
            result['agent_used'] = 'rag_fallback_web'
            return result

    result['agent_used'] = route
    return result


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    retriever = get_retriever()

    test_queries = [
        "What is TCS's business model?",
        "What are the penalties for insider trading in India?",
        "Calculate the P/E ratio if EPS is ₹50 and share price is ₹3500",
        "What is the latest news about Reliance Industries?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = process_query(q, retriever)
        print(f"Agent: {result['agent_used']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Answer: {result['answer'][:300]}...")
