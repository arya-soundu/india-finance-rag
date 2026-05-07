# src/evaluate.py
# ============================================================
# INDIA FINANCE RAG — RAGAS EVALUATION PIPELINE
# Covers: Week 8
# ============================================================
#
# WHAT THIS FILE DOES:
#   Evaluates how good our RAG system is using RAGAS metrics.
#
# WHAT IS RAGAS?
#   RAGAS (Retrieval Augmented Generation Assessment) is a
#   framework that automatically scores RAG systems on:
#
#   1. FAITHFULNESS (0-1):
#      Does the answer only use information from the retrieved chunks?
#      Score = 1.0 → nothing was made up
#      Score = 0.5 → half the answer was hallucinated
#
#   2. ANSWER RELEVANCY (0-1):
#      Does the answer actually address the question?
#      Score = 1.0 → perfectly on-topic
#      Score = 0.0 → completely off-topic
#
#   3. CONTEXT RECALL (0-1):
#      Did we retrieve all the relevant information that exists?
#      Score = 1.0 → nothing important was missed
#      Score = 0.5 → we only found half the relevant info
#
#   4. CONTEXT PRECISION (0-1):
#      Was the retrieved context actually relevant?
#      Score = 1.0 → every chunk was useful
#      Score = 0.5 → half the chunks were irrelevant noise
#
# HOW IT WORKS:
#   We have a test set of 30 questions with known good answers.
#   We run each question through our RAG system.
#   RAGAS compares the RAG output to the known good answer.
#   We get scores for each metric, per question and overall.
#
# WHY THIS MATTERS FOR YOUR PROJECT:
#   "The system works" is not evidence.
#   "Faithfulness: 0.89, Answer Relevancy: 0.82" IS evidence.
#   Evaluators and interviewers will ask you to prove quality.
# ============================================================

import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import rag_answer, get_llm
from src.retriever import get_retriever, format_context_for_llm


# ============================================================
# CONFIGURATION
# ============================================================

# Output paths
EVAL_OUTPUT_DIR  = "./logs"
EVAL_RESULTS_FILE = "ragas_evaluation_results.json"
EVAL_REPORT_FILE  = "ragas_evaluation_report.csv"

# ── Logging ────────────────────────────────────────────────
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{EVAL_OUTPUT_DIR}/evaluate.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# TEST DATASET
# 30 questions covering all document types in our DB
# ground_truth = what a correct answer should contain
# ============================================================

INDIA_FINANCE_TEST_SET = [
    # ── Company questions ──────────────────────────────────
    {
        "question": "What sector does Reliance Industries operate in?",
        "ground_truth": "Reliance Industries operates in the energy, petrochemicals, natural gas, retail, telecommunications, and textile sectors.",
        "category": "company_overview",
    },
    {
        "question": "What is TCS known for as a company?",
        "ground_truth": "TCS (Tata Consultancy Services) is one of India's largest IT services and consulting companies, providing software development, digital transformation, and business process outsourcing services.",
        "category": "company_overview",
    },
    {
        "question": "What does Infosys do?",
        "ground_truth": "Infosys is a global leader in next-generation digital services and consulting, helping clients in over 50 countries navigate their digital transformation.",
        "category": "company_overview",
    },
    {
        "question": "What type of company is HDFC Bank?",
        "ground_truth": "HDFC Bank is one of India's largest private sector banks, offering banking products and financial services.",
        "category": "company_overview",
    },
    {
        "question": "What is ICICI Bank's primary business?",
        "ground_truth": "ICICI Bank is a leading private sector bank in India offering a wide range of banking products and financial services to corporate and retail customers.",
        "category": "company_overview",
    },

    # ── SEBI regulation questions ──────────────────────────
    {
        "question": "What is the full form of SEBI LODR?",
        "ground_truth": "SEBI LODR stands for Securities and Exchange Board of India (Listing Obligations and Disclosure Requirements) Regulations, 2015.",
        "category": "sebi_regulation",
    },
    {
        "question": "What are a listed company's disclosure requirements under SEBI?",
        "ground_truth": "Under SEBI LODR Regulations 2015, listed companies must make timely disclosures of material information, financial results, corporate governance reports, and shareholding patterns to the stock exchanges.",
        "category": "sebi_regulation",
    },
    {
        "question": "What is insider trading according to SEBI?",
        "ground_truth": "According to SEBI Prohibition of Insider Trading Regulations 2015, insider trading involves trading in securities of a company by persons who have access to unpublished price-sensitive information (UPSI).",
        "category": "sebi_regulation",
    },
    {
        "question": "Who is considered an insider under SEBI regulations?",
        "ground_truth": "Under SEBI Insider Trading Regulations, an insider is any person who is connected with the company or is in possession of unpublished price-sensitive information (UPSI) about the company.",
        "category": "sebi_regulation",
    },
    {
        "question": "What is unpublished price-sensitive information (UPSI)?",
        "ground_truth": "UPSI is any information relating to a company or its securities that is not generally available and which upon becoming generally available is likely to materially affect the price of the securities.",
        "category": "sebi_regulation",
    },

    # ── Financial metrics questions ─────────────────────────
    {
        "question": "What is a P/E ratio in stock market investing?",
        "ground_truth": "The Price-to-Earnings (P/E) ratio is a valuation metric calculated by dividing the current stock price by the earnings per share (EPS). It indicates how much investors are willing to pay per rupee of earnings.",
        "category": "financial_concept",
    },
    {
        "question": "What does EPS mean in financial terms?",
        "ground_truth": "EPS (Earnings Per Share) is a company's profit divided by the number of outstanding shares. It indicates how much profit each share generates.",
        "category": "financial_concept",
    },
    {
        "question": "What is market capitalisation?",
        "ground_truth": "Market capitalisation is the total market value of a company's outstanding shares, calculated by multiplying the current share price by the total number of outstanding shares.",
        "category": "financial_concept",
    },
    {
        "question": "What is dividend yield?",
        "ground_truth": "Dividend yield is the annual dividend payment divided by the current stock price, expressed as a percentage. It measures how much a company pays out in dividends relative to its share price.",
        "category": "financial_concept",
    },
    {
        "question": "What is the debt to equity ratio?",
        "ground_truth": "The debt to equity ratio measures a company's financial leverage by dividing total liabilities by shareholder equity. A higher ratio indicates more debt relative to equity.",
        "category": "financial_concept",
    },

    # ── NSE/BSE market questions ────────────────────────────
    {
        "question": "What is NSE in India?",
        "ground_truth": "NSE (National Stock Exchange) is one of India's leading stock exchanges, known for its NIFTY 50 index. It uses electronic trading and is one of the largest exchanges in the world by equity trading volume.",
        "category": "market_structure",
    },
    {
        "question": "What is the difference between NSE and BSE?",
        "ground_truth": "NSE (National Stock Exchange) uses the NIFTY index while BSE (Bombay Stock Exchange) uses the SENSEX index. BSE is the older exchange while NSE has higher trading volumes for equity derivatives.",
        "category": "market_structure",
    },
    {
        "question": "What is NIFTY 50?",
        "ground_truth": "NIFTY 50 is a stock market index representing 50 of the largest and most liquid Indian companies listed on the National Stock Exchange (NSE). It is used as a benchmark for the Indian equity market.",
        "category": "market_structure",
    },

    # ── RBI / Macro questions ──────────────────────────────
    {
        "question": "What is the repo rate set by RBI?",
        "ground_truth": "The repo rate is the rate at which the Reserve Bank of India (RBI) lends money to commercial banks. Changes in the repo rate affect borrowing costs across the economy and influence inflation.",
        "category": "rbi_policy",
    },
    {
        "question": "What does RBI stand for and what is its role?",
        "ground_truth": "RBI stands for Reserve Bank of India. It is India's central bank responsible for monetary policy, regulating the banking system, managing foreign exchange, and issuing currency.",
        "category": "rbi_policy",
    },

    # ── Unanswerable questions (tests abstention) ──────────
    {
        "question": "What will be Reliance's stock price next month?",
        "ground_truth": "This question cannot be answered as future stock prices cannot be predicted.",
        "category": "unanswerable",
    },
    {
        "question": "What is the secret investment strategy of TCS insiders?",
        "ground_truth": "This question cannot be answered as it asks for non-public insider information.",
        "category": "unanswerable",
    },
    {
        "question": "What is the personal net worth of the Ambani family?",
        "ground_truth": "Personal net worth figures for the Ambani family are not available in company filings or SEBI regulatory documents.",
        "category": "unanswerable",
    },

    # ── Cross-document questions ───────────────────────────
    {
        "question": "How do SEBI regulations protect retail investors in India?",
        "ground_truth": "SEBI protects retail investors through mandatory disclosures by listed companies, prohibition of insider trading, grievance redressal mechanisms, and regulations like LODR that ensure transparency.",
        "category": "sebi_regulation",
    },
    {
        "question": "What compliance requirements do NSE-listed companies have?",
        "ground_truth": "NSE-listed companies must comply with SEBI LODR Regulations including quarterly financial results, corporate governance reports, shareholder meeting disclosures, and material event notifications.",
        "category": "sebi_regulation",
    },
    {
        "question": "What is return on equity and why does it matter?",
        "ground_truth": "Return on Equity (ROE) measures a company's profitability relative to shareholder equity. A higher ROE indicates more efficient use of shareholder capital to generate profits.",
        "category": "financial_concept",
    },
    {
        "question": "What is the significance of beta in stock analysis?",
        "ground_truth": "Beta measures a stock's volatility relative to the market. Beta > 1 means more volatile than the market, beta < 1 means less volatile. It is used in risk assessment and portfolio construction.",
        "category": "financial_concept",
    },
    {
        "question": "What are IT sector stocks in India?",
        "ground_truth": "Major Indian IT sector stocks include TCS, Infosys, Wipro, HCL Technologies, and Tech Mahindra, all listed on NSE and BSE. They are known for IT services exports.",
        "category": "company_overview",
    },
    {
        "question": "What does profit margin indicate about a company?",
        "ground_truth": "Profit margin indicates how much profit a company makes for every rupee of revenue. Higher margins indicate better cost efficiency and pricing power.",
        "category": "financial_concept",
    },
    {
        "question": "What is a rights issue in Indian capital markets?",
        "ground_truth": "A rights issue is when a company offers existing shareholders the right to buy additional shares at a discounted price. It is regulated by SEBI ICDR Regulations 2018.",
        "category": "sebi_regulation",
    },
]


# ============================================================
# MANUAL RAGAS-STYLE EVALUATION
# We implement the core metrics manually because RAGAS
# sometimes requires an OpenAI key. This is fully free.
# ============================================================

def evaluate_faithfulness(answer: str, context_chunks: list) -> float:
    """
    FAITHFULNESS METRIC:
    Measures whether the answer is grounded in the retrieved context.

    Approach:
    - Extract key factual claims from the answer
    - Check whether each claim appears in the context
    - Score = claims_supported / total_claims

    This is a simplified version — proper RAGAS uses an LLM
    to check each claim. Our version uses string matching
    which is faster and costs no API calls.
    """
    if not answer or not context_chunks:
        return 0.0

    # Combine all context text
    context_text = " ".join(c.get('text', '') for c in context_chunks).lower()
    answer_lower = answer.lower()

    # Extract sentences from answer (basic sentence tokenizer)
    # Split on period, question mark, exclamation mark
    sentences = re.split(r'[.!?]', answer_lower)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return 0.5  # Can't evaluate very short answers

    # For each sentence, check if key content words appear in context
    supported = 0
    for sentence in sentences:
        # Get meaningful words (skip common words)
        stop_words = {'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and',
                      'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                      'with', 'by', 'from', 'this', 'that', 'these', 'those',
                      'it', 'its', 'as', 'be', 'has', 'have', 'had', 'will'}
        words = [w for w in sentence.split() if w not in stop_words and len(w) > 3]

        if not words:
            supported += 1  # Skip trivial sentences
            continue

        # Check if majority of key words appear in context
        found = sum(1 for w in words if w in context_text)
        if found / len(words) >= 0.4:   # 40% of key words found
            supported += 1

    return round(supported / len(sentences), 3)


def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """
    ANSWER RELEVANCY METRIC:
    Measures whether the answer addresses the question.

    Approach:
    - Extract key question words
    - Check if the answer addresses those topics
    - Penalise if answer says "I don't know" (when it shouldn't)
    """
    if not question or not answer:
        return 0.0

    question_lower = question.lower()
    answer_lower   = answer.lower()

    # Penalise "I don't know" style answers for factual questions
    unsupported_phrases = [
        "i don't have", "not available", "cannot find",
        "no information", "unclear", "i couldn't find"
    ]
    if any(p in answer_lower for p in unsupported_phrases):
        return 0.3   # Partial credit — at least it's honest

    # Extract question keywords
    stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'does', 'do',
                  'how', 'why', 'when', 'where', 'who', 'which', 'can'}
    question_words = [
        w.strip('?.,') for w in question_lower.split()
        if w not in stop_words and len(w) > 3
    ]

    if not question_words:
        return 0.5

    # Check how many question topics appear in the answer
    found = sum(1 for w in question_words if w in answer_lower)
    relevancy = found / len(question_words)

    return round(min(relevancy * 1.2, 1.0), 3)   # Slight boost, cap at 1.0


def evaluate_context_recall(ground_truth: str, context_chunks: list) -> float:
    """
    CONTEXT RECALL METRIC:
    Measures whether the retrieved context contains the information
    needed to answer the question (compared to ground truth).

    Score = how much of the ground truth is supported by context
    """
    if not ground_truth or not context_chunks:
        return 0.0

    context_text  = " ".join(c.get('text', '') for c in context_chunks).lower()
    truth_lower   = ground_truth.lower()

    # Extract key phrases from ground truth
    sentences = re.split(r'[.!?]', truth_lower)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        return 0.5

    recalled = 0
    for sentence in sentences:
        words = [w for w in sentence.split() if len(w) > 4]
        if not words:
            continue
        found = sum(1 for w in words if w in context_text)
        if found / len(words) >= 0.35:
            recalled += 1

    return round(recalled / len(sentences), 3)


def evaluate_context_precision(question: str, context_chunks: list) -> float:
    """
    CONTEXT PRECISION METRIC:
    Measures what fraction of retrieved chunks were actually relevant
    to the question.

    A high-precision retrieval means we didn't fetch irrelevant noise.
    """
    if not question or not context_chunks:
        return 0.0

    question_lower = question.lower()
    question_words = [w for w in question_lower.split() if len(w) > 3]

    if not question_words:
        return 0.5

    relevant_chunks = 0
    for chunk in context_chunks:
        chunk_text = chunk.get('text', '').lower()
        found = sum(1 for w in question_words if w in chunk_text)
        if found / len(question_words) >= 0.25:   # 25% word overlap
            relevant_chunks += 1

    return round(relevant_chunks / len(context_chunks), 3)


import re   # Needed for sentence splitting above


# ============================================================
# MAIN EVALUATION RUNNER
# ============================================================

def run_evaluation(
    test_set: list = None,
    retriever=None,
    output_dir: str = None
) -> dict:
    """
    Runs the full evaluation pipeline.

    For each question in the test set:
    1. Run it through our RAG system
    2. Calculate all 4 RAGAS-style metrics
    3. Log results per question
    4. Calculate overall averages
    5. Save results to JSON and CSV

    Returns: dict with all scores and per-question breakdown
    """
    if test_set is None:
        test_set = INDIA_FINANCE_TEST_SET

    if retriever is None:
        retriever = get_retriever()

    if output_dir is None:
        try:
            import google.colab
            output_dir = "/content/drive/MyDrive/Finance_RAG/logs"
        except ImportError:
            output_dir = EVAL_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("INDIA FINANCE RAG — RAGAS EVALUATION")
    logger.info(f"Test questions: {len(test_set)}")
    logger.info("=" * 60)

    results = []
    total = len(test_set)

    for i, test_item in enumerate(test_set):
        question     = test_item['question']
        ground_truth = test_item['ground_truth']
        category     = test_item.get('category', 'general')

        logger.info(f"\n[{i+1}/{total}] {question[:60]}...")

        # Run through RAG system with category mapping
        domain = category.split('_')[0] if '_' in category else 'company'
        if domain not in ['company', 'sebi', 'rbi', 'tax']:
            domain = 'company'

        try:
            rag_result = rag_answer(question, retriever, domain=domain)
            answer     = rag_result.get('answer', '')
            sources    = rag_result.get('sources', [])
            confidence = float(rag_result.get('confidence', 0.0))
        except Exception as e:
            logger.error(f"RAG failed for Q{i+1}: {e}")
            answer     = f"Error: {e}"
            sources    = []
            confidence = 0.0

        # Calculate metrics
        faithfulness       = evaluate_faithfulness(answer, sources)
        answer_relevancy   = evaluate_answer_relevancy(question, answer)
        context_recall     = evaluate_context_recall(ground_truth, sources)
        context_precision  = evaluate_context_precision(question, sources)

        # Overall score for this question (simple average)
        overall = float(round(
            (faithfulness + answer_relevancy + context_recall + context_precision) / 4,
            3
        ))

        q_result = {
            'question':          question,
            'category':          category,
            'ground_truth':      ground_truth,
            'answer':            answer[:500],   # Truncate for storage
            'faithfulness':      faithfulness,
            'answer_relevancy':  answer_relevancy,
            'context_recall':    context_recall,
            'context_precision': context_precision,
            'overall_score':     overall,
            'retrieval_confidence': confidence,
            'num_sources':       len(sources),
        }

        results.append(q_result)

        logger.info(
            f"  Faithfulness: {faithfulness:.2f} | "
            f"Relevancy: {answer_relevancy:.2f} | "
            f"Recall: {context_recall:.2f} | "
            f"Precision: {context_precision:.2f} | "
            f"Overall: {overall:.2f}"
        )

    # ── Calculate aggregate metrics ────────────────────────
    def avg(key):
        vals = [r[key] for r in results]
        return round(sum(vals) / len(vals), 3)

    summary = {
        'evaluation_date':   datetime.now().isoformat(),
        'total_questions':   total,
        'metrics': {
            'avg_faithfulness':      avg('faithfulness'),
            'avg_answer_relevancy':  avg('answer_relevancy'),
            'avg_context_recall':    avg('context_recall'),
            'avg_context_precision': avg('context_precision'),
            'avg_overall':           avg('overall_score'),
        },
        'per_category': {},
        'per_question': results,
    }

    # ── Per-category breakdown ─────────────────────────────
    categories = list({r['category'] for r in results})
    for cat in categories:
        cat_results = [r for r in results if r['category'] == cat]
        summary['per_category'][cat] = {
            'count':            len(cat_results),
            'avg_faithfulness': round(sum(r['faithfulness'] for r in cat_results) / len(cat_results), 3),
            'avg_overall':      round(sum(r['overall_score'] for r in cat_results) / len(cat_results), 3),
        }

    # ── Save results ───────────────────────────────────────
    # JSON — full detail
    json_path = os.path.join(output_dir, EVAL_RESULTS_FILE)
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✅ Full results saved: {json_path}")

    # CSV — easy to open in Excel
    csv_path = os.path.join(output_dir, EVAL_REPORT_FILE)
    df = pd.DataFrame([{
        'Question':          r['question'],
        'Category':          r['category'],
        'Faithfulness':      r['faithfulness'],
        'Answer_Relevancy':  r['answer_relevancy'],
        'Context_Recall':    r['context_recall'],
        'Context_Precision': r['context_precision'],
        'Overall_Score':     r['overall_score'],
        'Confidence':        r['retrieval_confidence'],
    } for r in results])
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ CSV report saved: {csv_path}")

    # ── Print summary ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for metric, value in summary['metrics'].items():
        bar_len = int(value * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        logger.info(f"{metric:35s} {bar} {value:.3f}")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")

    return summary


# ============================================================
# QUICK TEST — single question evaluation
# ============================================================

def quick_eval(question: str, retriever=None) -> None:
    """
    Evaluates a single question and prints all metrics.
    Useful for testing individual queries.
    """
    if retriever is None:
        retriever = get_retriever()

    print(f"\nQuestion: {question}")
    print("-" * 60)

    result = rag_answer(question, retriever)
    answer  = result.get('answer', '')
    sources = result.get('sources', [])

    print(f"Answer: {answer[:300]}...")
    print(f"\nSources found: {len(sources)}")

    if sources:
        print("\nMetrics:")
        faith = evaluate_faithfulness(answer, sources)
        relev = evaluate_answer_relevancy(question, answer)
        print(f"  Faithfulness:     {faith:.3f}")
        print(f"  Answer Relevancy: {relev:.3f}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("Running India Finance RAG evaluation...")
    print("This will test 30 questions — takes ~10 minutes\n")

    summary = run_evaluation()

    print("\n🎉 Evaluation complete!")
    print(f"Overall score: {summary['metrics']['avg_overall']:.3f}")
    print(f"Faithfulness:  {summary['metrics']['avg_faithfulness']:.3f}")
    print(f"Relevancy:     {summary['metrics']['avg_answer_relevancy']:.3f}")
