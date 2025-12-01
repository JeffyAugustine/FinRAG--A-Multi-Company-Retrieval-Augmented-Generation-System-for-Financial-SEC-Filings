# 🏢 FinRAG--A-Multi-Company-Retrieval-Augmented-Generation-System-for-Financial-SEC-Filings

A Retrieval-Augmented Generation (RAG) platform for analyzing and
comparing SEC 10-K filings across Apple, Microsoft, and Tesla using
advanced semantic search and AI-driven insights.


## 🎯 Overview

This project implements a sophisticated RAG (Retrieval-Augmented
Generation) system that enables financial analysts, researchers, and
investors to:

-   Analyze individual SEC 10-K filings with semantic precision\
-   Compare financial disclosures across multiple companies\
-   Extract key insights using AI-powered natural language queries\
-   Filter results by year, section, and company for targeted analysis

Supported Companies: **Apple (AAPL), Microsoft (MSFT), Tesla (TSLA)**\
Years: **2023--2024 SEC 10-K filings**

## ✨ Key Features

### 🔍 Three Search Modes

-   **Semantic Search** -- AI-powered conceptual understanding\
-   **Hybrid Search** -- Combines semantic + keyword + filters\
-   **Multi-Company Analysis** -- Compare filings across AAPL, MSFT,
    TSLA

### 🏢 Multi-Company Capabilities

-   Single-company deep analysis\
-   Side-by-side comparison of 2--3 companies\
-   Cross-company benchmarking\
-   Auto-detection of companies and years from queries

### ⚡ Advanced Filtering

-   Company filter (AAPL, MSFT, TSLA, All)\
-   Year filter (2023, 2024, or both)\
-   SEC 10-K section targeting\
-   Real-time semantic search using OpenAI embeddings

## 🏗️ System Architecture

*(Image placeholder)*

## 🚀 Installation

### Prerequisites

-   Python 3.9+\
-   OpenAI API key\
-   SEC API key\

### Step-by-Step Setup

``` bash
# 1. Clone the repository
git clone https://github.com/JeffyAugustine/FinRAG--A-Multi-Company-Retrieval-Augmented-Generation-System-for-Financial-SEC-Filings.git
cd FinRAG--A-Multi-Company-Retrieval-Augmented-Generation-System-for-Financial-SEC-Filings

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Add OpenAI API key inside .env
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
# Add SEC API key inside .env
SEC_API_KEY=your_sec_api_token_here

# 5.  Download SEC filings
python src/download_multi_company_api.py

# 6. Process & embed filings
python src/process_10k_files.py
python src/chunk_processor.py
python src/generate_embeddings.py

# 7. Launch the application
cd src
streamlit run streamlit_app.py
```

## 📊 Usage

### Running the App

``` bash
cd src
streamlit run streamlit_app.py
```

### Query Examples

#### Single-Company

-   "What are Apple's main competitive risks?"\
-   "Microsoft's cloud computing strategy"\
-   "What were Tesla's revenue in 2023?"

#### Multi-Company Comparison

-   "Compare risk factors of Apple and Microsoft"\
-   "How do Apple and Tesla differ in R&D spending?"\
-   "Compare business models across all three companies"

#### Filtered Queries

-   "Apple 2023 financial metrics"\
-   "Tesla risk factors in supply chain"\
-   "Microsoft legal proceedings 2024"

## 📈 Performance Evaluation

### Overall Search Method Performance

  Search Method   Score (/5.0)   Avg Response Time   Best For
  --------------- -------------- ------------------- ---------------------
  Semantic        **4.50 ⭐**    8.92s               Conceptual queries
  Hybrid          4.00           9.81s               Mixed queries
  Keyword         3.67           7.16s               Exact term matching

(Performance chart placeholder)

### Detailed Evaluation

-   **Overall Score:** 4.47/5.0 🏆\
-   **Faithfulness:** 4.60\
-   **Relevance:** 4.20\
-   **Citation Accuracy:** 4.60

By difficulty:\
- Easy: **5.0**\
- Medium: **4.11**\
- Hard: **5.0**

(Evaluation chart placeholder)

## 🎯 Search Method Recommendations

  Method     Best For                Example
  ---------- ----------------------- --------------------------------
  Semantic   Conceptual & abstract   "competitive market pressures"
  Hybrid     Mixed queries           "R&D expenses 2023"
  Keyword    Exact phrases           "single-source suppliers"

## 🏢 Multi-Company Analysis

### When to Use

Use multi-company mode for: - Financial metric comparison\
- Industry trend analysis\
- Competitive positioning\
- Cross-company risks

### Example Queries

-   "Compare Apple and Microsoft revenue streams"\
-   "Common risk factors for AAPL, MSFT, TSLA"\
-   "Tesla R&D vs Apple R&D"\
-   "Supply chain comparison of MSFT and AAPL"

### Features

-   Side-by-side comparison\
-   Benchmarking\
-   Industry insights\
-   Choose 2--3 companies

## 📁 Project Structure

    sec-filings-rag/
    ├── src/
    │   ├── streamlit_app.py
    │   ├── multicompany_semantic_rag.py
    │   ├── semantic_rag.py
    │   ├── hybrid_rag.py
    │   ├── download_multi_company_api.py
    │   ├── process_10k_files.py
    │   ├── chunk_processor.py
    │   └── generate_embeddings.py
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   ├── chunks/
    │   ├── embeddings/
    │   └── outputs/
    ├── screenshots/
    ├── requirements.txt
    ├── .env.example
    ├── .gitignore
    └── README.md

## 🔧 Technical Details

### Technologies

-   OpenAI API\
-   Streamlit\
-   BeautifulSoup\
-   NumPy\
-   Tiktoken

### Data Pipeline

1.  Download 10-K filings\
2.  Extract sections\
3.  Chunk text\
4.  Generate embeddings\
5.  Build vector index

## 📄 License

This project is for educational and research purposes. SEC filings are
public data from the U.S. SEC.

*Not financial advice.*
