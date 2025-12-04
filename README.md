# 🏢 FinRAG--A-Multi-Company-Retrieval-Augmented-Generation-System-for-Financial-SEC-Filings

A Retrieval-Augmented Generation (RAG) platform for analyzing and
comparing SEC 10-K filings across Apple, Microsoft, and Tesla using
advanced semantic search and AI-driven insights.


## 🎯 Overview

This project implements a sophisticated RAG (Retrieval-Augmented
Generation) system that enables financial analysts, researchers, and
investors to:

-   Analyze individual SEC 10-K filings with semantic precision  
-   Compare financial disclosures across multiple companies  
-   Extract key insights using AI-powered natural language queries  
-   Filter results by year, section, and company for targeted analysis

Supported Companies: **Apple (AAPL), Microsoft (MSFT), Tesla (TSLA)**  
Years: **2023--2024 SEC 10-K filings**

## ✨ Key Features

### 🔍 Three Search Modes

-   **Semantic Search** -- AI-powered conceptual understanding  
-   **Hybrid Search** -- Combines semantic + keyword + filters  
-   **Multi-Company Analysis** -- Compare filings across AAPL, MSFT,
    TSLA

### 🏢 Multi-Company Capabilities

-   Single-company deep analysis  
-   Side-by-side comparison of 2--3 companies  
-   Cross-company benchmarking  
-   Auto-detection of companies and years from queries

### ⚡ Advanced Filtering

-   Company filter (AAPL, MSFT, TSLA, All)  
-   Year filter (2023, 2024, or both)  
-   SEC 10-K section targeting  
-   Real-time semantic search using OpenAI embeddings

## 🏗️ System Architecture

*![Flowchart](https://github.com/user-attachments/assets/79aef0d1-66ea-4980-8761-7ac33863fc56)*

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
create .env
# Add OpenAI API key inside .env
OPENAI_API_KEY=your-actual-openai-api-key-here
# Add SEC API key inside .env
SEC_API_KEY=your_sec_api_token_here

# 5.  Download SEC filings
python src/download_multi_company_api.py

# 6. Process & embed filings
python src/multi_processor.py
python src/multi_chunking.py
python src/multi_embedding.py

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

-   "What are Apple's main competitive risks?"
-   "Microsoft's cloud computing strategy"
-   "What were Tesla's revenue in 2023?"

#### Multi-Company Comparison

-   "Compare risk factors of Apple and Microsoft"
-   "How do Apple and Tesla differ in R&D spending?"
-   "Compare business models across all three companies"

#### Filtered Queries

-   "Apple 2023 financial metrics"
-   "Tesla risk factors in supply chain"
-   "Microsoft legal proceedings 2024"

## 📈 Performance Evaluation
### IR Metrics (Retrieval Accuracy)

| Method | Precision@3 | Recall@10 | MRR | Avg Time |
|--------|-------------|-----------|-----|----------|
| **Hybrid** | **0.778** | 0.122 | **0.833** | 2.78s |
| **Semantic** | 0.667 | 0.122 | **0.833** | 2.46s |
| **Keyword** | 0.556 | 0.114 | 0.500 | **2.11s** |

### RAG Metrics (Retrieval Quality)

| Method | Semantic Cohesion | Usable Chunks | Unique Sections | Avg Chunk Length |
|--------|-------------------|---------------|-----------------|------------------|
| **Keyword** | **0.633** | **100%** | 2.0 | 294 words |
| **Semantic** | 0.582 | **100%** | **2.3** | **338 words** |
| **Hybrid** | 0.481 | **100%** | **2.3** | **338 words** |

### Overall Search Method Performance

| Search Method | Score (/5.0) | Avg Response Time | Best For             |
|---------------|--------------|-------------------|----------------------|
| **Semantic**  | **4.50 ⭐**   | 8.92s             | Conceptual queries   |
| Hybrid        | 4.00         | 9.81s             | Mixed queries        |
| Keyword       | 3.67         | 7.16s             | Exact term matching  |

*<img width="3569" height="1470" alt="hybrid_comparison_charts" src="https://github.com/user-attachments/assets/b2eb8307-ae31-4b60-9b4a-2d7188d6d221" />*

### Detailed Evaluation

-   **Overall Score:** 4.47/5.0 🏆
-   **Faithfulness:** 4.60
-   **Relevance:** 4.20
-   **Citation Accuracy:** 4.60

By difficulty:
- Easy: **5.0**
- Medium: **4.11**
- Hard: **5.0**

*<img width="3569" height="2965" alt="evaluation_results" src="https://github.com/user-attachments/assets/332e9d5a-5e68-4677-a74d-2df398ff7588" />*

## 🎯 Search Method Recommendations

| Method   | Best For                 | Example                        |
|----------|---------------------------|--------------------------------|
| Semantic | Conceptual & abstract     | "competitive market pressures" |
| Hybrid   | Mixed queries             | "R&D expenses 2023"            |
| Keyword  | Exact phrases             | "single-source suppliers"      |

*<img width="2969" height="1769" alt="search_method_recommendations" src="https://github.com/user-attachments/assets/2dd40463-817b-4a1d-8058-dfad88472926" />*

## 🏢 Multi-Company Analysis

### When to Use

Use multi-company mode for: - Financial metric comparison\
- Industry trend analysis
- Competitive positioning
- Cross-company risks

### Example Queries

-   "Compare Apple and Microsoft revenue streams"
-   "Common risk factors for AAPL, MSFT, TSLA"
-   "Tesla R&D vs Apple R&D"
-   "Supply chain comparison of MSFT and AAPL"

### Features

-   Side-by-side comparison
-   Benchmarking
-   Industry insights
-   Choose 2--3 companies

## 🔧 Technical Details

### Technologies

-   OpenAI API
-   Streamlit
-   BeautifulSoup
-   NumPy
-   Tiktoken

### Data Pipeline

1.  Download 10-K filings
2.  Extract sections
3.  Chunk text
4.  Generate embeddings
5.  Build vector index

## 📄 License

This project is for educational and research purposes. SEC filings are
public data from the U.S. SEC.

*Not financial advice.*
