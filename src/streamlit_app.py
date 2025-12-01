import streamlit as st
import openai
import numpy as np
import json
import time
from dotenv import load_dotenv
import os
import pandas as pd
from typing import List, Dict
import sys

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="SEC 10-K Analyst Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Path adjustments for your structure
current_dir = os.path.dirname(os.path.abspath(__file__))  # src directory
project_root = os.path.dirname(current_dir)  # project root (parent of src)

class StreamlitRAG:
    """RAG system optimized for Streamlit interface"""
    
    def __init__(self):
        # Path to Apple-only embeddings
        embeddings_path = os.path.join(project_root, "data", "embeddings", "embedded_chunks.jsonl")
        self.embeddings_path = embeddings_path
        self.chunks = self._load_embeddings()
    
    def _load_embeddings(self):
        """Load pre-computed embeddings"""
        try:
            with open(self.embeddings_path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f]
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
            return []
    
    def cosine_similarity(self, vec_a, vec_b):
        """Calculate cosine similarity"""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        return dot_product / (norm_a * norm_b)
    
    def semantic_search(self, query, top_k=3):
        """Find most relevant chunks"""
        # Embed the query
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        query_embedding = response.data[0].embedding
        
        # Calculate similarities
        similarities = []
        for chunk in self.chunks:
            similarity = self.cosine_similarity(query_embedding, chunk['embedding'])
            similarities.append((similarity, chunk))
        
        # Sort and get top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:top_k]
        
        return [{"similarity": score, **chunk} for score, chunk in top_results]
    
    def ask_question(self, question, top_k=3):
        """Main Q&A function"""
        # Semantic search
        search_results = self.semantic_search(question, top_k)
        
        if not search_results:
            return None
        
        # Build context
        context_parts = []
        for i, result in enumerate(search_results):
            context_parts.append(
                f"[Source {i+1}: {result['metadata']['chunk_id']} | "
                f"Section: {result['metadata']['section']} | "
                f"Year: {result['metadata']['year']} | "
                f"Similarity: {result['similarity']:.3f}]\n"
                f"{result['text']}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = f"""Based ONLY on the following context from Apple's SEC filings, answer the question.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer using ONLY the information from the context above
- Be specific and include relevant numbers/dates when available
- Cite your sources using the format [Source X]
- If the information isn't in the context, say "I cannot find this information in the available documents"
- Format your answer clearly with bullet points if appropriate"""

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "You are a financial analyst analyzing SEC filings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            return {
                "question": question,
                "answer": answer,
                "sources": [
                    {
                        "chunk_id": result['metadata']['chunk_id'],
                        "section": result['metadata']['section'],
                        "year": result['metadata']['year'],
                        "similarity": round(result['similarity'], 3),
                        "search_type": "semantic",
                        "text_preview": result['text'][:150] + "...",
                        "full_text": result['text']
                    }
                    for result in search_results
                ],
                "tokens_used": response.usage.total_tokens,
                "search_time": time.time()
            }
            
        except Exception as e:
            st.error(f"Error generating answer: {e}")
            return None

class HybridSearchSystem:
    """Wrapper for hybrid search functionality"""
    
    def __init__(self):
        # Import from same directory (src)
        from hybrid_rag import HybridRAG
        self.hybrid_system = HybridRAG()
    
    def ask_question(self, question, top_k=3, filters=None):
        """Ask question using hybrid search"""
        result = self.hybrid_system.ask_question(
            question=question,
            top_k=top_k,
            filters=filters,
            search_type="hybrid"
        )
        
        if 'error' not in result:
            # Convert to same format as semantic results
            for source in result['sources']:
                source['full_text'] = next(
                    (chunk['text'] for chunk in self.hybrid_system.hybrid_search.chunks 
                     if chunk['metadata']['chunk_id'] == source['chunk_id']),
                    source['text_preview']
                )
        return result

class MultiCompanyRAG:
    """Multi-company RAG system for Streamlit"""
    
    def __init__(self):
        try:
            # Import from same directory (src)
            from multicompany_semantic_rag import MultiCompanySemanticRAG
            
            # Path to multi-company embeddings
            multicompany_path = os.path.join(project_root, "data", "embeddings", "multicompany_embedded_chunks.jsonl")
            apple_only_path = os.path.join(project_root, "data", "embeddings", "embedded_chunks.jsonl")
            
            # Use multi-company if exists, fallback to Apple-only
            if os.path.exists(multicompany_path):
                embeddings_path = multicompany_path
                st.sidebar.success("✅ Using multi-company data (AAPL+MSFT+TSLA)")
            else:
                embeddings_path = apple_only_path
                st.sidebar.warning("⚠️ Using Apple-only data (multi-company not found)")
            
            self.rag_system = MultiCompanySemanticRAG(embeddings_path)
            self.initialized = True
        except Exception as e:
            st.error(f"Failed to initialize Multi-Company RAG: {e}")
            self.initialized = False
    
    def ask_question(self, question, top_k=3, company=None, year=None, 
                     section=None, comparison_mode=False, companies_to_compare=None):
        """Ask question using multi-company RAG"""
        if not self.initialized:
            return {"error": "Multi-Company RAG not initialized"}
        
        try:
            # Use the multi-company RAG system
            result = self.rag_system.ask_question(
                question=question,
                company=company,
                year=year,
                section=section,
                comparison_mode=comparison_mode,
                companies_to_compare=companies_to_compare
            )
            
            # Convert to consistent format
            if isinstance(result, dict):
                if 'sources' in result:
                    for source in result['sources']:
                        source['search_type'] = 'multicompany'
                        source['full_text'] = source.get('text_preview', '') + "..."
                else:
                    # For comparison results, extract sources from companies
                    all_sources = []
                    if 'sources_by_company' in result:
                        for company_sources in result['sources_by_company'].values():
                            for source in company_sources:
                                source['search_type'] = 'multicompany'
                                source['full_text'] = source.get('text_preview', '') + "..."
                                all_sources.append(source)
                        result['sources'] = all_sources
                
                # Ensure tokens_used is present
                if 'tokens_used' not in result.get('metadata', {}):
                    result['tokens_used'] = result.get('metadata', {}).get('tokens_used', 0)
            
            return result
            
        except Exception as e:
            st.error(f"Error in multi-company query: {e}")
            return {"error": str(e)}

def main():
    """Main Streamlit application"""
    
    # Sidebar
    st.sidebar.title("🔍 SEC 10-K Analyst Bot")
    st.sidebar.markdown("Ask questions about SEC 10-K filings")
    
    # Search configuration - UPDATED WITH MULTI-COMPANY OPTION
    st.sidebar.subheader("🔧 Search Configuration")
    search_method = st.sidebar.radio(
        "Search Method",
        ["Semantic Search", "Hybrid Search", "Multi-Company Questions"],
        help="""
        Semantic: Better for conceptual questions. 
        Hybrid: Combines semantic + keyword search.
        Multi-Company: Compare Apple, Microsoft, and Tesla filings.
        """
    )
    
    # Initialize filters based on search method
    year_filter = None
    section_filter = None
    company_filter = None
    comparison_mode = False
    companies_to_compare = None
    
    if search_method == "Hybrid Search":
        st.sidebar.subheader("🎯 Search Filters")
        year_filter = st.sidebar.multiselect(
            "Filter by Year",
            [2023, 2024],
            default=[2023, 2024]
        )
        section_filter = st.sidebar.multiselect(
            "Filter by Section",
            ["business", "risk_factors", "mda", "financials", "legal", "properties"],
            default=["business", "risk_factors", "mda", "financials", "legal", "properties"]
        )
    
    elif search_method == "Multi-Company Questions":
        st.sidebar.subheader("🏢 Company Selection")
        
        # Company filter for single queries
        company_filter = st.sidebar.selectbox(
            "Select Company",
            ["All Companies", "Apple (AAPL)", "Microsoft (MSFT)", "Tesla (TSLA)"],
            index=0
        )
        
        # Year filter
        year_filter = st.sidebar.multiselect(
            "Filter by Year",
            [2023, 2024, "All Years"],
            default=["All Years"]
        )
        if "All Years" in year_filter:
            year_filter = [2023, 2024]
        
        # Section filter
        section_filter = st.sidebar.multiselect(
            "Filter by Section",
            ["All Sections", "business", "risk_factors", "mda", "financials", "legal", "properties"],
            default=["All Sections"]
        )
        if "All Sections" in section_filter:
            section_filter = ["business", "risk_factors", "mda", "financials", "legal", "properties"]
        
        # Comparison mode
        st.sidebar.subheader("🔄 Comparison Mode")
        comparison_mode = st.sidebar.checkbox("Enable Company Comparison", value=False)
        
        if comparison_mode:
            companies_to_compare = st.sidebar.multiselect(
                "Select Companies to Compare",
                ["Apple (AAPL)", "Microsoft (MSFT)", "Tesla (TSLA)"],
                default=["Apple (AAPL)", "Microsoft (MSFT)"]
            )
            # Convert display names to tickers
            companies_to_compare = [
                "AAPL" if "Apple" in c else "MSFT" if "Microsoft" in c else "TSLA" 
                for c in companies_to_compare
            ]
        else:
            # Convert single company selection to ticker
            if company_filter == "Apple (AAPL)":
                company_filter = "AAPL"
            elif company_filter == "Microsoft (MSFT)":
                company_filter = "MSFT"
            elif company_filter == "Tesla (TSLA)":
                company_filter = "TSLA"
            else:
                company_filter = None  # All companies
    
    # Initialize systems
    if 'rag_system' not in st.session_state:
        with st.spinner("Loading RAG system..."):
            st.session_state.rag_system = StreamlitRAG()
            st.session_state.chat_history = []
    
    if 'hybrid_system' not in st.session_state and search_method == "Hybrid Search":
        with st.spinner("Loading hybrid search system..."):
            st.session_state.hybrid_system = HybridSearchSystem()
    
    if 'multicompany_system' not in st.session_state and search_method == "Multi-Company Questions":
        with st.spinner("Loading multi-company system..."):
            st.session_state.multicompany_system = MultiCompanyRAG()
    
    # Main content
    st.title("📊 SEC 10-K Financial Analyst Bot")
    
    if search_method == "Semantic Search":
        st.markdown("""
        **Semantic Search Mode**: Uses AI to understand the meaning of your question and find conceptually relevant content.
        - Best for: Conceptual questions, business analysis, complex topics
        - **Data**: Apple 2023-2024 10-K filings only
        """)
    elif search_method == "Hybrid Search":
        st.markdown("""
        **Hybrid Search Mode**: Combines semantic understanding with keyword matching and filters.
        - Best for: Exact terms, specific numbers, filtered searches
        - **Data**: Apple 2023-2024 10-K filings only
        """)
    else:  # Multi-Company Questions
        st.markdown("""
        **Multi-Company Mode**: Compare and analyze across multiple companies.
        - **Companies**: Apple (AAPL), Microsoft (MSFT), Tesla (TSLA)
        - **Years**: 2023 and 2024 filings
        - **Features**: Single company queries, company comparisons, cross-company analysis
        - **Examples**: "Compare Apple and Microsoft revenue", "Tesla 2023 risk factors"
        """)
    
    # Question input
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            f"Ask a question about {search_method}:",
            placeholder=get_placeholder(search_method),
            key="question_input"
        )
    with col2:
        top_k = st.selectbox("Results to show", [3, 5, 7], index=0)
    
    # Process question
    if question:
        with st.spinner(f"🔍 Searching SEC filings using {search_method}..."):
            if search_method == "Semantic Search":
                result = st.session_state.rag_system.ask_question(question, top_k)
            elif search_method == "Hybrid Search":
                # Build filters for hybrid search
                filters = {}
                if year_filter:
                    filters['year'] = year_filter
                if section_filter:
                    filters['section'] = section_filter
                
                result = st.session_state.hybrid_system.ask_question(
                    question, top_k, filters if filters else None
                )
            else:  # Multi-Company Questions
                # Convert filters for multi-company system
                year = year_filter[0] if year_filter and len(year_filter) == 1 else None
                section = section_filter[0] if section_filter and len(section_filter) == 1 else None
                
                result = st.session_state.multicompany_system.ask_question(
                    question=question,
                    top_k=top_k,
                    company=company_filter if not comparison_mode else None,
                    year=year,
                    section=section,
                    comparison_mode=comparison_mode,
                    companies_to_compare=companies_to_compare if comparison_mode else None
                )
        
        if result and 'error' not in result:
            # Store in chat history with search method
            result['search_method'] = search_method
            if search_method == "Multi-Company Questions":
                result['company_filter'] = company_filter
                result['comparison_mode'] = comparison_mode
            elif search_method == "Hybrid Search":
                result['filters'] = {"year": year_filter, "section": section_filter} if year_filter or section_filter else None
            
            st.session_state.chat_history.insert(0, result)
            
            # Display answer
            st.subheader("💡 Answer")
            st.markdown(result['answer'])
            
            # Display search info
            if search_method == "Multi-Company Questions":
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Search Method", "Multi-Company")
                with col2:
                    if comparison_mode:
                        st.metric("Mode", "Comparison")
                    else:
                        st.metric("Mode", "Single Company")
                with col3:
                    if comparison_mode and companies_to_compare:
                        st.metric("Companies", len(companies_to_compare))
                    elif company_filter:
                        st.metric("Company", company_filter)
                    else:
                        st.metric("Company", "All")
                with col4:
                    st.metric("Sources", len(result.get('sources', [])))
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Search Method", search_method)
                with col2:
                    st.metric("Sources Found", len(result.get('sources', [])))
                with col3:
                    if search_method == "Hybrid Search" and result.get('filters'):
                        st.metric("Filters Applied", "Yes")
                    else:
                        st.metric("Filters Applied", "No")
            
            # Display sources with different styling for multi-company
            st.subheader("📚 Sources")
            
            sources = result.get('sources', [])
            if not sources and 'sources_by_company' in result:
                # For comparison results without merged sources
                st.info("Sources are organized by company in the comparison results above.")
            elif sources:
                for i, source in enumerate(sources):
                    if search_method == "Multi-Company Questions":
                        company = source.get('company', 'Unknown')
                        company_badge = f"🏢 {company}"
                        with st.expander(f"{company_badge} | {source.get('section', 'Unknown').replace('_', ' ').title()} | Score: {source.get('similarity', 0):.3f}"):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.metric("Company", company)
                                st.metric("Year", source.get('year', 'Unknown'))
                                st.metric("Section", source.get('section', 'Unknown').replace('_', ' ').title())
                                st.metric("Similarity", f"{source.get('similarity', 0):.3f}")
                            with col2:
                                st.text_area(
                                    "Content",
                                    value=source.get('full_text', 'No content available'),
                                    height=200,
                                    key=f"mc_source_{i}_{hash(question)}"
                                )
                    else:
                        search_type_badge = f"🔍 {source.get('search_type', 'semantic').upper()}"
                        with st.expander(f"{search_type_badge} | Source {i+1}: {source['chunk_id']} (Score: {source['similarity']:.3f})"):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.metric("Section", source['section'].replace('_', ' ').title())
                                st.metric("Year", source['year'])
                                st.metric("Similarity", f"{source['similarity']:.3f}")
                                
                                # Show breakdown for hybrid results
                                if source.get('search_type') == 'hybrid':
                                    st.metric("Semantic Score", f"{source.get('semantic_score', 0):.3f}")
                                    st.metric("Keyword Score", f"{source.get('keyword_score', 0):.3f}")
                            
                            with col2:
                                st.text_area(
                                    "Content",
                                    value=source['full_text'],
                                    height=200,
                                    key=f"source_{i}_{hash(question)}"
                                )
            
            # Performance metrics
            st.subheader("⚡ Performance")
            if search_method == "Multi-Company Questions":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tokens Used", result.get('tokens_used', result.get('metadata', {}).get('tokens_used', 0)))
                with col2:
                    if sources:
                        avg_similarity = np.mean([s.get('similarity', 0) for s in sources])
                        st.metric("Avg Similarity", f"{avg_similarity:.3f}")
                    else:
                        st.metric("Avg Similarity", "N/A")
                with col3:
                    if 'comparison' in result:
                        companies_found = list(result['comparison'].keys())
                        st.metric("Companies Found", len(companies_found))
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tokens Used", result['tokens_used'])
                with col2:
                    avg_similarity = np.mean([s['similarity'] for s in sources])
                    st.metric("Avg Similarity", f"{avg_similarity:.3f}")
                with col3:
                    if search_method == "Hybrid Search":
                        hybrid_sources = [s for s in sources if s.get('search_type') == 'hybrid']
                        semantic_sources = [s for s in sources if s.get('search_type') == 'semantic']
                        keyword_sources = [s for s in sources if s.get('search_type') == 'keyword']
                        st.metric("Source Types", f"H:{len(hybrid_sources)} S:{len(semantic_sources)} K:{len(keyword_sources)}")
        
        elif result and 'error' in result:
            st.error(f"Search error: {result['error']}")
        else:
            st.warning("No relevant information found for your question.")
    
    # Chat history with search method badges
    if st.session_state.chat_history:
        st.sidebar.subheader("💬 Recent Questions")
        for i, chat in enumerate(st.session_state.chat_history[:5]):
            method_badges = {
                "Semantic Search": "🔷",
                "Hybrid Search": "🔶", 
                "Multi-Company Questions": "🏢"
            }
            badge = method_badges.get(chat.get('search_method'), "❓")
            button_text = f"{badge} {chat['question'][:45]}..."
            if st.sidebar.button(button_text, key=f"history_{i}"):
                st.session_state.question_input = chat['question']
                st.rerun()
    
    # Sample questions optimized for different search methods
    st.sidebar.subheader("💡 Sample Questions")
    
    if search_method == "Semantic Search":
        sample_questions = [
            "What are Apple's main competitive risks?",
            "How does Apple's business model work?",
            "What are the strategic challenges Apple faces?",
            "How does Apple manage innovation?",
            "What is Apple's approach to sustainability?"
        ]
    elif search_method == "Hybrid Search":
        sample_questions = [
            "R&D expenses 2023",
            "single-source suppliers in risk factors", 
            "iPhone revenue 2024",
            "legal proceedings antitrust",
            "supply chain dependencies Asia"
        ]
    else:  # Multi-Company Questions
        sample_questions = [
            "Compare Apple and Microsoft business models",
            "Tesla 2023 risk factors",
            "What are Microsoft's main products?",
            "Compare R&D spending across all companies",
            "Apple vs Tesla innovation strategy"
        ]
    
    for q in sample_questions:
        if st.sidebar.button(q, key=f"sample_{hash(q)}"):
            st.session_state.question_input = q
            st.rerun()
    
    # About section
    st.sidebar.markdown("---")
    if search_method == "Multi-Company Questions":
        st.sidebar.markdown("""
        **About Multi-Company Mode:**
        - **Companies**: AAPL, MSFT, TSLA (2023-2024)
        - **Sections**: Business, Risk Factors, Financials, MDA, Legal, Properties
        - **Features**: Single queries, comparisons, cross-analysis
        - **Auto-detection**: Detects companies/years from natural language
        """)
    else:
        st.sidebar.markdown("""
        **About This App:**
        - **Semantic Search**: AI-powered conceptual understanding
        - **Hybrid Search**: Semantic + keyword + filters
        - **Data**: Apple 2023-2024 10-K filings
        - **Embeddings**: OpenAI embeddings
        - **LLM**: GPT-3.5-turbo for answer generation
        - **All answers are sourced and verifiable**
        """)

def get_placeholder(search_method):
    """Get placeholder text based on search method"""
    if search_method == "Semantic Search":
        return "e.g., What are Apple's main competitive risks?"
    elif search_method == "Hybrid Search":
        return "e.g., R&D expenses 2023"
    else:  # Multi-Company Questions
        return "e.g., Compare Apple and Microsoft business models"

if __name__ == "__main__":
    main()