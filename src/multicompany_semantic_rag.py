import openai
import numpy as np
import json
import os
from dotenv import load_dotenv
import time
import re
from typing import List, Dict, Optional, Union, Tuple
from collections import defaultdict
from datetime import datetime

load_dotenv()

class MultiCompanySemanticRAG:
    """Multi-company RAG system with filtering, comparison, and analytics"""
    
    def __init__(self, embeddings_path="../data/embeddings/multicompany_embedded_chunks.jsonl"):
        self.embeddings_path = embeddings_path
        self.chunks = self._load_embeddings()
        print(f" Loaded {len(self.chunks)} chunks with embeddings")
        self._build_index()
        
    def _load_embeddings(self):
        """Load our pre-computed multi-company embeddings"""
        chunks = []
        try:
            if not os.path.exists(self.embeddings_path):
                # Try alternative names
                alt_paths = [
                    "../data/embeddings/embedded_chunks.jsonl",
                    "../data/chunks/multicompany_chunks.jsonl"
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        self.embeddings_path = alt_path
                        print(f"  Using alternative file: {alt_path}")
                        break
            
            with open(self.embeddings_path, 'r', encoding='utf-8') as f:
                for line in f:
                    chunks.append(json.loads(line))
            
            # Print summary
            companies = defaultdict(int)
            years = defaultdict(int)
            sections = defaultdict(int)
            
            for chunk in chunks:
                meta = chunk.get('metadata', {})
                companies[meta.get('company', 'UNKNOWN')] += 1
                years[str(meta.get('year', 'UNKNOWN'))] += 1
                sections[meta.get('section', 'UNKNOWN')] += 1
            
            print(f" Companies: {', '.join(companies.keys())}")
            print(f" Years: {', '.join(years.keys())}")
            print(f" Sections: {', '.join(sections.keys())}")
            
            return chunks
            
        except Exception as e:
            print(f" Error loading embeddings: {e}")
            return []
    
    def _build_index(self):
        """Build fast lookup indices for filtering"""
        self.company_index = defaultdict(list)
        self.year_index = defaultdict(list)
        self.section_index = defaultdict(list)
        
        for i, chunk in enumerate(self.chunks):
            meta = chunk.get('metadata', {})
            company = meta.get('company', 'UNKNOWN')
            year = str(meta.get('year', 'UNKNOWN'))
            section = meta.get('section', 'UNKNOWN')
            
            self.company_index[company].append(i)
            self.year_index[year].append(i)
            self.section_index[section].append(i)
    
    def cosine_similarity(self, vec_a, vec_b):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot_product / (norm_a * norm_b)
    
    def filter_chunks(self, company: Optional[str] = None, 
                      year: Optional[Union[str, int]] = None, 
                      section: Optional[str] = None) -> List[Dict]:
        """Filter chunks by metadata"""
        if not any([company, year, section]):
            return self.chunks
        
        # Start with all indices
        valid_indices = set(range(len(self.chunks)))
        
        # Apply filters
        if company:
            company = company.upper()
            if company in self.company_index:
                valid_indices &= set(self.company_index[company])
            else:
                return []
        
        if year:
            year_str = str(year)
            if year_str in self.year_index:
                valid_indices &= set(self.year_index[year_str])
            else:
                return []
        
        if section:
            section_lower = section.lower()
            # Try to find matching section
            matching_sections = [s for s in self.section_index.keys() 
                               if section_lower in s.lower()]
            if matching_sections:
                section_indices = set()
                for sec in matching_sections:
                    section_indices |= set(self.section_index[sec])
                valid_indices &= section_indices
            else:
                return []
        
        return [self.chunks[i] for i in valid_indices]
    
    def detect_query_type(self, query: str) -> Dict:
        """Analyze query to determine company, year, and comparison intent"""
        query_lower = query.lower()
        
        # Company detection
        companies = {
            'apple': 'AAPL',
            'aapl': 'AAPL',
            'microsoft': 'MSFT',
            'msft': 'MSFT',
            'tesla': 'TSLA',
            'tsla': 'TSLA'
        }
        
        detected_companies = []
        for keyword, ticker in companies.items():
            if keyword in query_lower:
                detected_companies.append(ticker)
        
        # Year detection
        year_pattern = r'\b(20[0-9]{2})\b'
        years = re.findall(year_pattern, query)
        detected_years = [int(y) for y in years] if years else []
        
        # Section detection
        sections = {
            'business': 'business',
            'risk': 'risk_factors',
            'risk factor': 'risk_factors',
            'property': 'properties',
            'legal': 'legal',
            'mda': 'mda',
            'management discussion': 'mda',
            'financial': 'financials',
            'revenue': 'business',
            'product': 'business'
        }
        
        detected_sections = []
        for keyword, section in sections.items():
            if keyword in query_lower:
                detected_sections.append(section)
        
        # Comparison detection
        comparison_keywords = ['compare', 'versus', 'vs', 'vs.', 'difference', 'contrast']
        is_comparison = any(keyword in query_lower for keyword in comparison_keywords)
        
        return {
            'companies': detected_companies,
            'years': detected_years,
            'sections': list(set(detected_sections)),
            'is_comparison': is_comparison,
            'raw_query': query
        }
    
    def semantic_search(self, query: str, 
                       company: Optional[str] = None,
                       year: Optional[Union[str, int]] = None,
                       section: Optional[str] = None,
                       top_k: int = 5) -> List[Dict]:
        """Find most relevant chunks with filtering"""
        start_time = time.time()
        
        # Filter chunks first
        filtered_chunks = self.filter_chunks(company, year, section)
        
        if not filtered_chunks:
            print(f"  No chunks found for filters: company={company}, year={year}, section={section}")
            return []
        
        print(f"🔍 Searching {len(filtered_chunks)} filtered chunks...")
        
        # Embed the query
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=[query]
            )
            query_embedding = response.data[0].embedding
            
        except Exception as e:
            print(f" Error embedding query: {e}")
            return []
        
        # Calculate similarities with filtered chunks
        similarities = []
        for chunk in filtered_chunks:
            if 'embedding' not in chunk:
                continue
            similarity = self.cosine_similarity(query_embedding, chunk['embedding'])
            similarities.append((similarity, chunk))
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:top_k]
        
        search_time = time.time() - start_time
        print(f" Found {len(top_results)} results in {search_time:.2f}s")
        
        return [{"similarity": score, **chunk} for score, chunk in top_results]
    
    def compare_companies(self, query: str, companies: List[str], 
                         year: Optional[int] = None, top_k_per_company: int = 3) -> Dict:
        """Run comparative analysis between companies"""
        print(f" COMPARING {len(companies)} companies: {', '.join(companies)}")
        
        all_results = {}
        
        for company in companies:
            print(f"   Searching {company}...")
            results = self.semantic_search(query, company=company, year=year, top_k=top_k_per_company)
            all_results[company] = results
            
            if results:
                print(f"     Found {len(results)} relevant chunks")
            else:
                print(f"      No results found for {company}")
        
        return all_results
    
    def ask_question(self, question: str, 
                    company: Optional[str] = None,
                    year: Optional[Union[str, int]] = None,
                    section: Optional[str] = None,
                    comparison_mode: bool = False,
                    companies_to_compare: Optional[List[str]] = None) -> Dict:
        """Main Q&A function with multiple modes"""
        print(f"\n QUESTION: {question}")
        print(f"   MODE: {'Comparison' if comparison_mode else 'Single Company'}")
        print(f"   FILTERS: company={company}, year={year}, section={section}")
        
        # Auto-detect query type if not specified
        if not any([company, comparison_mode]):
            query_analysis = self.detect_query_type(question)
            print(f" Auto-detected: {query_analysis}")
            
            if query_analysis['is_comparison'] and query_analysis['companies']:
                comparison_mode = True
                companies_to_compare = query_analysis['companies'][:2]  # Compare first 2 detected
            
            if not company and query_analysis['companies'] and not comparison_mode:
                company = query_analysis['companies'][0]
            
            if not year and query_analysis['years']:
                year = query_analysis['years'][0]
            
            if not section and query_analysis['sections']:
                section = query_analysis['sections'][0]
        
        # Mode 1: Comparison
        if comparison_mode:
            if not companies_to_compare:
                companies_to_compare = ['AAPL', 'MSFT']  # Default comparison
            
            if len(companies_to_compare) < 2:
                return {"error": "Need at least 2 companies for comparison"}
            
            return self._generate_comparative_answer(question, companies_to_compare, year, section)
        
        # Mode 2: Single company query
        else:
            return self._generate_single_answer(question, company, year, section)
    
    def _generate_single_answer(self, question: str, 
                               company: Optional[str] = None,
                               year: Optional[Union[str, int]] = None,
                               section: Optional[str] = None) -> Dict:
        """Generate answer for single company query"""
        # Step 1: Semantic search with filtering
        search_results = self.semantic_search(question, company, year, section, top_k=5)
        
        if not search_results:
            return {
                "question": question,
                "answer": " No relevant information found for the specified filters.",
                "sources": [],
                "metadata": {
                    "company": company,
                    "year": year,
                    "section": section,
                    "mode": "single"
                }
            }
        
        # Step 2: Build context
        context_parts = []
        for i, result in enumerate(search_results):
            meta = result['metadata']
            context_parts.append(
                f"[Source {i+1}: {meta.get('company', 'Unknown')} {meta.get('year', 'Unknown')} "
                f"| Section: {meta.get('section', 'Unknown')} | "
                f"Similarity: {result['similarity']:.3f}]\n"
                f"{result['text']}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Step 3: Build prompt
        company_info = f" for {company}" if company else ""
        year_info = f" ({year})" if year else ""
        section_info = f" [Section: {section}]" if section else ""
        
        prompt = f"""Based ONLY on the following context from SEC 10-K filings{company_info}{year_info}{section_info}, answer the question.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer using ONLY the information from the context above
2. Be specific and include relevant numbers/dates when available
3. Cite your sources using the format [Source X]
4. If the information isn't in the context, say "I cannot find this information in the available documents"
5. Format your answer clearly with bullet points if appropriate
6. Mention which company and year you're referring to if relevant"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst analyzing SEC filings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            answer = response.choices[0].message.content
            
            # Step 4: Return structured result
            return {
                "question": question,
                "answer": answer,
                "sources": [
                    {
                        "chunk_id": result['metadata'].get('chunk_id', 'N/A'),
                        "company": result['metadata'].get('company', 'N/A'),
                        "year": result['metadata'].get('year', 'N/A'),
                        "section": result['metadata'].get('section', 'N/A'),
                        "similarity": round(result['similarity'], 3),
                        "text_preview": result['text'][:150] + "..."
                    }
                    for result in search_results
                ],
                "metadata": {
                    "company": company,
                    "year": year,
                    "section": section,
                    "mode": "single",
                    "tokens_used": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"❌ Error generating answer: {e}",
                "sources": [],
                "metadata": {}
            }
    
    def _generate_comparative_answer(self, question: str, 
                                    companies: List[str],
                                    year: Optional[int] = None,
                                    section: Optional[str] = None) -> Dict:
        """Generate comparative analysis answer"""
        # Step 1: Get results for each company
        all_results = self.compare_companies(question, companies, year, top_k_per_company=3)
        
        # Check if we have results for all companies
        companies_with_results = [c for c in companies if all_results.get(c)]
        if len(companies_with_results) < 2:
            return {
                "question": question,
                "answer": f"❌ Insufficient data for comparison. Only found data for: {', '.join(companies_with_results)}",
                "comparison": {},
                "metadata": {
                    "mode": "comparison",
                    "companies_requested": companies,
                    "companies_found": companies_with_results
                }
            }
        
        # Step 2: Build comparative context
        context_parts = []
        company_contexts = {}
        
        for company in companies_with_results:
            results = all_results[company]
            if not results:
                continue
            
            company_parts = []
            for i, result in enumerate(results):
                meta = result['metadata']
                company_parts.append(
                    f"[{company} Source {i+1}: {meta.get('section', 'Unknown')} | "
                    f"Similarity: {result['similarity']:.3f}]\n"
                    f"{result['text']}"
                )
            
            company_context = "\n\n".join(company_parts)
            company_contexts[company] = company_context
            context_parts.append(f"=== {company} ===\n{company_context}")
        
        context = "\n\n" + "\n\n".join(context_parts) + "\n\n"
        
        # Step 3: Build comparison prompt
        year_info = f" for the year {year}" if year else ""
        companies_str = " and ".join(companies_with_results)
        
        prompt = f"""Compare {companies_str} based on their SEC 10-K filings{year_info} to answer the question.

CONTEXT FOR EACH COMPANY:{context}

QUESTION: {question}

INSTRUCTIONS FOR COMPARATIVE ANALYSIS:
1. Compare and contrast each company's approach/position on this topic
2. Highlight key similarities and differences
3. Include specific numbers, dates, or metrics when available
4. Cite sources using format [{company} Source X]
5. If information is missing for a company, note that explicitly
6. Organize your answer with clear headings for each company
7. End with a summary comparing all companies

ANSWER FORMAT:
- Start with a brief introduction
- Have sections for each company
- Include a comparison table if relevant
- End with overall insights"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst comparing companies based on SEC filings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1200
            )
            
            answer = response.choices[0].message.content
            
            # Step 4: Return structured comparison result
            comparison_data = {}
            for company in companies_with_results:
                results = all_results[company]
                comparison_data[company] = {
                    "chunks_found": len(results),
                    "avg_similarity": round(np.mean([r['similarity'] for r in results]), 3) if results else 0,
                    "sections": list(set(r['metadata'].get('section', 'N/A') for r in results))
                }
            
            return {
                "question": question,
                "answer": answer,
                "comparison": comparison_data,
                "sources_by_company": {
                    company: [
                        {
                            "chunk_id": result['metadata'].get('chunk_id', 'N/A'),
                            "section": result['metadata'].get('section', 'N/A'),
                            "similarity": round(result['similarity'], 3),
                            "text_preview": result['text'][:120] + "..."
                        }
                        for result in results
                    ]
                    for company, results in all_results.items() if results
                },
                "metadata": {
                    "mode": "comparison",
                    "companies": companies_with_results,
                    "year": year,
                    "section": section,
                    "tokens_used": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f" Error generating comparative answer: {e}",
                "comparison": {},
                "metadata": {}
            }
    
    def save_conversation(self, results: Dict, filename: Optional[str] = None):
        """Save conversation results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = results.get('metadata', {}).get('mode', 'unknown')
            filename = f"multicompany_rag_conversation_{mode}_{timestamp}.json"
        
        output_dir = "../data/outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f" Conversation saved to: {filepath}")
        return filepath

def main():
    """Demo the multi-company semantic RAG system"""
    print(" MULTI-COMPANY SEMANTIC RAG SYSTEM")
    
    # Initialize RAG system
    rag = MultiCompanySemanticRAG()
    
    # Test scenarios
    test_scenarios = [
        # Single company queries
        {
            "question": "What are Apple's main products?",
            "company": "AAPL",
            "description": "Single company query"
        },
        {
            "question": "What risks does Tesla mention in their 2023 filing?",
            "company": "TSLA",
            "year": 2023,
            "description": "Company + year filter"
        },
        {
            "question": "What are Microsoft's financial metrics?",
            "company": "MSFT",
            "section": "financials",
            "description": "Company + section filter"
        },
        
        # Auto-detection queries
        {
            "question": "What are Apple's risk factors in 2024?",
            "description": "Auto-detect company and year"
        },
        {
            "question": "Tell me about Tesla's legal proceedings",
            "description": "Auto-detect company and section"
        },
        
        # Comparison queries
        {
            "question": "Compare the business models of Apple and Microsoft",
            "comparison_mode": True,
            "companies": ["AAPL", "MSFT"],
            "description": "Comparison between two companies"
        },
        {
            "question": "Compare risk factors across all three companies",
            "comparison_mode": True,
            "companies": ["AAPL", "MSFT", "TSLA"],
            "description": "Three-way comparison"
        },
        {
            "question": "How do Apple and Tesla differ in their R&D spending in 2023?",
            "comparison_mode": True,
            "year": 2023,
            "description": "Comparison with year filter"
        },
        
        # Natural language queries (auto-detection)
        {
            "question": "What risks does Apple face versus Microsoft?",
            "description": "Natural comparison query"
        },
        {
            "question": "Compare Tesla 2023 and Microsoft 2024 financials",
            "description": "Natural query with years"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"TEST {i}: {scenario['description']}")
        print(f"Q: {scenario['question']}")
        
        result = rag.ask_question(
            question=scenario['question'],
            company=scenario.get('company'),
            year=scenario.get('year'),
            section=scenario.get('section'),
            comparison_mode=scenario.get('comparison_mode', False),
            companies_to_compare=scenario.get('companies')
        )
        
        # Display results
        if 'answer' in result:
            print(f"\n ANSWER:\n{result['answer']}\n")
            
            if result.get('metadata', {}).get('mode') == 'single':
                print(" SOURCES:")
                for source in result.get('sources', []):
                    print(f"   - [{source['company']} {source['year']}] {source['section']} "
                          f"(score: {source['similarity']})")
                    print(f"     {source['text_preview']}")
            
            elif result.get('metadata', {}).get('mode') == 'comparison':
                print(" COMPARISON STATS:")
                for company, data in result.get('comparison', {}).items():
                    print(f"   {company}: {data['chunks_found']} chunks, "
                          f"avg score: {data['avg_similarity']}, "
                          f"sections: {', '.join(data['sections'])}")
        
        print(f" Tokens used: {result.get('metadata', {}).get('tokens_used', 'N/A')}")
        
        # Save this conversation
        save_file = rag.save_conversation(result)
        
        # Pause between tests
        if i < len(test_scenarios):
            time.sleep(2)

if __name__ == "__main__":
    main()