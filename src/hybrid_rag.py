from semantic_rag import SimpleSemanticRAG
from keyword_search import KeywordSearch
from typing import List, Dict
import openai
import time

class HybridRAG:
    """Combines semantic and keyword search for better retrieval"""
    
    def __init__(self, embeddings_path="../data/embeddings/embedded_chunks.jsonl"):
        self.semantic_search = SimpleSemanticRAG(embeddings_path)
        self.keyword_search = KeywordSearch(embeddings_path)
    
    def search_with_type(self, query: str, top_k: int = 3, filters: Dict = None, 
                        search_type: str = "hybrid") -> List[Dict]:
        """Search with proper type handling"""
        
        if search_type == "semantic":
            results = self.semantic_search.semantic_search(query, top_k)
            if filters:
                results = [r for r in results if self._matches_filters(r['metadata'], filters)]
            # Label as semantic
            for result in results:
                result['search_type'] = 'semantic'
                
        elif search_type == "keyword":
            results = self.keyword_search.search(query, top_k, filters)
            # Convert keyword results to same format and label properly
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'similarity': result['score'],  # Use score as similarity
                    'search_type': 'keyword',
                    'keyword_score': result['score'],
                    'semantic_score': 0
                })
            results = formatted_results
            
        else:  # hybrid
            semantic_results = self.semantic_search.semantic_search(query, top_k)
            keyword_results = self.keyword_search.search(query, top_k, filters)
            
            if filters:
                semantic_results = [r for r in semantic_results if self._matches_filters(r['metadata'], filters)]
            
            # Combine and re-rank
            all_results = {}
            
            # Add semantic results
            for result in semantic_results:
                chunk_id = result['metadata']['chunk_id']
                all_results[chunk_id] = {
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'semantic_score': result['similarity'],
                    'keyword_score': 0,
                    'combined_score': result['similarity'] * 0.7,
                    'search_type': 'hybrid'
                }
            
            # Add keyword results
            for result in keyword_results:
                chunk_id = result['metadata']['chunk_id']
                if chunk_id in all_results:
                    # Update existing
                    all_results[chunk_id]['keyword_score'] = result['score']
                    all_results[chunk_id]['combined_score'] += result['score'] * 0.3
                else:
                    # New entry
                    all_results[chunk_id] = {
                        'text': result['text'],
                        'metadata': result['metadata'],
                        'semantic_score': 0,
                        'keyword_score': result['score'],
                        'combined_score': result['score'] * 0.3,
                        'search_type': 'hybrid'
                    }
            
            # Convert to final format
            ranked = sorted(all_results.values(), key=lambda x: x['combined_score'], reverse=True)
            results = []
            for item in ranked[:top_k]:
                results.append({
                    'text': item['text'],
                    'metadata': item['metadata'],
                    'similarity': item['combined_score'],
                    'search_type': item['search_type'],
                    'semantic_score': item['semantic_score'],
                    'keyword_score': item['keyword_score']
                })
        
        return results
    
    def ask_question(self, question: str, top_k: int = 3, filters: Dict = None, 
                    search_type: str = "hybrid") -> Dict:
        """Main Q&A with proper search type handling"""
        
        results = self.search_with_type(question, top_k, filters, search_type)
        
        if not results:
            return {"error": "No relevant information found"}
        
        # Build context
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(
                f"[Source {i+1}: {result['metadata']['chunk_id']} | "
                f"Section: {result['metadata']['section']} | "
                f"Year: {result['metadata']['year']} | "
                f"Score: {result['similarity']:.3f} | "
                f"Type: {result['search_type']}]\n"
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
- If the information isn't in the context, say so"""

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
            
            return {
                "question": question,
                "answer": answer,
                "sources": [
                    {
                        "chunk_id": result['metadata']['chunk_id'],
                        "section": result['metadata']['section'],
                        "year": result['metadata']['year'],
                        "similarity": round(result['similarity'], 3),
                        "search_type": result['search_type'],
                        "semantic_score": round(result.get('semantic_score', 0), 3),
                        "keyword_score": round(result.get('keyword_score', 0), 3),
                        "text_preview": result['text'][:100] + "..."
                    }
                    for result in results
                ],
                "search_type": search_type,
                "filters_applied": filters,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            return {"error": f"Error generating answer: {e}"}
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if chunk matches all filters"""
        for key, value in filters.items():
            if key in metadata:
                if isinstance(value, list):
                    if metadata[key] not in value:
                        return False
                else:
                    if metadata[key] != value:
                        return False
        return True