import openai
import numpy as np
import json
import os
from dotenv import load_dotenv
import time

load_dotenv()

class SimpleSemanticRAG:
    """Simple semantic RAG system using our pre-computed embeddings"""
    
    def __init__(self, embeddings_path="../data/embeddings/embedded_chunks.jsonl"):
        self.embeddings_path = embeddings_path
        self.chunks = self._load_embeddings()
        print(f" Loaded {len(self.chunks)} chunks with embeddings")
        
    def _load_embeddings(self):
        """Load our pre-computed embeddings and chunks"""
        chunks = []
        try:
            with open(self.embeddings_path, 'r', encoding='utf-8') as f:
                for line in f:
                    chunks.append(json.loads(line))
            return chunks
        except Exception as e:
            print(f" Error loading embeddings: {e}")
            return []
    
    def cosine_similarity(self, vec_a, vec_b):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        return dot_product / (norm_a * norm_b)
    
    def semantic_search(self, query, top_k=3):
        """Find most relevant chunks for a query"""
        start_time = time.time()
        
        # Embed the query
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        query_embedding = response.data[0].embedding
        
        # Calculate similarities with all chunks
        similarities = []
        for chunk in self.chunks:
            similarity = self.cosine_similarity(query_embedding, chunk['embedding'])
            similarities.append((similarity, chunk))
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:top_k]
        
        search_time = time.time() - start_time
        
        print(f"🔍 Found {len(top_results)} results in {search_time:.2f}s")
        return [{"similarity": score, **chunk} for score, chunk in top_results]
    
    def ask_question(self, question, top_k=3):
        """Main Q&A function"""
        print(f"\n QUESTION: {question}")
        
        # Step 1: Semantic search
        search_results = self.semantic_search(question, top_k)
        
        if not search_results:
            return " No relevant information found in the documents."
        
        # Step 2: Build context for GPT
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
        
        # Step 3: Generate answer with GPT
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
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst analyzing SEC filings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # Step 4: Return structured result
            return {
                "question": question,
                "answer": answer,
                "sources": [
                    {
                        "chunk_id": result['metadata']['chunk_id'],
                        "section": result['metadata']['section'],
                        "year": result['metadata']['year'],
                        "similarity": round(result['similarity'], 3),
                        "text_preview": result['text'][:100] + "..."
                    }
                    for result in search_results
                ],
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            return f" Error generating answer: {e}"

def main():
    """Demo the semantic RAG system"""
    print(" SIMPLE SEMANTIC RAG SYSTEM")
    
    # Initialize RAG system
    rag = SimpleSemanticRAG()
    
    # Test questions
    test_questions = [
        "What are Apple's main competitive risks?",
        "How does Apple generate revenue?",
        "What were Apple's research and development expenses?",
        "Tell me about Apple's supply chain risks",
        "What legal proceedings is Apple involved in?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Processing question...")
        
        result = rag.ask_question(question)
        
        if isinstance(result, dict):
            print(f" ANSWER: {result['answer']}")
            print(f"\n SOURCES:")
            for source in result['sources']:
                print(f"   - {source['chunk_id']} (score: {source['similarity']})")
                print(f"     {source['text_preview']}")
            print(f" Tokens used: {result['tokens_used']}")
        else:
            print(result)
        

if __name__ == "__main__":
    main()