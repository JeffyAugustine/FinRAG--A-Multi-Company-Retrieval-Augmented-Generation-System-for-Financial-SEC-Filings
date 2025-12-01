import json
import re
from typing import List, Dict
import math
from collections import Counter
import string

class KeywordSearch:
    """Traditional keyword search using TF-IDF and exact matching"""
    
    def __init__(self, embeddings_path="../data/embeddings/embedded_chunks.jsonl"):
        self.chunks = self._load_chunks(embeddings_path)
        self.preprocessed_chunks = self._preprocess_chunks()
        self.vocabulary = self._build_vocabulary()
        self.idf = self._calculate_idf()
    
    def _load_chunks(self, path):
        """Load chunks without embeddings"""
        chunks = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk_data = json.loads(line)
                # Only keep text and metadata, not embeddings
                chunks.append({
                    'text': chunk_data['text'],
                    'metadata': chunk_data['metadata']
                })
        return chunks
    
    def _preprocess_text(self, text):
        """Basic text preprocessing"""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize
        tokens = text.split()
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        return tokens
    
    def _preprocess_chunks(self):
        """Preprocess all chunks"""
        return [self._preprocess_text(chunk['text']) for chunk in self.chunks]
    
    def _build_vocabulary(self):
        """Build vocabulary from all chunks"""
        vocabulary = set()
        for tokens in self.preprocessed_chunks:
            vocabulary.update(tokens)
        return list(vocabulary)
    
    def _calculate_idf(self):
        """Calculate Inverse Document Frequency"""
        N = len(self.chunks)
        idf = {}
        for word in self.vocabulary:
            # Count chunks containing this word
            doc_count = sum(1 for tokens in self.preprocessed_chunks if word in tokens)
            idf[word] = math.log((N + 1) / (doc_count + 1)) + 1
        return idf
    
    def _calculate_tf(self, tokens):
        """Calculate Term Frequency for a document"""
        tf = {}
        total_terms = len(tokens)
        for token in tokens:
            tf[token] = tokens.count(token) / total_terms
        return tf
    
    def search(self, query: str, top_k: int = 3, filters: Dict = None) -> List[Dict]:
        """Keyword search with optional filters"""
        
        # Preprocess query
        query_tokens = self._preprocess_text(query)
        
        # Calculate scores for each chunk
        scores = []
        for i, (chunk, tokens) in enumerate(zip(self.chunks, self.preprocessed_chunks)):
            
            # Apply filters
            if filters and not self._matches_filters(chunk['metadata'], filters):
                continue
            
            # Calculate TF-IDF score
            score = 0
            chunk_tf = self._calculate_tf(tokens)
            
            for word in query_tokens:
                if word in chunk_tf and word in self.idf:
                    score += chunk_tf[word] * self.idf[word]
            
            # Boost exact phrase matches
            exact_phrase_score = self._exact_phrase_match(chunk['text'], query)
            score += exact_phrase_score * 2  
            
            if score > 0:
                scores.append((score, chunk))
        
        # Sort by score and return top results
        scores.sort(key=lambda x: x[0], reverse=True)
        return [{"score": score, "source": "keyword", **chunk} for score, chunk in scores[:top_k]]
    
    def _exact_phrase_match(self, text: str, query: str) -> float:
        """Check for exact phrase matches"""
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Simple exact phrase matching
        if query_lower in text_lower:
            return 1.0
        
        # Check for significant word overlaps
        query_words = set(self._preprocess_text(query))
        text_words = set(self._preprocess_text(text))
        overlap = len(query_words.intersection(text_words)) / len(query_words)
        
        return overlap * 0.5  # Partial match score
    
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