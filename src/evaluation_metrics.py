import numpy as np
from typing import List, Dict, Set, Optional
from collections import Counter
import json

class RetrievalMetrics:
        
    def __init__(self):
        self.metrics_history = []
    
    
    def precision_at_k(self, retrieved_ids: List[str], relevant_ids: Set[str], k: int = 3) -> float:
        """
        Precision@K
        Measures: Of top K results, how many are relevant?
        Why important: Direct measure of retrieval accuracy for RAG (top results matter most)
        """
        if k <= 0 or not retrieved_ids:
            return 0.0
        
        top_k = retrieved_ids[:min(k, len(retrieved_ids))]
        relevant_count = sum(1 for chunk_id in top_k if chunk_id in relevant_ids)
        return relevant_count / len(top_k)
    
    def recall_at_k(self, retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
        """
        Recall@K  
        Measures: Of all relevant documents, how many did we find in top K?
        Why important: Measures coverage - did we miss important information?
        """
        if not relevant_ids:
            return 0.0
        
        top_k = retrieved_ids[:min(k, len(retrieved_ids))]
        relevant_found = sum(1 for chunk_id in top_k if chunk_id in relevant_ids)
        return relevant_found / len(relevant_ids)
    
    def mean_reciprocal_rank(self, retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """
        Mean Reciprocal Rank (MRR)
        Measures: How quickly do we find the first relevant result?
        Why important: Users want relevant answers quickly (first result matters)
        """
        for rank, chunk_id in enumerate(retrieved_ids, 1):
            if chunk_id in relevant_ids:
                return 1.0 / rank
        return 0.0
    
    
    def retrieval_coverage(self, retrieved_chunks: List[Dict]) -> Dict:
        """
        Retrieval Coverage
        Measures: How well do we cover different document sections?
        Why important: Good answers need diverse perspectives (not just one section)
        """
        if not retrieved_chunks:
            return {"sections_found": 0, "unique_sections": 0, "section_distribution": {}}
        
        # Count sections
        sections = []
        for chunk in retrieved_chunks:
            metadata = chunk.get('metadata', {})
            section = metadata.get('section', 'unknown')
            sections.append(section)
        
        section_counts = Counter(sections)
        
        return {
            "sections_found": len(sections),
            "unique_sections": len(section_counts),
            "section_diversity": len(section_counts) / len(sections) if sections else 0,
            "section_distribution": dict(section_counts)
        }
    
    def semantic_cohesion(self, retrieved_chunks: List[Dict]) -> float:
        """
        Semantic Cohesion
        Measures: How similar are retrieved chunks to each other?
        Why important: High cohesion = focused answer, Low cohesion = diverse but potentially irrelevant
        """
        if len(retrieved_chunks) < 2:
            return 0.0
        
        # Get similarity scores if available
        similarities = []
        for chunk in retrieved_chunks:
            if 'similarity' in chunk:
                similarities.append(chunk['similarity'])
        
        if similarities:
            return np.mean(similarities)
        
        # Fallback: Use search type consistency
        search_types = [chunk.get('search_type', 'unknown') for chunk in retrieved_chunks]
        if all(st == search_types[0] for st in search_types):
            return 0.8  # High cohesion if all same search type
        else:
            return 0.4  # Medium cohesion if mixed
    
    def chunk_quality_metrics(self, retrieved_chunks: List[Dict]) -> Dict:
        """
        Chunk Quality
        Measures: Are chunks usable for answer generation?
        Why important: Long/short chunks, similarity scores affect answer quality
        """
        if not retrieved_chunks:
            return {"avg_length": 0, "avg_similarity": 0, "usable_chunks": 0}
        
        lengths = []
        similarities = []
        
        for chunk in retrieved_chunks:
            # Text length in words
            text = chunk.get('text', '')
            lengths.append(len(text.split()))
            
            # Similarity score
            if 'similarity' in chunk:
                similarities.append(chunk['similarity'])
        
        # Determine usable chunks (not too short, decent similarity)
        usable_chunks = 0
        for chunk in retrieved_chunks:
            text = chunk.get('text', '')
            sim = chunk.get('similarity', 0)
            if len(text.split()) > 30 and sim > 0.3:
                usable_chunks += 1
        
        return {
            "avg_length": np.mean(lengths) if lengths else 0,
            "avg_similarity": np.mean(similarities) if similarities else 0,
            "usable_chunks": usable_chunks,
            "usable_percentage": (usable_chunks / len(retrieved_chunks)) * 100 if retrieved_chunks else 0
        }
    
    
    def evaluate_retrieval(self,
                          query: str,
                          retrieved_chunks: List[Dict],
                          ground_truth: Optional[Dict] = None) -> Dict:
        
        results = {
            "query": query,
            "retrieved_count": len(retrieved_chunks)
        }
        
        # Extract chunk IDs
        retrieved_ids = []
        for chunk in retrieved_chunks:
            chunk_id = chunk.get('metadata', {}).get('chunk_id', 'unknown')
            retrieved_ids.append(chunk_id)
        
        # 1. IR METRICS 
        if ground_truth and 'relevant_ids' in ground_truth:
            relevant_ids = set(ground_truth['relevant_ids'])
            
            results["ir_metrics"] = {
                "precision@3": self.precision_at_k(retrieved_ids, relevant_ids, 3),
                "recall@10": self.recall_at_k(retrieved_ids, relevant_ids, 10),
                "mrr": self.mean_reciprocal_rank(retrieved_ids, relevant_ids)
            }
        
        # 2. RAG METRICS 
        results["rag_metrics"] = {
            "coverage": self.retrieval_coverage(retrieved_chunks),
            "cohesion": self.semantic_cohesion(retrieved_chunks),
            "chunk_quality": self.chunk_quality_metrics(retrieved_chunks)
        }
        
        self.metrics_history.append(results)
        
        return results
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics from all evaluations"""
        if not self.metrics_history:
            return {}
        
        # Initialize accumulators
        precision_scores = []
        recall_scores = []
        mrr_scores = []
        cohesion_scores = []
        usable_percentages = []
        
        for result in self.metrics_history:
            # IR metrics
            if "ir_metrics" in result:
                ir = result["ir_metrics"]
                precision_scores.append(ir.get("precision@3", 0))
                recall_scores.append(ir.get("recall@10", 0))
                mrr_scores.append(ir.get("mrr", 0))
            
            # RAG metrics
            rag = result["rag_metrics"]
            cohesion_scores.append(rag.get("cohesion", 0))
            usable_percentages.append(rag["chunk_quality"].get("usable_percentage", 0))
        
        return {
            "total_evaluations": len(self.metrics_history),
            "average_precision@3": np.mean(precision_scores) if precision_scores else None,
            "average_recall@10": np.mean(recall_scores) if recall_scores else None,
            "average_mrr": np.mean(mrr_scores) if mrr_scores else None,
            "average_cohesion": np.mean(cohesion_scores) if cohesion_scores else 0,
            "average_usable_percentage": np.mean(usable_percentages) if usable_percentages else 0
        }
