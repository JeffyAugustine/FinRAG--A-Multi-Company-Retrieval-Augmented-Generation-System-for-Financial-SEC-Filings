import json
import sys
import os
from typing import List, Dict, Tuple
import time

class RAGEvaluationRunner:
    """Run evaluation """
    
    def __init__(self):
        # Load ground truth
        self.ground_truth = self._load_ground_truth()
        
        # Test queries 
        self.test_queries = [
            "What were Apple's main competitive risks mentioned in the 2023 10-K?",
            "What were Apple's R&D expenses in 2023 and 2024?",
            "What supply chain risks does Apple face?"
        ]
        
        print(f"🔍 RAG Evaluation - {len(self.test_queries)} queries")
    
    def _load_ground_truth(self) -> Dict:
        """Load the ground truth file"""
        try:
            with open('ground_truth.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("ERROR: ground_truth.json not found. Run auto_ground_truth.py first.")
            sys.exit(1)
    
    def _get_search_results(self, query: str, method: str) -> Tuple[List[Dict], float]:
        
        start_time = time.time()
        
        try:
            if method == "semantic":
                from semantic_rag import SimpleSemanticRAG
                rag = SimpleSemanticRAG()
                
                # Use semantic_search directly 
                results = rag.semantic_search(query, top_k=5)
                
            elif method == "hybrid":
                from hybrid_rag import HybridRAG
                rag = HybridRAG()
                
                # Use search_with_type directly 
                results = rag.search_with_type(query, top_k=5, search_type="hybrid")
                
            elif method == "keyword":
                from hybrid_rag import HybridRAG
                rag = HybridRAG()
                
                # Use search_with_type for keyword
                results = rag.search_with_type(query, top_k=5, search_type="keyword")
            
            else:
                results = []
            
            search_time = time.time() - start_time
            return results, search_time
            
        except Exception as e:
            print(f"  ⚠️  Error in {method} search: {e}")
            return [], 0
    
    def evaluate_method(self, method: str):
        """Evaluate a single search method"""
        print(f"\n📊 Evaluating {method.upper()} search...")
        
        results = []
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"  {i}. Query: {query[:50]}...")
            
            # Get search results 
            search_results, search_time = self._get_search_results(query, method)
            
            if not search_results:
                print(f"     No results")
                continue
            
            # Get ground truth for this query
            gt = self.ground_truth.get(query, {"relevant_ids": [], "relevance_scores": {}})
            
            # Calculate metrics locally 
            metrics = self._calculate_metrics_locally(query, search_results, gt)
            
            # Store results
            result = {
                "query": query,
                "search_method": method,
                "search_time": search_time,
                "retrieved_count": len(search_results),
                "metrics": metrics,
                "retrieved_ids": [r.get('metadata', {}).get('chunk_id', 'unknown') 
                                 for r in search_results[:5]]  # Store top 5 IDs
            }
            
            results.append(result)
            
            # Print quick stats
            print(f"     Retrieved: {len(search_results)} chunks")
            print(f"    ⏱  Time: {search_time:.2f}s")
            if metrics.get("ir_metrics"):
                print(f"     P@3: {metrics['ir_metrics'].get('precision@3', 0):.3f}")
        
        return results
    
    def _calculate_metrics_locally(self, query: str, retrieved_chunks: List[Dict], 
                                 ground_truth: Dict) -> Dict:
        
        metrics = {
            "rag_metrics": {},
            "ir_metrics": {}
        }
        
        # Extract chunk IDs
        retrieved_ids = []
        for chunk in retrieved_chunks:
            chunk_id = chunk.get('metadata', {}).get('chunk_id', 'unknown')
            retrieved_ids.append(chunk_id)
        
        # 1. IR METRICS 
        if ground_truth and "relevant_ids" in ground_truth:
            relevant_ids = set(ground_truth["relevant_ids"])
            
            # Precision@3
            top_3 = retrieved_ids[:3]
            precision = sum(1 for cid in top_3 if cid in relevant_ids) / 3 if top_3 else 0
            
            # Recall@10
            top_10 = retrieved_ids[:10]
            recall = sum(1 for cid in top_10 if cid in relevant_ids) / len(relevant_ids) if relevant_ids else 0
            
            # MRR
            mrr = 0
            for rank, cid in enumerate(retrieved_ids, 1):
                if cid in relevant_ids:
                    mrr = 1.0 / rank
                    break
            
            metrics["ir_metrics"] = {
                "precision@3": precision,
                "recall@10": recall,
                "mrr": mrr
            }
        
        # 2. RAG METRICS 
        # Section coverage
        sections = []
        similarities = []
        lengths = []
        
        for chunk in retrieved_chunks:
            metadata = chunk.get('metadata', {})
            sections.append(metadata.get('section', 'unknown'))
            
            if 'similarity' in chunk:
                similarities.append(chunk['similarity'])
            
            text = chunk.get('text', '')
            lengths.append(len(text.split()))
        
        # Unique sections
        unique_sections = len(set(sections))
        
        # Chunk quality
        usable_chunks = 0
        for chunk in retrieved_chunks:
            text = chunk.get('text', '')
            sim = chunk.get('similarity', 0)
            if len(text.split()) > 30 and sim > 0.3:
                usable_chunks += 1
        
        metrics["rag_metrics"] = {
            "coverage": {
                "unique_sections": unique_sections,
                "total_sections": len(sections),
                "section_diversity": unique_sections / len(sections) if sections else 0
            },
            "cohesion": np.mean(similarities) if similarities else 0,
            "chunk_quality": {
                "usable_chunks": usable_chunks,
                "usable_percentage": (usable_chunks / len(retrieved_chunks)) * 100 if retrieved_chunks else 0,
                "avg_length": np.mean(lengths) if lengths else 0,
                "avg_similarity": np.mean(similarities) if similarities else 0
            }
        }
        
        return metrics
    
    def run_evaluation(self):
        """Run complete evaluation """
        print(" RUNNING RAG EVALUATION")
        
        global np
        import numpy as np
        
        all_results = {}
        
        # Evaluate each method
        methods = ["semantic", "hybrid", "keyword"]
        
        for method in methods:
            results = self.evaluate_method(method)
            all_results[method] = results
        
        # Generate comparison
        comparison = self._generate_comparison(all_results)
        
        # Save results
        self._save_results(all_results, comparison)
        
        print(" EVALUATION COMPLETE!")
        
        self._print_key_insights(comparison)
        
        return all_results, comparison
    
    def _generate_comparison(self, all_results: Dict) -> Dict:
        """Generate comparison across methods"""
        comparison = {
            "methods": ["semantic", "hybrid", "keyword"],
            "averages": {},
            "cost_saving": "No answer generation"
        }
        
        for method in comparison["methods"]:
            results = all_results.get(method, [])
            
            if not results:
                continue
            
            # Initialize accumulators
            precision_scores = []
            recall_scores = []
            mrr_scores = []
            search_times = []
            unique_sections = []
            
            for result in results:
                metrics = result.get("metrics", {})
                
                # IR metrics
                if "ir_metrics" in metrics:
                    ir = metrics["ir_metrics"]
                    precision_scores.append(ir.get("precision@3", 0))
                    recall_scores.append(ir.get("recall@10", 0))
                    mrr_scores.append(ir.get("mrr", 0))
                
                # RAG metrics
                if "rag_metrics" in metrics:
                    rag = metrics["rag_metrics"]
                    unique_sections.append(rag["coverage"].get("unique_sections", 0))
                
                # Timing
                search_times.append(result.get("search_time", 0))
            
            # Calculate averages
            comparison["averages"][method] = {
                "precision@3": self._safe_mean(precision_scores),
                "recall@10": self._safe_mean(recall_scores),
                "mrr": self._safe_mean(mrr_scores),
                "avg_search_time": self._safe_mean(search_times),
                "avg_unique_sections": self._safe_mean(unique_sections),
                "queries_evaluated": len(results)
            }
        
        return comparison
    
    def _safe_mean(self, values: List[float]) -> float:
        """Safe mean calculation"""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def _print_key_insights(self, comparison: Dict):
        """Print key insights from evaluation"""
        print("\n KEY INSIGHTS:")
        
        if "averages" not in comparison:
            return
        
        avgs = comparison["averages"]
        
        # Find best method for each metric
        metrics_to_check = ["precision@3", "recall@10", "mrr", "avg_search_time"]
        
        for metric in metrics_to_check:
            best_method = None
            best_value = None
            
            for method, values in avgs.items():
                value = values.get(metric)
                if value is not None:
                    if best_value is None:
                        best_value = value
                        best_method = method
                    elif metric == "avg_search_time":  # Lower is better for time
                        if value < best_value:
                            best_value = value
                            best_method = method
                    else:  # Higher is better for other metrics
                        if value > best_value:
                            best_value = value
                            best_method = method
            
            if best_method:
                if metric == "avg_search_time":
                    print(f"  • Fastest: {best_method.upper()} ({best_value:.2f}s)")
                elif metric == "precision@3":
                    print(f"  • Most Accurate: {best_method.upper()} ({best_value:.3f})")
                elif metric == "recall@10":
                    print(f"  • Best Coverage: {best_method.upper()} ({best_value:.3f})")
                elif metric == "mrr":
                    print(f"  • Quickest Relevant: {best_method.upper()} ({best_value:.3f})")
        
        print("\n RECOMMENDATIONS:")
        
        # Overall recommendation based on precision
        best_precision_method = max(avgs.items(), 
                                  key=lambda x: x[1].get("precision@3", 0))
        print(f"  • Use {best_precision_method[0].upper()} for accuracy")
        
        # Fastest for quick searches
        fastest_method = min(avgs.items(), 
                           key=lambda x: x[1].get("avg_search_time", 100))
        print(f"  • Use {fastest_method[0].upper()} for speed")
    
    def _save_results(self, all_results: Dict, comparison: Dict):
        """Save evaluation results with both IR and RAG metrics"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    
        # Save JSON results
        output_file = f"evaluation_results_{timestamp}.json"
    
        data_to_save = {
            "timestamp": timestamp,
            "test_queries": self.test_queries,
            "ground_truth_available": list(self.ground_truth.keys()),
            "results": all_results,
            "comparison": comparison,
            "notes": "Low-cost evaluation - no answer generation"
        }
    
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)
    
        # Save COMPREHENSIVE summary with IR + RAG metrics
        summary_file = f"evaluation_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("RAG EVALUATION SUMMARY\n")
            f.write(f"Date: {timestamp}\n")
            f.write(f"Queries: {len(self.test_queries)}\n\n")
        
            for method in ["semantic", "hybrid", "keyword"]:
                if method in all_results:
                    method_results = all_results[method]
                
                 # Calculate averages for this method
                    precision_scores = []
                    recall_scores = []
                    mrr_scores = []
                    times = []
                
                    # RAG METRICS accumulators
                    cohesion_scores = []
                    usable_percentages = []
                    unique_sections_list = []
                    avg_lengths = []
                    avg_similarities = []
                
                    for result in method_results:
                        metrics = result.get("metrics", {})
                    
                        # IR metrics
                        if "ir_metrics" in metrics:
                            ir = metrics["ir_metrics"]
                            precision_scores.append(ir.get("precision@3", 0))
                            recall_scores.append(ir.get("recall@10", 0))
                            mrr_scores.append(ir.get("mrr", 0))
                    
                        # RAG metrics
                        if "rag_metrics" in metrics:
                            rag = metrics["rag_metrics"]
                            cohesion_scores.append(rag.get("cohesion", 0))
                        
                            quality = rag.get("chunk_quality", {})
                            usable_percentages.append(quality.get("usable_percentage", 0))
                            avg_lengths.append(quality.get("avg_length", 0))
                            avg_similarities.append(quality.get("avg_similarity", 0))
                        
                            coverage = rag.get("coverage", {})
                            unique_sections_list.append(coverage.get("unique_sections", 0))
                    
                        times.append(result.get("search_time", 0))
                
                    # Write method summary
                    f.write(f"\n{method.upper()} SEARCH:\n")
                
                    # IR Metrics
                    f.write("IR METRICS:\n")
                    if precision_scores:
                        f.write(f"  • Precision@3: {np.mean(precision_scores):.3f}\n")
                    if recall_scores:
                        f.write(f"  • Recall@10:   {np.mean(recall_scores):.3f}\n")
                    if mrr_scores:
                        f.write(f"  • MRR:         {np.mean(mrr_scores):.3f}\n")
                    if times:
                        f.write(f"  • Avg Time:    {np.mean(times):.2f}s\n")
                
                    # RAG Metrics
                    f.write("\nRAG METRICS:\n")
                    if cohesion_scores:
                        f.write(f"  • Semantic Cohesion: {np.mean(cohesion_scores):.3f}\n")
                    if usable_percentages:
                        f.write(f"  • Usable Chunks:     {np.mean(usable_percentages):.1f}%\n")
                    if unique_sections_list:
                        f.write(f"  • Unique Sections:   {np.mean(unique_sections_list):.1f}\n")
                    if avg_lengths:
                        f.write(f"  • Avg Chunk Length:  {np.mean(avg_lengths):.0f} words\n")
                    if avg_similarities:
                        f.write(f"  • Avg Similarity:    {np.mean(avg_similarities):.3f}\n")
                
                    f.write(f"  • Queries Evaluated: {len(method_results)}\n")
    
        print(f"\n Results saved:")
        print(f"   • {output_file}")
        print(f"   • {summary_file}")

if __name__ == "__main__":
    print(" RAG EVALUATION")
    
    try:
        evaluator = RAGEvaluationRunner()
        results, comparison = evaluator.run_evaluation()
        
        print("\n EVALUATION COMPLETE WITHOUT ANSWER GENERATION")
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        print("\nMake sure:")
        print("1. ground_truth.json exists (run auto_ground_truth.py)")
        print("2. Your RAG systems are in the same directory")
        print("3. OpenAI API key is set in .env")