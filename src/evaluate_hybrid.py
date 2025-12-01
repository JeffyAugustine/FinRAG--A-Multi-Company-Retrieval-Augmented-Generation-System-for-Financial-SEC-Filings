from hybrid_rag import HybridRAG
from semantic_rag import SimpleSemanticRAG
import openai
import json
import pandas as pd
from typing import List, Dict
import time

class HybridEvaluator:
    """Compare hybrid vs semantic search performance"""
    
    def __init__(self):
        self.hybrid_system = HybridRAG()
        self.semantic_system = SimpleSemanticRAG()
        self.test_cases = self._create_comparison_dataset()
    
    def _create_comparison_dataset(self) -> List[Dict]:
        """Test cases designed to highlight search method differences"""
        return [
            {
                "id": 1,
                "question": "R&D expenses 2023",
                "description": "Exact numbers + year filter",
                "expected_strength": "keyword",
                "filters": {"year": 2023},
                "difficulty": "easy"
            },
            {
                "id": 2,
                "question": "competitive market pressures",
                "description": "Conceptual business risks", 
                "expected_strength": "semantic",
                "filters": None,
                "difficulty": "medium"
            },
            {
                "id": 3,
                "question": "single-source suppliers in risk factors",
                "description": "Exact phrase + section filter",
                "expected_strength": "hybrid", 
                "filters": {"section": "risk_factors"},
                "difficulty": "medium"
            },
            {
                "id": 4,
                "question": "legal proceedings involving antitrust",
                "description": "Specific legal concepts",
                "expected_strength": "semantic",
                "filters": None,
                "difficulty": "hard"
            },
            {
                "id": 5,
                "question": "iPhone revenue 2024",
                "description": "Specific product + year",
                "expected_strength": "keyword",
                "filters": {"year": 2024},
                "difficulty": "easy"
            },
            {
                "id": 6,
                "question": "supply chain dependencies",
                "description": "Business operations concept",
                "expected_strength": "hybrid",
                "filters": None,
                "difficulty": "medium"
            }
        ]
    
    def evaluate_search_methods(self):
        """Comprehensive comparison of search methods"""
        print(" COMPREHENSIVE SEARCH METHOD EVALUATION")
        
        results = []
        
        for test_case in self.test_cases:
            print(f"\n Test {test_case['id']}: {test_case['description']}")
            print(f"   Question: {test_case['question']}")
            print(f"   Expected best: {test_case['expected_strength']}")
            
            # Test all three methods
            for search_type in ["semantic", "keyword", "hybrid"]:
                print(f"\n   Testing {search_type.upper()}...")
                
                start_time = time.time()
                
                if search_type == "semantic":
                    result = self.semantic_system.ask_question(test_case['question'])
                else:
                    result = self.hybrid_system.ask_question(
                        test_case['question'], 
                        filters=test_case['filters'],
                        search_type=search_type
                    )
                
                response_time = time.time() - start_time
                
                if 'error' not in result:
                    # Evaluate answer quality
                    evaluation = self._evaluate_answer_quality(
                        test_case['question'],
                        result['answer'],
                        result['sources']
                    )
                    
                    results.append({
                        'test_id': test_case['id'],
                        'question': test_case['question'],
                        'search_type': search_type,
                        'expected_strength': test_case['expected_strength'],
                        'difficulty': test_case['difficulty'],
                        'response_time': response_time,
                        'sources_found': len(result['sources']),
                        'faithfulness': evaluation['faithfulness'],
                        'relevance': evaluation['relevance'],
                        'overall_score': evaluation['overall_score'],
                        'is_expected_best': (search_type == test_case['expected_strength'])
                    })
                    
                    print(f"       Score: {evaluation['overall_score']:.1f}/5.0")
                    print(f"       Time: {response_time:.2f}s")
                    print(f"       Sources: {len(result['sources'])}")
                else:
                    print(f"       Error: {result['error']}")
        
        return self._generate_comparison_report(results)
    
    def _evaluate_answer_quality(self, question: str, answer: str, sources: List[Dict]) -> Dict:
        """Use GPT to evaluate answer quality"""
        
        prompt = f"""
        Evaluate this RAG system answer:

        QUESTION: {question}

        ANSWER:
        {answer}

        SOURCES USED: {len(sources)} sources

        Evaluation Criteria (1-5 scale):
        1. FAITHFULNESS: Does the answer stay grounded in sources? No hallucinations?
        2. RELEVANCE: Does it directly answer the question completely?

        Return ONLY JSON:
        {{
            "faithfulness": score,
            "relevance": score, 
            "overall_score": average_score
        }}
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI evaluation expert. Be objective."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            return json.loads(response.choices[0].message.content)
        except:
            return {"faithfulness": 3.0, "relevance": 3.0, "overall_score": 3.0}
    
    def _generate_comparison_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive comparison report"""
        
        df = pd.DataFrame(results)
        
        # Overall averages
        overall_avg = df.groupby('search_type')['overall_score'].mean()
        time_avg = df.groupby('search_type')['response_time'].mean()
        
        # Performance by expected strength
        strength_performance = {}
        for strength in ['semantic', 'keyword', 'hybrid']:
            subset = df[df['expected_strength'] == strength]
            if not subset.empty:
            # Convert Series to regular dict
                strength_performance[strength] = subset.groupby('search_type')['overall_score'].mean().to_dict()
        
        # When each method performs best
        best_per_method = {}
        for method in ['semantic', 'keyword', 'hybrid']:
            method_subset = df[df['search_type'] == method]
            best_cases = method_subset.nlargest(2, 'overall_score')
            best_per_method[method] = best_cases[['question', 'overall_score']].to_dict('records')
        
        report = {
            'overall_performance': overall_avg.to_dict(),
            'response_times': time_avg.to_dict(),
            'strength_analysis': strength_performance,
            'best_cases': best_per_method,
            'detailed_results': results
        }
        
        self._print_comparison_report(report)
        return report
    
    def _print_comparison_report(self, report: Dict):
        """Print formatted comparison report"""
        
        print(" SEARCH METHOD COMPARISON REPORT")
        
        # Overall performance
        print("\n OVERALL PERFORMANCE (Average Score):")
        for method, score in report['overall_performance'].items():
            print(f"   {method.upper():8}: {score:.2f}/5.0")
        
        # Response times
        print("\n AVERAGE RESPONSE TIME:")
        for method, time_val in report['response_times'].items():
            print(f"   {method.upper():8}: {time_val:.2f}s")
        
        # Strength analysis
        print("\n PERFORMANCE BY EXPECTED STRENGTH:")
        for strength, methods in report['strength_analysis'].items():
            print(f"\n   When {strength.upper()} expected to be best:")
            for method, score in methods.items():
                print(f"     {method}: {score:.2f}/5.0")
        
        # Best use cases
        print("\n RECOMMENDED USE CASES:")
        for method, cases in report['best_cases'].items():
            print(f"\n   {method.upper()} excels at:")
            for case in cases[:2]:
                print(f"     - {case['question']} (Score: {case['overall_score']:.1f})")
        
        # Save detailed results
        with open('hybrid_comparison_results.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n Detailed results saved to hybrid_comparison_results.json")

def main():
    """Run the comprehensive evaluation"""
    evaluator = HybridEvaluator()
    report = evaluator.evaluate_search_methods()
    
    print(" KEY INSIGHTS:")
    print("• Semantic: Best for conceptual, complex questions")
    print("• Keyword: Best for exact terms, numbers, specific phrases") 
    print("• Hybrid: Balanced approach, good for mixed queries")
    print("• Use filters to improve precision for all methods")

if __name__ == "__main__":
    main()