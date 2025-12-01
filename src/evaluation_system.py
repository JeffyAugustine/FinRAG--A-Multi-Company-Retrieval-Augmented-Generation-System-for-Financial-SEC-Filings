import openai
import json
from semantic_rag import SimpleSemanticRAG
from typing import List, Dict
import pandas as pd

class RAGEvaluator:
    """Comprehensive evaluation framework for RAG system"""
    
    def __init__(self):
        self.rag_system = SimpleSemanticRAG()
        self.golden_dataset = self._create_golden_dataset()
    
    def _create_golden_dataset(self) -> List[Dict]:
        """Create golden dataset of verified Q&A pairs"""
        return [
            {
                "id": 1,
                "question": "What were Apple's main competitive risks mentioned in the 2023 10-K?",
                "ideal_answer": "Apple's main competitive risks included: 1) Intense competition from companies like Samsung and Google in smartphone and personal computer markets, 2) Aggressive price competition putting downward pressure on gross margins, 3) Rapid technological changes requiring continuous innovation, 4) Intellectual property infringement by competitors, 5) Minority market share in key product categories.",
                "expected_sections": ["risk_factors"],
                "expected_years": [2023],
                "difficulty": "medium"
            },
            {
                "id": 2, 
                "question": "What were Apple's R&D expenses in 2023 and 2024?",
                "ideal_answer": "In 2023, Apple's R&D expenses were $29,915 million. In 2024, R&D expenses increased to $31,370 million.",
                "expected_sections": ["financials", "mda"],
                "expected_years": [2023, 2024],
                "difficulty": "easy"
            },
            {
                "id": 3,
                "question": "What supply chain risks does Apple face?",
                "ideal_answer": "Apple faces several supply chain risks: 1) Dependence on single-source suppliers for key components, 2) Manufacturing concentration in Asia with outsourcing partners, 3) Potential disruptions from natural disasters, geopolitical issues, or trade disputes, 4) Reliance on custom components available from limited sources, 5) Global logistics and transit vulnerabilities.",
                "expected_sections": ["risk_factors", "business"],
                "expected_years": [2023, 2024],
                "difficulty": "medium"
            },
            {
                "id": 4,
                "question": "How does Apple generate revenue from services?",
                "ideal_answer": "Apple generates service revenue through: 1) Digital content subscriptions (Apple Music, Apple TV+), 2) App Store commissions from third-party apps, 3) AppleCare extended warranties, 4) iCloud storage services, 5) Advertising services. Service revenue was $96,169 million in 2024.",
                "expected_sections": ["business", "financials"],
                "expected_years": [2024],
                "difficulty": "medium"
            },
            {
                "id": 5,
                "question": "What legal proceedings involving antitrust is Apple currently facing?",
                "ideal_answer": "Apple faces several antitrust legal proceedings: 1) European Commission investigations under Digital Markets Act regarding App Store rules, 2) U.S. Department of Justice lawsuit alleging smartphone market monopolization, 3) Ongoing litigation with Epic Games over App Store policies, 4) Various state antitrust lawsuits in the U.S.",
                "expected_sections": ["legal"],
                "expected_years": [2024],
                "difficulty": "hard"
            }
        ]
    
    def evaluate_rag_system(self) -> Dict:
        """Comprehensive evaluation of RAG system"""
        print(" EVALUATING RAG SYSTEM PERFORMANCE")
        
        results = []
        
        for test_case in self.golden_dataset:
            print(f"\n Testing: {test_case['question']}")
            
            rag_result = self.rag_system.ask_question(test_case['question'])
            
            if isinstance(rag_result, dict):
                # Evaluate with LLM-as-judge
                evaluation = self._llm_judge_evaluation(
                    test_case['question'],
                    test_case['ideal_answer'], 
                    rag_result['answer'],
                    rag_result['sources']
                )
                
                results.append({
                    'test_id': test_case['id'],
                    'question': test_case['question'],
                    'difficulty': test_case['difficulty'],
                    'faithfulness': evaluation['faithfulness'],
                    'answer_relevance': evaluation['answer_relevance'],
                    'citation_accuracy': evaluation['citation_accuracy'],
                    'overall_score': evaluation['overall_score'],
                    'rag_answer': rag_result['answer'],
                    'sources_used': [s['chunk_id'] for s in rag_result['sources']]
                })
                
                print(f"    Faithfulness: {evaluation['faithfulness']}/5")
                print(f"    Relevance: {evaluation['answer_relevance']}/5")
                print(f"    Citations: {evaluation['citation_accuracy']}/5")
                print(f"    Overall: {evaluation['overall_score']}/5")
        
        return self._generate_evaluation_report(results)
    
    def _llm_judge_evaluation(self, question: str, ideal_answer: str, 
                            rag_answer: str, sources: List[Dict]) -> Dict:
        """Use GPT-4 as judge to evaluate answer quality"""
        
        prompt = f"""
        Evaluate the following RAG system answer:

        QUESTION: {question}

        IDEAL ANSWER (for reference):
        {ideal_answer}

        RAG SYSTEM ANSWER:
        {rag_answer}

        SOURCES USED:
        {json.dumps(sources, indent=2)}

        Please evaluate on these criteria (1-5 scale):
        1. FAITHFULNESS: Does the answer stay grounded in the provided sources? No hallucinations?
        2. ANSWER RELEVANCE: Does it directly and completely answer the question?
        3. CITATION ACCURACY: Are sources properly cited and relevant to claims?

        Return ONLY a JSON object with scores and brief explanation:
        {{
            "faithfulness": score,
            "answer_relevance": score, 
            "citation_accuracy": score,
            "overall_score": average_score,
            "explanation": "brief explanation"
        }}
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of AI systems. Be strict but fair."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
            
        except Exception as e:
            print(f" Evaluation error: {e}")
            return {"faithfulness": 3, "answer_relevance": 3, "citation_accuracy": 3, "overall_score": 3, "explanation": "Evaluation failed"}
    
    def _generate_evaluation_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive evaluation report"""
        
        df = pd.DataFrame(results)
        
        report = {
            'summary': {
                'total_tests': len(results),
                'average_faithfulness': df['faithfulness'].mean(),
                'average_relevance': df['answer_relevance'].mean(), 
                'average_citations': df['citation_accuracy'].mean(),
                'overall_score': df['overall_score'].mean()
            },
            'by_difficulty': {
                'easy': df[df['difficulty'] == 'easy']['overall_score'].mean(),
                'medium': df[df['difficulty'] == 'medium']['overall_score'].mean(),
                'hard': df[df['difficulty'] == 'hard']['overall_score'].mean()
            },
            'detailed_results': results
        }
        
        self._print_report(report)
        return report
    
    def _print_report(self, report: Dict):
        """Print formatted evaluation report"""
        print(" RAG SYSTEM EVALUATION REPORT")
        
        summary = report['summary']
        print(f"\n OVERALL SCORES:")
        print(f"   Faithfulness: {summary['average_faithfulness']:.2f}/5")
        print(f"   Answer Relevance: {summary['average_relevance']:.2f}/5")
        print(f"   Citation Accuracy: {summary['average_citations']:.2f}/5")
        print(f"    OVERALL SCORE: {summary['overall_score']:.2f}/5")
        
        print(f"\n BY DIFFICULTY:")
        for difficulty, score in report['by_difficulty'].items():
            if not pd.isna(score):
                print(f"   {difficulty.upper()}: {score:.2f}/5")
        
        print(f"\n TOTAL TESTS: {summary['total_tests']}")
        
        # Save detailed results
        with open('evaluation_results.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n Detailed results saved to evaluation_results.json")

def main():
    """Run comprehensive evaluation"""
    evaluator = RAGEvaluator()
    report = evaluator.evaluate_rag_system()
    
    # Interpretation
    print(" INTERPRETATION GUIDE:")
    print("4.0-5.0: Excellent - Production ready")
    print("3.0-3.9: Good - Minor improvements needed") 
    print("2.0-2.9: Fair - Significant improvements needed")
    print("1.0-1.9: Poor - Major issues")

if __name__ == "__main__":
    main()