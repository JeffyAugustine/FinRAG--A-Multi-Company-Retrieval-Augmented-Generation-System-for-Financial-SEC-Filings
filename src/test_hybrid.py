from hybrid_rag import HybridRAG

def test_hybrid_system():
    """Test the hybrid search system"""
    
    print(" TESTING HYBRID RAG SYSTEM")
    
    hybrid = HybridRAG()
    
    test_cases = [
        {
            "question": "R&D expenses",
            "search_type": "keyword",
            "description": "Keyword search should show keyword sources"
        },
        {
            "question": "competitive risks",
            "search_type": "semantic", 
            "description": "Semantic search should show semantic sources"
        },
        {
            "question": "supply chain",
            "search_type": "hybrid",
            "description": "Hybrid should show hybrid sources"
        }
    ]
    
    for test in test_cases:
        print(f"\n {test['description']}")
        print(f"Question: {test['question']}")
        print(f"Search type: {test['search_type']}")
        
        result = hybrid.ask_question(
            question=test['question'],
            search_type=test['search_type']
        )
        
        if 'error' not in result:
            print(f" SUCCESS!")
            print(f"Answer preview: {result['answer'][:100]}...")
            print(f"Sources found: {len(result['sources'])}")
            for source in result['sources']:
                print(f"   - {source['chunk_id']} ({source['search_type']}) - Score: {source['similarity']:.3f}")
                if source['search_type'] == 'keyword':
                    print(f"     Keyword score: {source['keyword_score']:.3f}")
                elif source['search_type'] == 'semantic':
                    print(f"     Semantic score: {source['semantic_score']:.3f}")
                else:  # hybrid
                    print(f"     Semantic: {source['semantic_score']:.3f}, Keyword: {source['keyword_score']:.3f}")
        else:
            print(f" Error: {result['error']}")

if __name__ == "__main__":
    test_hybrid_system()