from hybrid_rag import HybridRAG
import time

def compare_search_methods():
    """Compare semantic vs keyword vs hybrid search"""
    
    print("COMPARING SEARCH METHODS")
    
    hybrid = HybridRAG()
    
    # Test questions that highlight differences
    test_cases = [
        {
            "question": "R&D expenses 2023",
            "description": "Exact numbers with year - keyword should excel",
            "filters": {"year": 2023}
        },
        {
            "question": "competitive market pressures", 
            "description": "Conceptual query - semantic should excel",
            "filters": None
        },
        {
            "question": "single-source suppliers",
            "description": "Exact phrase - hybrid might combine both",
            "filters": None
        },
        {
            "question": "legal proceedings 2024",
            "description": "Specific section + year - filters help",
            "filters": {"year": 2024, "section": "legal"}
        }
    ]
    
    for test in test_cases:
        print(f"\n {test['description']}")
        print(f"Question: {test['question']}")
        if test['filters']:
            print(f"Filters: {test['filters']}")
        
        
        # Test all three methods
        for search_type in ["semantic", "keyword", "hybrid"]:
            start_time = time.time()
            
            result = hybrid.ask_question(
                question=test['question'],
                filters=test['filters'],
                search_type=search_type
            )
            
            search_time = time.time() - start_time
            
            if 'error' not in result:
                print(f"  {search_type.upper():8} | Time: {search_time:.2f}s | Sources: {len(result['sources'])}")
                
                # Show top source details
                if result['sources']:
                    top_source = result['sources'][0]
                    print(f"           | Top: {top_source['chunk_id']} ({top_source['search_type']})")
                    print(f"           | Score: {top_source['similarity']:.3f}")
            else:
                print(f"  {search_type.upper():8} | ERROR: {result['error']}")

def debug_keyword_search():
    """Debug why keyword search shows semantic results"""
    print("\n DEBUGGING KEYWORD SEARCH")
    
    from keyword_search import KeywordSearch
    
    keyword_searcher = KeywordSearch()
    
    # Test direct keyword search
    results = keyword_searcher.search("R&D expenses", top_k=3)
    
    print("Direct keyword search results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['metadata']['chunk_id']}")
        print(f"     Score: {result['score']:.3f}")
        print(f"     Text: {result['text'][:100]}...")

if __name__ == "__main__":
    compare_search_methods()
    debug_keyword_search()