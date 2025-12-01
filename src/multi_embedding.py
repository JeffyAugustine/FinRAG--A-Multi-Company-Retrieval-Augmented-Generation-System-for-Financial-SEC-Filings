import openai
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import time

def setup_openai():
    """Setup OpenAI API key"""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OpenAI API key not found!")
        print("Please set OPENAI_API_KEY in your .env file")
        print("Example: OPENAI_API_KEY=sk-...")
        return False
    
    print(" OpenAI API key loaded successfully")
    return True

def load_chunks_with_retry():
    """Load chunks from JSONL file with retry logic"""
    chunks_dir = "../data/chunks"
    chunks_file = os.path.join(chunks_dir, "multicompany_chunks.jsonl")  
    
    
    if not os.path.exists(chunks_file):
        print(f" ERROR: Chunks file not found!")
        print(f"Looking for: {chunks_file}")
        print(f"Available files in {chunks_dir}:")
        if os.path.exists(chunks_dir):
            for f in os.listdir(chunks_dir):
                print(f"  - {f}")
        else:
            print(f"  Directory doesn't exist: {chunks_dir}")
        return None
    
    print(f" Loading chunks from: {chunks_file}")
    chunks = []
    
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    chunk = json.loads(line)
                    # Validate chunk structure
                    if 'text' not in chunk or 'metadata' not in chunk:
                        print(f"  Warning: Invalid chunk structure on line {line_num}")
                        continue
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    print(f"  Warning: JSON decode error on line {line_num}: {e}")
                    continue
        
        print(f" Loaded {len(chunks)} chunks")
        
        # Show chunk breakdown
        if chunks:
            company_counts = {}
            for chunk in chunks[:100]:  # Sample first 100 
                company = chunk['metadata'].get('company', 'UNKNOWN')
                company_counts[company] = company_counts.get(company, 0) + 1
            
            print(" Chunk breakdown by company:")
            for company, count in company_counts.items():
                print(f"   {company}: {count} chunks (sample)")
        
        return chunks
        
    except Exception as e:
        print(f" Error loading chunks: {e}")
        return None

def generate_embeddings():
    """Generate embeddings for multi-company chunks"""
    
    if not setup_openai():
        return None
    
    # Load chunks
    chunks = load_chunks_with_retry()
    if not chunks:
        return None
    
    embeddings_dir = "../data/embeddings"
    Path(embeddings_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"🔧 Generating embeddings for {len(chunks)} multi-company chunks...")
    print("=" * 60)
    
    # Batch processing with progress tracking
    batch_size = 500  # OpenAI's max for text-embedding-3-small
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch_num = (i // batch_size) + 1
        total_batches = (len(chunks) - 1) // batch_size + 1
        
        batch = chunks[i:i + batch_size]
        texts = [chunk['text'] for chunk in batch]
        
        print(f"   Batch {batch_num}/{total_batches}: {len(batch)} chunks")
        
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Show progress
            progress = min(i + batch_size, len(chunks))
            print(f"      Generated embeddings for {progress}/{len(chunks)} chunks")
            
            # Add small delay between batches to avoid rate limiting
            if i + batch_size < len(chunks):
                time.sleep(1)
            
        except openai.RateLimitError:
            print("       Rate limit hit, waiting 30 seconds...")
            time.sleep(30)
            continue
            
        except Exception as e:
            print(f" Error generating embeddings for batch {batch_num}: {e}")
            
            # Try smaller batch size if error persists
            if batch_size > 100:
                print(f"   Trying smaller batch size: {batch_size//2}")
                batch_size = batch_size // 2
                continue
            else:
                return None
    
    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = all_embeddings[i]
    
    # Save embedded chunks with multi-company name
    embeddings_file = os.path.join(embeddings_dir, "multicompany_embedded_chunks.jsonl")
    with open(embeddings_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    
    print(f" SAVED: {embeddings_file}")
    
    # Cost estimation
    total_tokens = sum(chunk['metadata'].get('tokens', 0) for chunk in chunks)
    cost_per_1k = 0.00002  # text-embedding-3-small cost
    cost = (total_tokens / 1000) * cost_per_1k
    
    print(f" Estimated cost: ${cost:.4f}")
    print(f" Total tokens: {total_tokens:,}")
    print(f" Companies processed: {len(set(c['metadata'].get('company', '') for c in chunks))}")
    
    summary_file = os.path.join(embeddings_dir, "multicompany_embeddings_summary.json")
    summary = {
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "estimated_cost": cost,
        "embedding_model": "text-embedding-3-small",
        "companies": list(set(c['metadata'].get('company', '') for c in chunks)),
        "years": list(set(c['metadata'].get('year', '') for c in chunks)),
        "sections": list(set(c['metadata'].get('section', '') for c in chunks))
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f" Summary saved: {summary_file}")
    
    if chunks:
        print(f"\n SAMPLE CHUNK WITH EMBEDDING:")
        sample = chunks[0]
        print(f"ID: {sample['metadata'].get('chunk_id', 'N/A')}")
        print(f"Company: {sample['metadata'].get('company', 'N/A')}")
        print(f"Year: {sample['metadata'].get('year', 'N/A')}")
        print(f"Section: {sample['metadata'].get('section', 'N/A')}")
        print(f"Text preview: {sample['text'][:100]}...")
        print(f"Embedding length: {len(sample['embedding'])} dimensions")
        print(f"First 5 embedding values: {sample['embedding'][:5]}")
    
    return chunks

if __name__ == "__main__":
    generate_embeddings()