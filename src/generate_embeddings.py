import openai
import json
import os
from pathlib import Path
from dotenv import load_dotenv  

def setup_openai():
    """Setup OpenAI API key"""
    load_dotenv()  
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(" ERROR: OpenAI API key not found!")
        print("Please set OPENAI_API_KEY in your .env file")
        return False
    
    print(" OpenAI API key loaded successfully")
    return True

def generate_embeddings():
    """Generate embeddings for all chunks"""
    
    if not setup_openai():
        return None
    
    chunks_dir = "../data/chunks"
    embeddings_dir = "../data/embeddings"
    Path(embeddings_dir).mkdir(parents=True, exist_ok=True)
    
    chunks_file = os.path.join(chunks_dir, "all_chunks.jsonl")
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f" Generating embeddings for {len(chunks)} chunks...")
    
    # Batch process embeddings
    batch_size = 500
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk['text'] for chunk in batch]
        
        print(f"   Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            
        except Exception as e:
            print(f" Error generating embeddings: {e}")
            return None
    
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = all_embeddings[i]
    
    embeddings_file = os.path.join(embeddings_dir, "embedded_chunks.jsonl")
    with open(embeddings_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    
    print(f" Saved embedded chunks to {embeddings_file}")
    
    total_tokens = sum(chunk['metadata']['tokens'] for chunk in chunks)
    cost = (total_tokens / 1000) * 0.00002
    print(f" Estimated cost: ${cost:.4f}")
    
    return chunks

if __name__ == "__main__":
    generate_embeddings()