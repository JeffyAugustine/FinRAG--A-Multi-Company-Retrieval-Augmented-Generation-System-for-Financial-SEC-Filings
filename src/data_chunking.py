import os
import tiktoken
from pathlib import Path
import json

def calculate_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Calculate the number of tokens in a text."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def smart_chunk_section(file_path: str, max_tokens: int = 512) -> list:
    """
    Chunk a section file with semantic boundaries and rich metadata
    """
    # Extract metadata from filename 
    filename = Path(file_path).stem  
    parts = filename.split('_')
    company = parts[0]  # "AAPL"
    year = parts[1]     # "2023"
    section = '_'.join(parts[2:])  # "risk_factors"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = []
    
    # Split by paragraphs first 
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 50]
    
    current_chunk = ""
    current_tokens = 0
    
    for paragraph in paragraphs:
        paragraph_tokens = calculate_tokens(paragraph)
        
        # If this single paragraph is too big, split it further
        if paragraph_tokens > max_tokens:
            # Split the big paragraph into sentences
            sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
            for sentence in sentences:
                sent_tokens = calculate_tokens(sentence)
                
                if current_tokens + sent_tokens > max_tokens and current_chunk:
                    chunks.append(create_chunk_dict(current_chunk, company, year, section, len(chunks)))
                    current_chunk = ""
                    current_tokens = 0
                
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sent_tokens
        else:
            # Normal paragraph processing
            if current_tokens + paragraph_tokens > max_tokens and current_chunk:
                chunks.append(create_chunk_dict(current_chunk, company, year, section, len(chunks)))
                current_chunk = ""
                current_tokens = 0
            
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
            current_tokens += paragraph_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(create_chunk_dict(current_chunk, company, year, section, len(chunks)))
    
    print(f"    {section}: {len(chunks)} chunks")
    return chunks

def create_chunk_dict(text: str, company: str, year: str, section: str, chunk_num: int) -> dict:
    """Create a standardized chunk dictionary with metadata"""
    return {
        "text": text.strip(),
        "metadata": {
            "company": company,
            "year": int(year),
            "section": section,
            "chunk_id": f"{company}_{year}_{section}_{chunk_num}",
            "tokens": calculate_tokens(text)
        }
    }

def process_all_sections():
    """Process all section files and create chunked dataset"""
    processed_dir = "../data/processed"
    chunks_dir = "../data/chunks"
    
    Path(chunks_dir).mkdir(parents=True, exist_ok=True)
    
    all_chunks = []
    
    print(" PROCESSING ALL SECTIONS:")
    
    
    # Process each section file in order
    sections_order = ['business', 'risk_factors', 'properties', 'legal', 'mda', 'financials']
    years = ['2023', '2024']
    
    for year in years:
        print(f"\n {year}:")
        for section in sections_order:
            filename = f"AAPL_{year}_{section}.txt"
            file_path = os.path.join(processed_dir, filename)
            
            if os.path.exists(file_path):
                print(f"    Processing {section}...")
                chunks = smart_chunk_section(file_path)
                all_chunks.extend(chunks)
            else:
                print(f"    Missing: {filename}")
    
    # Save all chunks to JSONL file
    chunks_file = os.path.join(chunks_dir, "all_chunks.jsonl")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + '\n')
    
    print(f"\n SUCCESS: Saved {len(all_chunks)} total chunks to {chunks_file}")
    
    # Print detailed summary
    print("\n DETAILED CHUNK SUMMARY:")
    
    year_summary = {}
    section_summary = {}
    
    for chunk in all_chunks:
        year = chunk['metadata']['year']
        section = chunk['metadata']['section']
        
        if year not in year_summary:
            year_summary[year] = 0
        year_summary[year] += 1
        
        if section not in section_summary:
            section_summary[section] = 0
        section_summary[section] += 1
    
    print("By Year:")
    for year in sorted(year_summary.keys()):
        print(f"   {year}: {year_summary[year]} chunks")
    
    print("\nBy Section:")
    for section in sorted(section_summary.keys()):
        print(f"   {section}: {section_summary[section]} chunks")
    
    # Show sample chunks
    print(f"\n SAMPLE CHUNKS (first 3):")
    for i, chunk in enumerate(all_chunks[:3]):
        print(f"\nSample {i+1}:")
        print(f"  ID: {chunk['metadata']['chunk_id']}")
        print(f"  Text: {chunk['text'][:150]}...")
        print(f"  Tokens: {chunk['metadata']['tokens']}")
    
    return all_chunks

if __name__ == "__main__":
    process_all_sections()