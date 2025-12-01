import os
import tiktoken
from pathlib import Path
import json
import re

def calculate_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Calculate the number of tokens in a text."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except:
        return len(text) // 4

def smart_chunk_section(file_path: str, max_tokens: int = 512) -> list:
    """
    Chunk a section file with semantic boundaries and rich metadata
    """
    filename = Path(file_path).stem  
    parts = filename.split('_')
    
    if len(parts) >= 3:
        company = parts[0]  # "AAPL", "MSFT", or "TSLA"
        year = parts[1]     # "2023" or "2024"
        section = '_'.join(parts[2:])  # "risk_factors"
    else:
        # Fallback for unexpected formats
        print(f"     Unexpected filename format: {filename}")
        company = "UNKNOWN"
        year = "UNKNOWN"
        section = filename
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    
    # If content is too small, skip or handle differently
    if len(content.strip()) < 100:
        print(f"     Very small content in {filename}, skipping")
        return []
    
    chunks = []
    
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 50]
    
    # If no paragraphs found, split by sentences
    if not paragraphs:
        sentences = [s.strip() + '.' for s in content.split('.') if s.strip() and len(s.strip()) > 20]
        paragraphs = sentences
    
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
    # Last chunk
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
            "tokens": calculate_tokens(text),
            "source_file": f"{company}_{year}_{section}.txt"
        }
    }

def get_all_processed_files(processed_dir: str) -> list:
    """Get all processed section files with company/year/section info"""
    all_files = []
    
    pattern = r'^(AAPL|MSFT|TSLA)_(2023|2024)_([a-z_]+)\.txt$'
    
    for filename in os.listdir(processed_dir):
        match = re.match(pattern, filename)
        if match:
            company, year, section = match.groups()
            all_files.append({
                'filename': filename,
                'path': os.path.join(processed_dir, filename),
                'company': company,
                'year': year,
                'section': section
            })
    
    # Sort by company, then year, then section
    all_files.sort(key=lambda x: (x['company'], x['year'], x['section']))
    return all_files

def process_all_sections():
    """Process all section files and create chunked dataset"""
    processed_dir = "../data/processed"
    chunks_dir = "../data/chunks"
    
    Path(chunks_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all processed files
    processed_files = get_all_processed_files(processed_dir)
    
    if not processed_files:
        print(" No processed files found!")
        print(f"Expected files in: {processed_dir}")
        print("Format: {COMPANY}_{YEAR}_{SECTION}.txt")
        print("Example: AAPL_2023_business.txt, MSFT_2024_risk_factors.txt")
        return []
    
    all_chunks = []
    
    print(" PROCESSING ALL SECTION FILES:")
    print(f"Found {len(processed_files)} processed files")
    
    current_company = None
    current_year = None
    
    for file_info in processed_files:
        filename = file_info['filename']
        file_path = file_info['path']
        company = file_info['company']
        year = file_info['year']
        section = file_info['section']
        
        if company != current_company:
            print(f"\n {company}:")
            current_company = company
            current_year = None
        
        if year != current_year:
            print(f"   {year}:")
            current_year = year
        
        if os.path.exists(file_path):
            print(f"     Processing {section}...", end=" ")
            chunks = smart_chunk_section(file_path)
            all_chunks.extend(chunks)
        else:
            print(f"     Missing: {filename}")
    
    # Save all chunks to JSONL file
    chunks_file = os.path.join(chunks_dir, "multicompany_chunks.jsonl")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + '\n')
    
    # Also save a company-specific summary file
    summary_file = os.path.join(chunks_dir, "chunk_summary.json")
    summary = {
        "total_chunks": len(all_chunks),
        "by_company": {},
        "by_year": {},
        "by_section": {}
    }
    
    for chunk in all_chunks:
        meta = chunk['metadata']
        company = meta['company']
        year = meta['year']
        section = meta['section']
        
        # Count by company
        if company not in summary["by_company"]:
            summary["by_company"][company] = 0
        summary["by_company"][company] += 1
        
        # Count by year
        if year not in summary["by_year"]:
            summary["by_year"][year] = 0
        summary["by_year"][year] += 1
        
        # Count by section
        if section not in summary["by_section"]:
            summary["by_section"][section] = 0
        summary["by_section"][section] += 1
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f" SUCCESS: Saved {len(all_chunks)} total chunks")
    print(f" JSONL file: {chunks_file}")
    print(f" Summary file: {summary_file}")
    
    print("\n DETAILED CHUNK SUMMARY:")
    
    print("\n By Company:")
    for company in sorted(summary["by_company"].keys()):
        count = summary["by_company"][company]
        print(f"   {company}: {count} chunks")
    
    print("\n By Year:")
    for year in sorted(summary["by_year"].keys()):
        count = summary["by_year"][year]
        print(f"   {year}: {count} chunks")
    
    print("\n By Section:")
    for section in sorted(summary["by_section"].keys()):
        count = summary["by_section"][section]
        print(f"   {section}: {count} chunks")
    
    # Show sample chunks from each company
    print(f"\n SAMPLE CHUNKS (2 from each company):")
    
    companies_seen = set()
    samples_shown = 0
    
    for chunk in all_chunks:
        company = chunk['metadata']['company']
        if company not in companies_seen:
            companies_seen.add(company)
            print(f"\n🏢 {company}:")
            for i in range(2):
                if samples_shown < len(all_chunks):
                    sample = all_chunks[samples_shown]
                    print(f"  Chunk {i+1}:")
                    print(f"    ID: {sample['metadata']['chunk_id']}")
                    print(f"    Text: {sample['text'][:120]}...")
                    print(f"    Tokens: {sample['metadata']['tokens']}")
                    print()
                    samples_shown += 1
    
    return all_chunks

if __name__ == "__main__":
    process_all_sections()