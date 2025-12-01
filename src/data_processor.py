import os
import re
from typing import List, Dict
from bs4 import BeautifulSoup

def extract_complete_10k_sections(text: str) -> Dict[str, str]:
    """
    Complete section extraction using multiple strategies
    """
    soup = BeautifulSoup(text, 'html.parser')
    sections = {}
    
    print("    Comprehensive section extraction...")
    
    header_based = extract_by_section_headers(soup)
    sections.update(header_based)
    
    if len(sections) < 4:  
        print("    Supplementing with content-based extraction...")
        content_based = extract_by_content_blocks(soup)
        
        for section_name, content in content_based.items():
            if section_name not in sections or len(content) > len(sections[section_name]):
                sections[section_name] = content
    
    return sections

def extract_by_section_headers(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract sections by finding headers and content between them"""
    sections = {}
    
    # Define section patterns and their order
    section_patterns = [
        ('business', [r'ITEM\s*1\s*[^A].*?BUSINESS', r'^BUSINESS$']),
        ('risk_factors', [r'ITEM\s*1A.*?RISK\s*FACTORS', r'^RISK\s*FACTORS$']),
        ('properties', [r'ITEM\s*2.*?PROPERTIES', r'^PROPERTIES$']),
        ('legal', [r'ITEM\s*3.*?LEGAL\s*PROCEEDINGS', r'^LEGAL\s*PROCEEDINGS$']),
        ('mda', [r'ITEM\s*7.*?MANAGEMENT.*?DISCUSSION', r'^MANAGEMENT.*?DISCUSSION']),
        ('financials', [r'ITEM\s*8.*?FINANCIAL\s*STATEMENTS', r'^FINANCIAL\s*STATEMENTS'])
    ]
    
    header_elements = []
    for element in soup.find_all(['div', 'span', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'td', 'th']):
        text = element.get_text(strip=True)
        if text and len(text) < 150:  
            header_elements.append((element, text))
    
    # Find and sort section headers
    found_headers = []
    for element, text in header_elements:
        for section_name, patterns in section_patterns:
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found_headers.append((section_name, element, text))
                    break
    
    # Sort by document position
    found_headers.sort(key=lambda x: get_element_position(x[1]))
    
    print(f"   Found {len(found_headers)} section headers")
    
    # Extract content between headers
    for i, (section_name, header_element, header_text) in enumerate(found_headers):
        print(f"   📖 Extracting {section_name}...")
        
        # Determine end point
        if i + 1 < len(found_headers):
            end_element = found_headers[i + 1][1]
            content = extract_content_between_elements(header_element, end_element)
        else:
            content = extract_content_after_element(header_element)
        
        if content and len(content) > 1000:
            sections[section_name] = content
            print(f"       {section_name}: {len(content):,} chars")
    
    return sections

def get_element_position(element):
    """Get approximate position of element in document"""
    try:
        return element.sourceline
    except:
        # Fallback: use string representation position
        return str(element).find('>')

def extract_content_between_elements(start_element, end_element) -> str:
    """Extract all content between two elements"""
    content_parts = []
    current = start_element
    
    header_text = start_element.get_text(strip=True)
    if header_text:
        content_parts.append(header_text)
    
    current = get_next_element(current)
    
    while current and current != end_element:
        if hasattr(current, 'get_text'):
            text = current.get_text(strip=True)
            if text and len(text) > 20:  
                content_parts.append(text)
        
        current = get_next_element(current)
        if not current:
            break
    
    return clean_content('\n\n'.join(content_parts))

def extract_content_after_element(start_element) -> str:
    """Extract all content after an element"""
    content_parts = []
    current = start_element
    
    # Include starting element
    start_text = start_element.get_text(strip=True)
    if start_text:
        content_parts.append(start_text)
    
    # Safety counter
    max_elements = 10000
    count = 0
    
    current = get_next_element(current)
    while current and count < max_elements:
        count += 1
        
        if hasattr(current, 'get_text'):
            text = current.get_text(strip=True)
            if text and len(text) > 20:
                content_parts.append(text)
        
        current = get_next_element(current)
        if not current:
            break
    
    return clean_content('\n\n'.join(content_parts))

def get_next_element(element):
    """Get the next element in document order"""
    # First try next sibling
    next_elem = element.next_sibling
    while next_elem and (not hasattr(next_elem, 'name') or next_elem.name in ['script', 'style', 'meta']):
        next_elem = next_elem.next_sibling
    
    if next_elem:
        return next_elem
    
    # If no next sibling, try parent's next sibling
    parent = element.parent
    if parent:
        parent_next = parent.next_sibling
        while parent_next and (not hasattr(parent_next, 'name') or parent_next.name in ['script', 'style', 'meta']):
            parent_next = parent_next.next_sibling
        return parent_next
    
    return None

def extract_by_content_blocks(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract and categorize large content blocks"""
    sections = {}
    
    print("    Finding large content blocks...")
    
    # Find all substantial text blocks
    large_blocks = []
    for element in soup.find_all(['div', 'p', 'table']):
        text = element.get_text(strip=True)
        if len(text) > 2000:  
            large_blocks.append(text)
    
    print(f"   Found {len(large_blocks)} large text blocks")
    
    # Categorize blocks
    categorized = categorize_content_blocks(large_blocks)
    
    for section_name, blocks in categorized.items():
        if blocks:
            content = '\n\n'.join(blocks)
            if len(content) > 5000:
                sections[section_name] = content
                print(f"      📦 {section_name}: {len(content):,} chars")
    
    return sections

def categorize_content_blocks(blocks: List[str]) -> Dict[str, List[str]]:
    """Categorize content blocks into sections"""
    categorized = {
        'business': [],
        'risk_factors': [],
        'mda': [],
        'financials': [],
        'properties': [],
        'legal': []
    }
    
    section_keywords = {
        'business': [
            'company was founded', 'products and services', 'business segments',
            'geographic information', 'company overview', 'operating segments',
            'revenue recognition', 'customer segments'
        ],
        'risk_factors': [
            'risk factors', 'could adversely affect', 'may negatively impact',
            'uncertainties include', 'potential risks', 'adverse effect',
            'competitive pressures', 'market risks'
        ],
        'mda': [
            'management discussion', 'results of operations', 'financial condition',
            'liquidity and capital', 'critical accounting', 'outlook',
            'quarterly results', 'gross margin'
        ],
        'financials': [
            'consolidated balance', 'consolidated statements', 'cash flows',
            'notes to financial', 'accounting policies', 'financial statement',
            'balance sheet', 'income statement'
        ],
        'properties': [
            'properties', 'facilities', 'manufacturing', 'retail stores',
            'corporate facilities', 'leased properties'
        ],
        'legal': [
            'legal proceedings', 'litigation', 'claims', 'regulatory',
            'contingencies', 'legal matters'
        ]
    }
    
    for block in blocks:
        block_lower = block.lower()
        best_section = None
        best_score = 0
        
        for section_name, keywords in section_keywords.items():
            score = sum(1 for keyword in keywords if keyword in block_lower)
            if score > best_score:
                best_score = score
                best_section = section_name
        
        if best_section and best_score >= 2:
            categorized[best_section].append(block)
    
    return categorized

def clean_content(text: str) -> str:
    """Clean extracted content"""
    if not text:
        return ""
    
    # Remove HTML tags but keep text
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove common artifacts
    text = re.sub(r'&[a-z]+;', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'&#\d+;', ' ', text)
    
    return text.strip()

def process_10k_files():
    """Process all raw 10-K files with improved extraction"""
    
    raw_dir = "../data/raw"
    processed_dir = "../data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    print("PROCESSING 10-K FILES (IMPROVED EXTRACTION)")
    
    for filename in os.listdir(raw_dir):
        if filename.endswith('.txt') and 'AAPL' in filename:
            filepath = os.path.join(raw_dir, filename)
            year = filename.split('_')[-1].split('.')[0]
            
            print(f"\n PROCESSING: {filename}")
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
            
            # Use improved extraction
            sections = extract_complete_10k_sections(raw_text)
            
            # Save sections
            section_count = 0
            total_chars = 0
            
            for section_name, content in sections.items():
                section_filename = f"AAPL_{year}_{section_name}.txt"
                section_path = os.path.join(processed_dir, section_filename)
                
                with open(section_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f" {section_name}: {len(content):,} characters")
                section_count += 1
                total_chars += len(content)
            
            if section_count == 0:
                print(" No sections extracted")
            else:
                print(f" Saved {section_count} sections ({total_chars:,} total chars) from {year}")

if __name__ == "__main__":
    process_10k_files()