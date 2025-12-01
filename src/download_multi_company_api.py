import os
import requests
import time
import random
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

def fetch_10k_filings(ticker, years_back=3):
    """
    Fetch 10-K filings for a given stock ticker using SEC EDGAR API.
    Returns: List of filing metadata
    """
    
    API_KEY = os.getenv('SEC_API_KEY')
    if not API_KEY:
        raise ValueError("SEC_API_KEY not found in .env file")
    
    base_url = "https://api.sec-api.io"
    
    from datetime import datetime
    current_year = datetime.now().year
    start_year = current_year - years_back
    
    print(f" Searching for {ticker} 10-K filings ({start_year}-{current_year})...")
    
    # SEC.gov headers
    sec_headers = {
        'User-Agent': 'University Research Project researcher@university.edu',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    
    session = requests.Session()
    session.headers.update(sec_headers)
    
    try:
        # Search for filings using sec-api.io
        query_url = f"{base_url}?token={API_KEY}"
        
        query = {
            "query": {
                "query_string": {
                    "query": f'ticker:{ticker} AND formType:"10-K" AND filedAt:[{start_year}-01-01 TO {current_year}-12-31]'
                }
            },
            "from": "0",
            "size": "10",
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        response = requests.post(query_url, json=query)
        response.raise_for_status()
        
        data = response.json()
        filings = data.get('filings', [])
        
        if not filings:
            print(f"    No 10-K filings found for {ticker}")
            return []
        
        print(f"    Found {len(filings)} 10-K filings")
        
        # Process each filing
        processed_filings = []
        for filing in filings:
            filing_url = filing.get('linkToTxt')
            filing_date = filing.get('filedAt', '')[:10]
            filing_year = filing_date[:4] if filing_date else "UNKNOWN"
            
            if filing_url and filing_year.isdigit():
                processed_filings.append({
                    'ticker': ticker,
                    'year': filing_year,
                    'url': filing_url,
                    'filedAt': filing_date
                })
        
        return processed_filings
        
    except Exception as e:
        print(f"    Search error for {ticker}: {e}")
        return []

def download_filing(filing_metadata, raw_dir, session):
    """Download a single filing with retry logic"""
    
    ticker = filing_metadata['ticker']
    year = filing_metadata['year']
    url = filing_metadata['url']
    
    filename = f"{ticker}_{year}_10K.txt"
    filepath = os.path.join(raw_dir, filename)
    
    # Check if already downloaded
    if os.path.exists(filepath):
        print(f"   ⏭  Already exists: {filename}")
        return True
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Random delay to SEC servers
            time.sleep(random.uniform(1.0, 3.0))
            
            response = session.get(url, timeout=30)
            
            if response.status_code == 200:
                # Save the file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                file_size = len(response.text)
                print(f"    Downloaded: {filename} ({file_size:,} chars)")
                return True
                
            elif response.status_code == 403:
                print(f"     Attempt {attempt + 1}: 403 Forbidden - waiting...")
                time.sleep(5)
                continue
                
            else:
                print(f"     Attempt {attempt + 1}: HTTP {response.status_code}")
                time.sleep(2)
                
        except requests.exceptions.Timeout:
            print(f"     Attempt {attempt + 1}: Timeout")
            continue
        except requests.exceptions.ConnectionError:
            print(f"     Attempt {attempt + 1}: Connection Error")
            time.sleep(3)
            continue
        except Exception as e:
            print(f"     Attempt {attempt + 1}: Error: {e}")
            continue
    
    print(f"    Failed to download {filename} after {max_retries} attempts")
    return False

def create_placeholder_filing(ticker, year, raw_dir):
    """Create a placeholder filing if download fails"""
    filename = f"{ticker}_{year}_10K.txt"
    filepath = os.path.join(raw_dir, filename)
    
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write(f"PLACEHOLDER: {ticker} {year} 10-K filing\n")
            f.write(f"SEC CIK numbers:\n")
            f.write(f"- Apple (AAPL): 0000320193\n")
            f.write(f"- Microsoft (MSFT): 0000789019\n")
            f.write(f"- Tesla (TSLA): 0001318605\n\n")
            f.write(f"To get real filing:\n")
            f.write(f"1. Go to https://www.sec.gov/edgar/search/\n")
            f.write(f"2. Search for company CIK\n")
            f.write(f"3. Download 10-K filing as .txt\n")
            f.write(f"4. Replace this file\n")
        
        print(f"    Created placeholder: {filename}")

def download_all_companies():
    """Download filings for all target companies"""
    
    companies = [
        ("AAPL", "Apple Inc."),
        ("MSFT", "Microsoft Corporation"),
        ("TSLA", "Tesla Inc.")
    ]
    
    raw_dir = "../data/raw"
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    
    print("🚀 STARTING MULTI-COMPANY 10-K DOWNLOAD")
    
    # Setup session
    sec_headers = {
        'User-Agent': 'University Research Project researcher@university.edu',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    
    session = requests.Session()
    session.headers.update(sec_headers)
    
    all_successful = []
    all_failed = []
    
    for ticker, company_name in companies:
        print(f"\n COMPANY: {company_name} ({ticker})")
        
        filings = fetch_10k_filings(ticker, years_back=2)
        
        if not filings:
            # Create placeholders for known years
            for year in [2023, 2024]:
                create_placeholder_filing(ticker, year, raw_dir)
            continue
        
        # Download each filing
        successful = 0
        for filing in filings:
            if download_filing(filing, raw_dir, session):
                successful += 1
                all_successful.append(f"{ticker}_{filing['year']}")
            else:
                all_failed.append(f"{ticker}_{filing['year']}")
                # Create placeholder for failed download
                create_placeholder_filing(ticker, filing['year'], raw_dir)
        
        print(f"    {successful}/{len(filings)} files downloaded")
    
    # Summary
    print(" DOWNLOAD SUMMARY")
    
    if all_successful:
        print(f" Successful downloads ({len(all_successful)}):")
        for item in all_successful:
            print(f"   - {item}")
    
    if all_failed:
        print(f"\n Failed/placeholder ({len(all_failed)}):")
        for item in all_failed:
            print(f"   - {item}")
    
    # Check existing files
    print(" CURRENT RAW FILES")
    
    existing_files = [f for f in os.listdir(raw_dir) if f.endswith('.txt')]
    if existing_files:
        for file in sorted(existing_files):
            filepath = os.path.join(raw_dir, file)
            size = os.path.getsize(filepath)
            print(f"   - {file} ({size:,} bytes)")
    else:
        print("   No files found in raw directory")

def check_api_key():
    """Check if SEC API key is configured"""
    API_KEY = os.getenv('SEC_API_KEY')
    
    if not API_KEY or API_KEY == "your_sec_api_key_here":
        print(" SEC_API_KEY not properly configured!")
        print("\nTo configure:")
        print("1. Get API key from https://sec-api.io")
        print("2. Add to .env file:")
        print("   SEC_API_KEY=your_actual_api_key_here")
        print("\nOr use manual download option above.")
        return False
    return True

if __name__ == "__main__":
    if check_api_key():
        download_all_companies()
