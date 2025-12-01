import os
import requests
import time
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def fetch_10k_filings(ticker, years_back=3):
    """
    Fetch 10-K filings for a given stock ticker using SEC EDGAR direct access.
    """
    
    # Get API key from environment variables 
    API_KEY = os.getenv('SEC_API_KEY')
    if not API_KEY:
        raise ValueError("SEC_API_KEY not found in .env file")
    
    base_url = "https://api.sec-api.io"
    
    from datetime import datetime
    current_year = datetime.now().year
    start_year = current_year - years_back
    
    print(f"Searching for 10-K filings for {ticker} from {start_year} to {current_year}...")
    
    sec_headers = {
        'User-Agent': 'Company Name myname@company.com',  
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    
    # Create a session to maintain headers and cookies
    session = requests.Session()
    session.headers.update(sec_headers)
    
    try:
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
            print(f"No 10-K filings found for {ticker}")
            return
        
        print(f"Found {len(filings)} 10-K filings")
        
        raw_dir = "../data/raw"
        os.makedirs(raw_dir, exist_ok=True)
        
        successful_downloads = 0
        for filing in filings:
            try:
                filing_url = filing.get('linkToTxt')
                filing_date = filing.get('filedAt', '')[:10]
                filing_year = filing_date[:4]
                
                if not filing_url:
                    print(f"No text URL found for {filing_date} filing")
                    continue
                
                print(f"Downloading {filing_year} 10-K...")
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Add random delay -server requirement
                        time.sleep(random.uniform(1.0, 3.0))
                        
                        filing_response = session.get(filing_url, timeout=30)
                        
                        if filing_response.status_code == 200:
                            filename = f"{raw_dir}/{ticker}_10K_{filing_year}.txt"
                            with open(filename, 'w', encoding='utf-8') as f:
                                f.write(filing_response.text)
                            
                            file_size = len(filing_response.text)
                            print(f" Saved: {filename} ({file_size:,} characters)")
                            successful_downloads += 1
                            break
                            
                        elif filing_response.status_code == 403:
                            print(f"  Attempt {attempt + 1}: 403 Forbidden - waiting longer...")
                            time.sleep(5)  
                            continue
                            
                        else:
                            print(f"  Attempt {attempt + 1}: HTTP {filing_response.status_code}")
                            time.sleep(2)
                            
                    except requests.exceptions.Timeout:
                        print(f"  Attempt {attempt + 1}: Timeout")
                        continue
                    except requests.exceptions.ConnectionError:
                        print(f"  Attempt {attempt + 1}: Connection Error")
                        time.sleep(3)
                        continue
                
                else:
                    # All retries failed
                    print(f" Failed to download {filing_year} filing after {max_retries} attempts")
                
            except Exception as e:
                print(f"Error with {filing_year} filing: {e}")
                continue
            
        print(f"\nDownload summary: {successful_downloads}/{len(filings)} files successfully saved to {raw_dir}/")
        
        if successful_downloads == 0:
            print("\n" + "="*60)
            print("ALTERNATIVE MANUAL OPTION:")
            print("Since automated download is being blocked, you can:")
            print("1. Manually download 10-K filings from:")
            print("   https://www.sec.gov/edgar/searchedgar/companysearch")
            print("2. Search for 'Apple Inc.' (CIK: 0000320193)")
            print("3. Download the '10-K' filings as text files")
            print("4. Save them to the 'data/raw/' folder")
            print("5. Name them as: AAPL_10K_2024.txt, AAPL_10K_2023.txt, etc.")
            print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Fetch 10-K filings for Apple Inc.
    fetch_10k_filings("AAPL", years_back=2)  