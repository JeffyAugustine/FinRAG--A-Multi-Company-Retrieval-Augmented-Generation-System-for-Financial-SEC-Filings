import os

def verify_downloads():
    raw_dir = "../data/raw"
    
    print("Checking downloaded files...")
    print("=" * 50)
    
    for filename in os.listdir(raw_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(raw_dir, filename)
            file_size = os.path.getsize(filepath)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                first_lines = [f.readline().strip() for _ in range(3)]
            
            print(f"📄 {filename}")
            print(f"   Size: {file_size:,} bytes")
            print(f"   First line: {first_lines[0][:100]}...")
            print()

if __name__ == "__main__":
    verify_downloads()