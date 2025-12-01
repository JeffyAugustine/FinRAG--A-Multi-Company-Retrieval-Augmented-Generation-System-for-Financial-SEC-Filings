import openai
import os
from dotenv import load_dotenv

load_dotenv()  # This loads your .env file

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("✅ API Key loaded successfully!")
    print(f"Key starts with: {api_key[:10]}...")
else:
    print("❌ API Key not found!")