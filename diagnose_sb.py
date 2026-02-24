import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client

def diagnose():
    load_dotenv(override=True)
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    
    if not url or not key:
        print("Missing environment variables!")
        return

    print(f"Attempting to connect to: {url}")
    
    # Strip whitespace/quotes which often cause getaddrinfo failed
    url = url.strip().strip('"').strip("'")
    key = key.strip().strip('"').strip("'")
    
    try:
        supabase: Client = create_client(url, key)
        print("Client created. Fetching table 'crowd_data'...")
        
        response = supabase.table('crowd_data').select('*').execute()
        
        with open('temp.txt', 'w') as f:
            f.write("--- Supabase Data (crowd_data) ---\n")
            f.write(json.dumps(response.data, indent=2))
        
        print(f"Success! Fetched {len(response.data)} rows. Data written to temp.txt")
        
    except Exception as e:
        print(f"Error: {e}")
        with open('temp.txt', 'w') as f:
            f.write(f"Error fetching data: {e}")

if __name__ == "__main__":
    diagnose()
