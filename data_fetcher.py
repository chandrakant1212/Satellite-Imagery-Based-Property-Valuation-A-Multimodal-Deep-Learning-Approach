import requests
import os
import pandas as pd
from tqdm import tqdm
import argparse
import time

# Configuration
MAPBOX_TOKEN = "pk.eyJ1IjoiY2hhbmRyYWthbnQxMSIsImEiOiJjbWpkdXpoZjcwYnJxM2RzZGUyMnNxYWc1In0.YP3JtdCa7lW_jpVZE4GJtg"
IMAGE_DIR = "data/images"
DATA_PATH = "data/raw/train.csv"
ZOOM_LEVEL = 18
SIZE = "256x256"

def fetch_satellite_image(lat, lon, house_id, output_dir, session=None):
    """
    Fetches a static satellite image from Mapbox.
    """
    filename = os.path.join(output_dir, f"{house_id}.png")
    
    # Skip if already exists
    if os.path.exists(filename):
        return "skipped"

    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"{lon},{lat},{ZOOM_LEVEL}/{SIZE}?access_token={MAPBOX_TOKEN}"
    )

    try:
        if session:
            response = session.get(url, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            return "downloaded"
        else:
            print(f" Failed for ID {house_id} ({lat}, {lon}): Status {response.status_code}")
            return "failed"
            
    except Exception as e:
        print(f" Error fetching ID {house_id}: {e}")
        return "error"

def main():
    parser = argparse.ArgumentParser(description="Download satellite images for property valuation.")
    parser.add_argument("--limit", type=int, help="Limit number of images to download (for testing)")
    parser.add_argument("--csv", type=str, default=DATA_PATH, help="Path to the dataset CSV")
    parser.add_argument("--output", type=str, default=IMAGE_DIR, help="Directory to save images")
    
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    if not os.path.exists(args.csv):
        print(f"Error: Dataset not found at {args.csv}")
        return

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} records from {args.csv}")
    
    if args.limit:
        df = df.head(args.limit)
        print(f"Limiting to first {args.limit} records.")

    # Use generic generic ThreadPoolExecutor for faster downloads
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Initialize session
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
    session.mount('https://', adapter)
    
    print(f"Starting concurrent download to {args.output} with limit={args.limit}...")
    
    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "error": 0}
    
    # helper for threads
    def download_wrapper(row):
        return fetch_satellite_image(row['lat'], row['long'], row['id'], args.output, session)

    # Run in threads
    rows = [row for _, row in df.iterrows()]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # submit all
        future_to_row = {executor.submit(download_wrapper, r): r for r in rows}
        
        for future in tqdm(as_completed(future_to_row), total=len(rows), desc="Fetching Images"):
            status = future.result()
            stats[status] += 1

    print("\nDownload Summary:")
    print(f" Downloaded: {stats['downloaded']}")
    print(f" Skipped:    {stats['skipped']}")
    print(f" Failed:     {stats['failed']}")
    print(f" Errors:     {stats['error']}")

if __name__ == "__main__":
    main()
