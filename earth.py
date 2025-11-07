import requests
import os

# your API key
API_KEY = "PLAK27df235af5f240a6b39711fd4abeab74"

# output folder
OUTPUT_DIR = "planet_thumbnails"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# create session with auth
session = requests.Session()
session.auth = (API_KEY, "")

# Planet quick-search endpoint
search_url = "https://api.planet.com/data/v1/quick-search"

# ---- CHANGE THESE COORDINATES ----
longitude = 77.5946   # Example: Bangalore
latitude = 12.9716

search_request = {
    "item_types": ["PSScene"],
    "filter": {
        "type": "AndFilter",
        "config": [
            {
                "type": "GeometryFilter",
                "field_name": "geometry",
                "config": {
                    "type": "Point",
                    "coordinates": [longitude, latitude]
                }
            },
            {
                "type": "DateRangeFilter",
                "field_name": "acquired",
                "config": {
                    "gte": "2023-01-01T00:00:00.000Z",
                    "lte": "2023-12-31T00:00:00.000Z"
                }
            },
            {
                "type": "RangeFilter",
                "field_name": "cloud_cover",
                "config": {"lte": 0.2}
            }
        ]
    }
}

# send request
response = session.post(search_url, json=search_request)
response.raise_for_status()
features = response.json().get("features", [])

print(f"✅ Found {len(features)} images for given coordinates. Downloading thumbnails...")

# download thumbnails
count = 0
for feature in features[:]:  # limit to 20 results
    image_id = feature["id"]
    thumb_url = feature["_links"]["thumbnail"]

    r = session.get(thumb_url)
    if r.status_code == 200:
        filename = os.path.join(OUTPUT_DIR, f"{image_id}_thumb.jpg")
        with open(filename, "wb") as f:
            f.write(r.content)
        count += 1
        print(f"🖼️ Saved thumbnail: {filename}")
    else:
        print(f"❌ Failed to get thumbnail for {image_id}")

print(f"\n✅ Done! Downloaded {count} thumbnails to {OUTPUT_DIR}/")
