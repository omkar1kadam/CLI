from serpapi import GoogleSearch
import time

API_KEY = "6f7206d749b26cad9556d07641c04c0216bcfde01b5805ad1b23f22b209298e9"

def fetch_urls(query, num_results=50):
    urls = []
    page = 0
    while len(urls) < num_results:
        params = {
            "engine": "google",
            "q": query,
            "num": 10,  # safer for testing
            "start": page * 10,
            "api_key": API_KEY
        }
        print(f"Fetching page {page + 1}...")
        search = GoogleSearch(params)
        results = search.get_dict()

        organic_results = results.get('organic_results', [])
        print(f"Found {len(organic_results)} organic results")

        if not organic_results:
            print("No more results or error occurred")
            break

        for res in organic_results:
            link = res.get('link')
            if link and link not in urls:
                urls.append(link)

        page += 1
        time.sleep(2)

    return urls[:num_results]

# ✅ Call the function outside
urls = fetch_urls('site:.com "terms and conditions"', num_results=20)
print(f"Collected {len(urls)} URLs:")
for i, url in enumerate(urls, 1):
    print(f"{i}. {url}")
