import os
import time
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import pdfplumber

# ------------------------
# CONFIGURATION
# ------------------------
API_KEY = "6f7206d749b26cad9556d07641c04c0216bcfde01b5805ad1b23f22b209298e9"  # Get from serpapi.com (Free tier available)
SEARCH_QUERY = '"terms and conditions"'
RESULTS_TO_FETCH = 1000
SAVE_FOLDER = "terms_and_conditions"
os.makedirs(SAVE_FOLDER, exist_ok=True)

def fetch_urls(query, num_results=1000):
    """Fetch URLs from Google using SerpAPI."""
    urls = ["https://www.termsfeed.com/blog/sample-terms-and-conditions-template/",
        "https://www.iubenda.com/en/help/2859-terms-and-conditions-when-are-they-needed",
        "https://www.icertis.com/contracting-basics/what-are-terms-and-conditions/",
        "https://www.enzuzo.com/blog/write-terms-and-conditions",
        "https://www.contractscounsel.com/t/us/terms-and-conditions",
        "https://www.termsfeed.com/blog/5-reasons-need-terms-conditions/",
        "https://www.termsandconditionsgenerator.com/",
        "https://www.youtube.com/watch?v=2lZ4ZbNGVUU",
        "https://www.docusign.com/legal/terms-and-conditions",
        "https://ironcladapp.com/journal/contracts/website-terms-and-conditions/",
        "https://www.iubenda.com/en/help/53008-terms-and-conditions-template",
        "https://www.lexisnexis.com/en-us/terms/general/default.page",
        "https://www.websitepolicies.com/blog/what-are-terms-and-conditions",
        "https://www.apple.com/legal/internet-services/itunes/",
        "https://www.enzuzo.com/blog/terms-and-conditions-examples",
        "https://www.spotify.com/legal",
        "https://www.privacypolicies.com/blog/how-to-write-terms-conditions/",
        "https://www.freeprivacypolicy.com/free-terms-and-conditions-generator/",
        "https://www.privacypolicyonline.com/sample-terms-conditions-template/",
        "https://mailchimp.com/resources/website-terms-of-use/"]
    page = 0

    while len(urls) < num_results:
        params = {
            "engine": "google",
            "q": query,
            "num": 100,
            "start": page * 100,
            "api_key": API_KEY
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get('organic_results', [])
        if not organic_results:
            break
        for res in organic_results:
            link = res.get('link')
            if link and link not in urls:
                urls.append(link)
        page += 1
        time.sleep(2)  # avoid hitting API too fast
    return urls[:num_results]

def extract_text_from_html(url):
    """Extract text from HTML page."""
    try:
        r = requests.get(url, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        return soup.get_text(separator="\n")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_text_from_pdf(url):
    """Extract text from PDF if page is a PDF."""
    try:
        r = requests.get(url, timeout=20)
        pdf_path = os.path.join(SAVE_FOLDER, "temp.pdf")
        with open(pdf_path, "wb") as f:
            f.write(r.content)
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        os.remove(pdf_path)
        return text
    except Exception as e:
        print(f"Error processing PDF {url}: {e}")
        return None

def save_text(content, index):
    """Save extracted text to a file."""
    file_path = os.path.join(SAVE_FOLDER, f"terms_{index}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

# ------------------------
# MAIN LOGIC
# ------------------------
print("Fetching URLs from Google...")
urls = fetch_urls(SEARCH_QUERY, RESULTS_TO_FETCH)
print(f"Found {len(urls)} URLs")

print("Downloading and extracting text...")
for i, url in enumerate(urls):
    print(f"[{i+1}/{len(urls)}] Processing: {url}")
    content = None
    if url.lower().endswith(".pdf"):
        content = extract_text_from_pdf(url)
    else:
        content = extract_text_from_html(url)

    if content:
        save_text(content, i)
    time.sleep(1)  # polite delay
print("✅ Download complete! Check the 'terms_and_conditions' folder.")
