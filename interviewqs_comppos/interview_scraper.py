import time
import random
import json
import trafilatura
import ollama
from duckduckgo_search import DDGS  # More reliable than Google for scripts

# --- CONFIGURATION ---
TARGET_COMPANY = "Google"
TARGET_POSITION = "Software Engineer"
OLLAMA_MODEL = "llama3" 
DEMO_MODE = False  # Set to TRUE if presentation wifi fails!

def get_smart_queries(company, position):
    """
    Target sources that are text-heavy and easier to scrape (GitHub, Reddit, Leetcode)
    Avoids heavy JS sites like Glassdoor which block scrapers.
    """
    return {
        "technical": [
            f'site:github.com "{company}" "{position}" interview questions',
            f'site:leetcode.com/discuss "{company}" "{position}" interview experience',
        ],
        "behavioral": [
            f'{company} leadership principles values',
            f'site:reddit.com r/cscareerquestions "{company}" interview',
        ]
    }

def search_ddg(query, max_links=2):
    """
    Wrapper for DuckDuckGo Search (No API Key needed)
    """
    links = []
    print(f"   🦆 DuckDuckGo: '{query}'")
    try:
        results = DDGS().text(query, max_results=max_links)
        if results:
            for r in results:
                links.append(r['href'])
    except Exception as e:
        print(f"      ❌ Search failed: {e}")
    return links

def scrape_clean_text(links):
    data = []
    for link in links:
        print(f"      🔗 Fetching: {link}")
        try:
            # If it's a GitHub link, try to get the Raw version for cleaner text
            if "github.com" in link and "blob" in link:
                link = link.replace("blob", "raw")
            
            downloaded = trafilatura.fetch_url(link)
            
            if downloaded:
                # Extract main text, strip navigation/footers
                text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                if text and len(text) > 100: 
                    # Truncate to avoid context window overflow
                    clean_chunk = text[:3000].replace("\n", " ") 
                    data.append(f"SOURCE: {link} | CONTENT: {clean_chunk}")
                else:
                    print("      ⚠️ Content empty.")
            else:
                print("      ⚠️ Download failed.")
                
            time.sleep(random.uniform(0.5, 1.5)) 
            
        except Exception as e:
            print(f"      ❌ Scraping error: {e}")
            
    return "\n\n".join(data)

def generate_interview_json(company, role, context_text):
    print("\n🧠 Sending to Llama (JSON Mode)...")
    
    # We enforce JSON structure in the prompt
    prompt = f"""
    You are an API that outputs strictly JSON.
    Context: A user is preparing for a {role} interview at {company}.
    
    Based on the scraped data below, generate a JSON object with 3 distinct lists.
    
    RAW DATA:
    {context_text}
    
    REQUIRED JSON FORMAT:
    {{
        "company_values": ["value 1", "value 2"],
        "technical_questions": ["question 1", "question 2", "question 3"],
        "behavioral_questions": ["question 1", "question 2", "question 3"]
    }}
    
    RULES:
    1. specific technical questions mentioned in the data.
    2. If data is missing, infer reasonable questions for {company}.
    3. Do NOT output markdown formatting (like ```json), just the raw JSON string.
    """

    if DEMO_MODE:
        return json.dumps({
            "company_values": ["Do the right thing", "Focus on the user"],
            "technical_questions": ["Reverse a linked list", "System Design: Instagram"],
            "behavioral_questions": ["Tell me about a time you failed."]
        })

    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {'role': 'user', 'content': prompt},
        ])
        content = response['message']['content']
        
        # Clean up if Llama adds markdown blocks
        content = content.replace("```json", "").replace("```", "").strip()
        return content
        
    except Exception as e:
        print(f"❌ Llama Error: {e}")
        return "{}"

def main():
    print(f"🚀 SuitUp Generator: {TARGET_COMPANY} - {TARGET_POSITION}\n")
    
    queries = get_smart_queries(TARGET_COMPANY, TARGET_POSITION)
    full_context = ""
    
    # 1. Gather Tech & Culture Data
    all_queries = queries["technical"] + queries["behavioral"]
    
    for q in all_queries:
        links = search_ddg(q, max_links=1)
        full_context += scrape_clean_text(links)
    
    print(f"\n📊 Context gathered: {len(full_context)} chars")
    
    # 2. Generate JSON for Frontend
    json_output = generate_interview_json(TARGET_COMPANY, TARGET_POSITION, full_context)
    
    # 3. Save/Print
    filename = "interview_data.json"
    try:
        # Validate it's real JSON
        parsed = json.loads(json_output)
        with open(filename, "w") as f:
            json.dump(parsed, f, indent=2)
        print(f"\n✅ Success! Saved structured data to {filename}")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print("\n⚠️ AI did not return valid JSON. Raw output:")
        print(json_output)

if __name__ == "__main__":
    main()
    