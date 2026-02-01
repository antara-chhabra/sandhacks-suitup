import time
import random
import json
import re
import trafilatura
import ollama
from ddgs import DDGS  # Updated package

# --- CONFIGURATION ---
import argparse
OLLAMA_MODEL = "llama3"
DEMO_MODE = False  # Set TRUE if wifi dies during demo


# ---------------- JSON EXTRACTION ----------------
def extract_json(text):
    """
    Extract first JSON object from LLM output.
    Handles extra words before/after JSON.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return None


# ---------------- QUERY BUILDER ----------------
def get_smart_queries(company, position):
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


# ---------------- SEARCH ----------------
def search_ddg(query, max_links=2):
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


# ---------------- SCRAPER ----------------
def scrape_clean_text(links):
    data = []

    for link in links:
        print(f"      🔗 Fetching: {link}")

        try:
            # Convert GitHub blob to raw text
            if "github.com" in link and "blob" in link:
                link = link.replace("blob", "raw")

            downloaded = trafilatura.fetch_url(link)

            if downloaded:
                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False
                )

                if text and len(text) > 100:
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


# ---------------- LLM GENERATOR ----------------
def generate_interview_json(company, role, context_text):
    print("\n🧠 Sending to Llama (JSON Mode)...")

    prompt = f"""
You are an API that outputs strictly JSON.

Context: User preparing for a {role} interview at {company}.

RAW DATA:
{context_text}

REQUIRED JSON FORMAT:
{{
    "company_values": ["value 1", "value 2"],
    "technical_questions": ["question 1", "question 2", "question 3"],
    "behavioral_questions": ["question 1", "question 2", "question 3"]
}}

RULES:
1. Only output JSON.
2. No explanations.
3. No markdown.
4. Infer missing info if needed.
"""

    if DEMO_MODE:
        return json.dumps({
            "company_values": ["Focus on the user", "Think big"],
            "technical_questions": [
                "Reverse a linked list",
                "Design a scalable chat system",
                "Explain time complexity of quicksort"
            ],
            "behavioral_questions": [
                "Tell me about a failure",
                "Describe a difficult teammate",
                "How do you handle ambiguity?"
            ]
        })

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            format="json"  # Native JSON enforcement
        )

        content = response['message']['content']
        return content

    except Exception as e:
        print(f"❌ Llama Error: {e}")
        return "{}"


# ---------------- MAIN ----------------
def main(company="Google", position="Software Engineer"):
    TARGET_COMPANY = company
    TARGET_POSITION = position
    print(f"🚀 SuitUp Generator: {TARGET_COMPANY} - {TARGET_POSITION}\n")

    queries = get_smart_queries(TARGET_COMPANY, TARGET_POSITION)
    full_context = ""

    all_queries = queries["technical"] + queries["behavioral"]

    for q in all_queries:
        links = search_ddg(q, max_links=1)
        full_context += scrape_clean_text(links)

    print(f"\n📊 Context gathered: {len(full_context)} chars")

    json_output = generate_interview_json(
        TARGET_COMPANY,
        TARGET_POSITION,
        full_context
    )

    filename = "interview_data.json"

    try:
        cleaned = extract_json(json_output)

        if not cleaned:
            raise ValueError("No JSON found")

        parsed = json.loads(cleaned)

        with open(filename, "w") as f:
            json.dump(parsed, f, indent=2)

        print(f"\n✅ Success! Saved structured data to {filename}")
        print(json.dumps(parsed, indent=2))

    except Exception:
        print("\n⚠️ AI did not return valid JSON.")
        print("Raw output:\n", json_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", default="Google", help="Target company")
    parser.add_argument("--position", default="Software Engineer", help="Target position")
    args = parser.parse_args()
    main(company=args.company, position=args.position)
