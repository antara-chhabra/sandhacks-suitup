"""
API-callable interview question generator - can be imported and run with company/position
"""
import json
import re
import os

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def generate_interview_questions(company: str, position: str, output_path: str = None) -> dict:
    """
    Generate interview questions for company/position.
    Uses web scraping + Llama if available, else returns demo data.
    """
    try:
        import trafilatura
        from ddgs import DDGS
        import ollama
        import time
        import random
    except ImportError as e:
        print(f"Import error: {e}")
        return _demo_output(company, position)

    def search_ddg(query, max_links=2):
        links = []
        try:
            results = DDGS().text(query, max_results=max_links)
            if results:
                for r in results:
                    links.append(r['href'])
        except Exception as e:
            print(f"Search failed: {e}")
        return links

    def scrape_clean_text(links):
        data = []
        for link in links:
            try:
                if "github.com" in link and "blob" in link:
                    link = link.replace("blob", "raw")
                downloaded = trafilatura.fetch_url(link)
                if downloaded:
                    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                    if text and len(text) > 100:
                        data.append(f"SOURCE: {link} | CONTENT: {text[:3000].replace(chr(10), ' ')}")
                time.sleep(random.uniform(0.5, 1.0))
            except Exception as e:
                print(f"Scrape error: {e}")
        return "\n\n".join(data)

    queries = [
        f'site:github.com "{company}" "{position}" interview questions',
        f'{company} leadership principles values',
    ]
    full_context = ""
    for q in queries:
        links = search_ddg(q, max_links=1)
        full_context += scrape_clean_text(links)

    prompt = f"""
You are an API that outputs strictly JSON.
Context: User preparing for a {position} interview at {company}.
RAW DATA:
{full_context[:4000] if full_context else "No context"}

REQUIRED JSON FORMAT:
{{
    "company_values": ["value 1", "value 2"],
    "technical_questions": ["question 1", "question 2", "question 3"],
    "behavioral_questions": ["question 1", "question 2", "question 3"]
}}
Only output JSON. No markdown."""

    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )
        content = response["message"]["content"]
        cleaned = extract_json(content)
        if cleaned:
            parsed = json.loads(cleaned)
            if output_path:
                with open(output_path, "w") as f:
                    json.dump(parsed, f, indent=2)
            return parsed
    except Exception as e:
        print(f"Ollama error: {e}")
    return _demo_output(company, position)


def _demo_output(company: str, position: str) -> dict:
    return {
        "company_values": [f"Excellence at {company}", "Collaboration", "Innovation"],
        "technical_questions": [
            f"Describe a technical challenge you solved as a {position}.",
            "Explain a system you designed.",
            "How do you approach debugging complex issues?",
        ],
        "behavioral_questions": [
            f"Why do you want to join {company}?",
            "Tell me about a time you faced a difficult teammate.",
            "How do you handle ambiguity and changing requirements?",
        ],
    }
