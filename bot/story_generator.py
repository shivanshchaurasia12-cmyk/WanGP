import os, json, time, requests

API_KEY = os.environ["GEMINI_API_KEY"]
MODEL = "gemini-2.5-flash"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

PROMPT = """Write a short, funny, kid-friendly Hindi story (about 45 seconds when read aloud).
Characters: Aalu, Tamatar, Gajar and Baingan (talking vegetables).
Return ONLY JSON in this format:
{
  "title": "Hindi title",
  "scenes": [
    {"hindi_text": "1-2 simple Hindi sentences", "visual_prompt": "English description of the scene, cartoon vegetables, bright colors"}
  ]
}
Use exactly 6 scenes. End with a small moral."""

def generate_story():
    body = {
        "contents": [{"parts": [{"text": PROMPT}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    headers = {"x-goog-api-key": API_KEY, "Content-Type": "application/json"}
    for attempt in range(1, 8):
        r = requests.post(URL, json=body, headers=headers, timeout=120)
        if r.status_code in (429, 500, 503):
            wait = attempt * 20
            print(f"Gemini busy ({r.status_code}). Retry {attempt}/7 in {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    raise RuntimeError("Gemini stayed busy. Try again later.")

if __name__ == "__main__":
    story = generate_story()
    with open("story.json", "w", encoding="utf-8") as f:
        json.dump(story, f, ensure_ascii=False, indent=2)
    print("Story saved:", story["title"])
