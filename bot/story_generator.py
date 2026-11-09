import os, json, time, requests

API_KEY = os.environ["GEMINI_API_KEY"].strip()
MODELS = ["gemini-3.8-flash", "gemini-2.5-flash-lite"]
BASE = "https://generativelanguage.googleapis.com/v1beta/models"

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


def clean_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def is_valid(story):
    if not isinstance(story, dict) or not story.get("title"):
        return False
    scenes = story.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 6:
        return False
    for s in scenes:
        if not isinstance(s, dict) or not s.get("hindi_text") or not s.get("visual_prompt"):
            return False
    return True


def get_text(data):
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def generate_story():
    body = {
        "contents": [{"parts": [{"text": PROMPT}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    headers = {"x-goog-api-key": API_KEY, "Content-Type": "application/json"}

    for model in MODELS:
        url = f"{BASE}/{model}:generateContent"
        print(f"Using model: {model}")
        for attempt in range(1, 6):
            wait = attempt * 20
            try:
                r = requests.post(url, json=body, headers=headers, timeout=120)
            except requests.exceptions.RequestException as e:
                print(f"Network error: {e}. Retry {attempt}/5 in {wait}s...")
                time.sleep(wait)
                continue

            if r.status_code in (429, 500, 503):
                print(f"Gemini busy ({r.status_code}). Retry {attempt}/5 in {wait}s...")
                time.sleep(wait)
                continue

            if r.status_code == 404:
                print(f"Model {model} not available. Trying next model...")
                break

            if r.status_code != 200:
                raise RuntimeError(f"Gemini error {r.status_code}: {r.text}")

            try:
                story = clean_json(get_text(r.json()))
            except Exception as e:
                print(f"Bad reply ({e}). Retry {attempt}/5...")
                time.sleep(5)
                continue

            if is_valid(story):
                return story
            print(f"Story format wrong. Retry {attempt}/5...")
            time.sleep(5)

    raise RuntimeError("Could not get a valid story. Try again later.")


if __name__ == "__main__":
    story = generate_story()
    with open("story.json", "w", encoding="utf-8") as f:
        json.dump(story, f, ensure_ascii=False, indent=2)
    print("Story saved:", story["title"])
