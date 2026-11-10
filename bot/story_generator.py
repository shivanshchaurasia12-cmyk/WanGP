import os, json, time, requests

API_KEY = os.environ["GEMINI_API_KEY"].strip()
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]
BASE = "https://generativelanguage.googleapis.com/v1beta/models"
SPEAKERS = ["Narrator", "Aalu", "Baingan", "Tamatar", "Gajar"]

PROMPT = """Write a short, funny, kid-friendly Hindi story (about 45 seconds when read aloud).
Characters: Aalu (potato, silly and cheerful), Baingan (eggplant, calm and wise),
Tamatar (tomato, hot-tempered and fast talker), Gajar (carrot, shy and sweet).
The story is mostly DIALOGUE between the vegetables, with a short Narrator when needed.
Return ONLY JSON in this format:
{
  "title": "Hindi title",
  "scenes": [
    {
      "lines": [
        {"speaker": "Narrator", "text": "one short Hindi sentence"},
        {"speaker": "Aalu", "text": "one short Hindi sentence"}
      ],
      "visual_prompt": "English description of the scene, 3D Pixar style cartoon vegetables, bright colors"
    }
  ]
}
Rules:
- Use exactly 6 scenes.
- Each scene has 1 to 3 lines. Each line is one short, simple Hindi sentence in Devanagari.
- The speaker must be exactly one of: Narrator, Aalu, Baingan, Tamatar, Gajar.
- End with a small moral said by the Narrator."""


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
        if not isinstance(s, dict) or not s.get("visual_prompt"):
            return False
        lines = s.get("lines")
        if not isinstance(lines, list) or not 1 <= len(lines) <= 3:
            return False
        for ln in lines:
            if not isinstance(ln, dict) or not ln.get("text"):
                return False
            if ln.get("speaker") not in SPEAKERS:
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
    models = list(MODELS)

    for round_no in range(1, 5):
        for model in list(models):
            url = f"{BASE}/{model}:generateContent"
            print(f"Round {round_no} | model: {model}")
            try:
                r = requests.post(url, json=body, headers=headers, timeout=120)
            except requests.exceptions.RequestException as e:
                print(f"  Network error: {e}")
                continue

            if r.status_code == 404:
                print("  Model not available. Removing it.")
                models.remove(model)
                continue

            if r.status_code in (429, 500, 503):
                print(f"  Busy ({r.status_code}). Next model...")
                continue

            if r.status_code != 200:
                print(f"  Error {r.status_code}: {r.text[:300]}")
                continue

            try:
                story = clean_json(get_text(r.json()))
            except Exception as e:
                print(f"  Bad reply ({e}). Next model...")
                continue

            if is_valid(story):
                print("  Story OK.")
                return story
            print("  Story format wrong. Next model...")

        if not models:
            break
        wait = round_no * 30
        print(f"All models busy. Waiting {wait}s, then trying again...")
        time.sleep(wait)

    raise RuntimeError("Could not get a valid story. Try again later.")


if __name__ == "__main__":
    story = generate_story()
    with open("story.json", "w", encoding="utf-8") as f:
        json.dump(story, f, ensure_ascii=False, indent=2)
    print("Story saved:", story["title"])
