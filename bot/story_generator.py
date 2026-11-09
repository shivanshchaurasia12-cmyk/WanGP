import os
import json
import random
import urllib.request
import urllib.error
import time

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

VEGETABLES = [
    "Aalu",
    "Tamatar",
    "Gajar",
    "Baingan"
]

STORY_TYPES = [
    "funny adventure",
    "unexpected problem",
    "friendship",
    "small mystery",
    "competition",
    "helping someone",
    "funny misunderstanding",
    "creative challenge"
]


def create_story():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    vegetable = random.choice(VEGETABLES)
    story_type = random.choice(STORY_TYPES)

    prompt = f"""
Create ONE original Hindi children's animated short story.

Main character: {vegetable}
Story type: {story_type}

Requirements:
- 45 to 60 seconds when spoken.
- Simple Hindi.
- Funny, warm and entertaining.
- Clear beginning, middle and ending.
- Give the character a real problem or goal.
- Include a surprising or funny moment.
- End with a small positive lesson.
- Use original dialogue.
- Do not copy any existing YouTube story.
- Do not use scary violence.
- Do not make the story feel like a repeated template.
- The same vegetable characters may appear in future stories,
  but this particular story must have a different situation and ending.

Return ONLY valid JSON.
Do not use Markdown code fences.
Use exactly this structure:

{{
  "title": "Hindi title",
  "characters": ["character names"],
  "story": "complete Hindi narration",
  "lesson": "small lesson",
  "scenes": [
    {{
      "scene": 1,
      "description": "what happens visually",
      "dialogue": "spoken Hindi dialogue"
    }}
  ]
}}
"""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.8-flash:generateContent"
    )

    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.9,
            "responseMimeType": "application/json"
        }
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        },
        method="POST"
    )

    last_error = None

    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))
            break

        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            last_error = (
                f"Gemini API request failed with HTTP {error.code}: "
                f"{error_body}"
            )

            if error.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** attempt)
                continue

            raise RuntimeError(last_error) from error

        except urllib.error.URLError as error:
            last_error = f"Gemini API request failed: {error}"

            if attempt < 2:
                time.sleep(2 ** attempt)
                continue

            raise RuntimeError(last_error) from error

    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            "Gemini returned an unexpected response: "
            + json.dumps(result, ensure_ascii=False)
        ) from error

    if text.startswith("```json"):
        text = text[7:]

    if text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        story = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Gemini returned invalid JSON: {text}"
        ) from error

    required_keys = [
        "title",
        "characters",
        "story",
        "lesson",
        "scenes"
    ]

    missing_keys = [
        key for key in required_keys
        if key not in story
    ]

    if missing_keys:
        raise RuntimeError(
            "Gemini JSON is missing required fields: "
            + ", ".join(missing_keys)
        )

    if not isinstance(story["scenes"], list) or not story["scenes"]:
        raise RuntimeError("Gemini returned no scenes.")

    return story


if __name__ == "__main__":
    story = create_story()
    print(json.dumps(story, ensure_ascii=False, indent=2))
