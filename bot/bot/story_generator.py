import os
import json
import random
import urllib.request

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

Return ONLY valid JSON in this format:

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
        "gemini-2.5-flash:generateContent?key="
        + GEMINI_API_KEY
    )

    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    text = result["candidates"][0]["content"]["parts"][0]["text"]

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    if text.endswith("```"):
        text = text[:-3]

    return json.loads(text.strip())


if __name__ == "__main__":
    story = create_story()
    print(json.dumps(story, ensure_ascii=False, indent=2))
