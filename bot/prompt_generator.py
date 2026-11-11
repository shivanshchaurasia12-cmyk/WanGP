import os, json, random, re, time, requests

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(HERE, "history.json")

GROQ_MODELS = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]

THEMES = [
    "misty ancient forest at dawn", "lone astronaut on a red desert planet",
    "glowing jellyfish in a deep ocean trench", "dragon flying over snowy mountains",
    "rainy neon city street at night", "floating islands above the clouds",
    "samurai standing in a bamboo storm", "giant whale drifting through the sky",
    "lightning storm over a lonely lighthouse", "crystal cave with glowing water",
    "wolf running through an autumn valley", "abandoned spaceship in a jungle",
    "northern lights over a frozen lake", "castle in the sky at golden hour",
    "underwater ruins of a lost city", "desert caravan under a giant moon",
]
MOODS = ["epic", "mysterious", "peaceful", "dramatic", "dreamlike", "powerful"]

SYSTEM = (
    "You are three experts in one: a Hollywood film director, a cinematographer, "
    "and a viral Instagram Reels and YouTube Shorts strategist. "
    "You write prompts for an AI text-to-video model that makes 5-second vertical (9:16) clips. "
    "RULES for scene_prompt: one flowing paragraph, present tense, 40-70 words, no lists. "
    "Always include: the subject, one clear action or motion, ONE camera move "
    "(slow push-in, orbit, tracking shot, crane up, or drone fly-over), "
    "lighting (golden hour, volumetric fog, neon glow, moonlight), "
    "colour mood, lens feel (35mm, anamorphic), and the words 'cinematic, ultra detailed'. "
    "Never use text, logos, real people, or celebrities. Keep motion simple so the video looks clean. "
    "RULES for the post: the title must hook in the first 3 words, under 60 characters. "
    "caption_instagram: 2 short lines plus a question for comments. "
    "caption_youtube: 1 line plus the words #shorts. "
    "hashtags: 10 items, mix of 3 big, 4 medium and 3 niche tags. "
    "Return ONLY valid JSON with keys: scene_prompt, title, caption_instagram, "
    "caption_youtube, hashtags (list of strings)."
)

def get_secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.environ.get(name)

def load_history():
    try:
        with open(HISTORY) as f:
            return json.load(f)
    except Exception:
        return []

def save_history(items):
    try:
        with open(HISTORY, "w") as f:
            json.dump(items[-40:], f)
    except Exception:
        pass

def clean_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])

def call_groq(model, user):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {get_secret('GROQ_API_KEY')}"},
        json={
            "model": model,
            "temperature": 0.9,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def call_gemini(model, user):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={get_secret('GEMINI_API_KEY')}")
    r = requests.post(
        url,
        json={
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.9,
                                 "responseMimeType": "application/json"},
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]

def is_valid(d):
    need = ["scene_prompt", "title", "caption_instagram", "caption_youtube", "hashtags"]
    if not all(k in d for k in need):
        return False
    words = len(str(d["scene_prompt"]).split())
    return 20 <= words <= 100 and isinstance(d["hashtags"], list) and len(d["hashtags"]) >= 5

def make_post():
    history = load_history()
    fresh = [t for t in THEMES if t not in history] or THEMES
    theme = random.choice(fresh)
    mood = random.choice(MOODS)
    user = (f"Theme: {theme}. Mood: {mood}. "
            f"Do not repeat these recent themes: {history[-10:]}. Write the post now.")

    attempts = [("groq", m) for m in GROQ_MODELS] + [("gemini", m) for m in GEMINI_MODELS]
    for provider, model in attempts:
        try:
            raw = call_groq(model, user) if provider == "groq" else call_gemini(model, user)
            data = clean_json(raw)
            if not is_valid(data):
                raise ValueError("bad output")
            data["hashtags"] = ["#" + str(h).lstrip("#").replace(" ", "") for h in data["hashtags"]]
            data["theme"] = theme
            data["model_used"] = f"{provider}:{model}"
            save_history(history + [theme])
            return data
        except Exception as e:
            print(f"[prompt] {provider}:{model} failed -> {e}")
            time.sleep(2)
    raise RuntimeError("All models failed. Check your API keys.")

if __name__ == "__main__":
    print(json.dumps(make_post(), indent=2))
