import json, os, time, requests

API_KEY = os.environ["GEMINI_API_KEY"].strip()
MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
          "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
BASE = "https://generativelanguage.googleapis.com/v1beta/models"
VIDEO_MODEL = "ltx2_distilled_gguf_q8_0"

CHARACTERS = (
    "Aalu is a round cheerful potato with big eyes. Baingan is a calm purple eggplant. "
    "Tamatar is a round red tomato with a fiery look. Gajar is a small shy orange carrot. "
    "All have big expressive eyes, small arms and legs."
)
STYLE = "Cute 3D Pixar style, bright colors, soft light, smooth animation, kid friendly."

EXPERTS = """You are a team of 4 master experts. Each one is the best in the world at their job.
1. STORY DIRECTOR (20 years in kids' animation): decides what the viewer must feel and see in each scene,
   and keeps the 6 scenes flowing as one story.
2. CINEMATOGRAPHER (Pixar level): chooses the camera shot and ONE simple camera move per scene
   (close-up, wide shot, slow push-in, gentle pan). Keeps the lighting warm and bright.
3. 3D CHARACTER ANIMATOR: decides each character's body language and facial emotion from the
   given emotion. Mouths move as if talking. Movements are small, clear and fun.
4. AI VIDEO PROMPT ENGINEER (expert in LTX video models): writes the final prompt as ONE flowing
   paragraph of 60 to 90 words, present tense, in this order: setting, characters and what they
   look like, action and emotion, camera move, lighting and style.
Hard rules for every prompt:
- Always describe each visible character fully again (never say "the same character").
- Only the vegetable characters. No humans. No text, no subtitles, no logos on screen.
- Safe for small kids. No scary or violent action.
- Keep one simple action per scene so the video does not glitch.
- Write in English."""


def _ask_gemini(text):
    body = {"contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"responseMimeType": "application/json"}}
    headers = {"x-goog-api-key": API_KEY, "Content-Type": "application/json"}
    for model in MODELS:
        try:
            r = requests.post(f"{BASE}/{model}:generateContent", json=body,
                              headers=headers, timeout=120)
            if r.status_code != 200:
                print(f"  {model}: error {r.status_code}")
                continue
            parts = r.json()["candidates"][0]["content"]["parts"]
            out = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            return json.loads(out.strip().strip("`").removeprefix("json").strip())
        except Exception as e:
            print(f"  {model}: {e}")
    return None


def expert_prompts(story):
    scenes = []
    for i, s in enumerate(story["scenes"], 1):
        scenes.append({
            "scene": i,
            "idea": s["visual_prompt"],
            "lines": [{"who": l["speaker"], "feeling": l["emotion"], "says": l["text"]}
                      for l in s["lines"]],
        })
    task = (EXPERTS + "\n\nCharacters:\n" + CHARACTERS + "\n\nStyle: " + STYLE +
            "\n\nStory title: " + story["title"] + "\nScenes:\n" +
            json.dumps(scenes, ensure_ascii=False) +
            '\n\nTeam, work together and return ONLY JSON: {"prompts": ["prompt for scene 1", ...]} '
            "with exactly one prompt per scene, in order.")
    for attempt in range(3):
        data = _ask_gemini(task)
        if data and isinstance(data.get("prompts"), list) and len(data["prompts"]) == len(scenes):
            return [str(p) for p in data["prompts"]]
        print("  Expert team retry...")
        time.sleep(10)
    print("  Using simple prompts instead.")
    return [s["visual_prompt"] + ". " + CHARACTERS + " " + STYLE for s in story["scenes"]]


def frames_for(seconds):
    f = int(round(seconds * 24 / 8)) * 8 + 1
    return max(49, min(f, 241))


def write_all(story, seconds_list, folder="scenes"):
    os.makedirs(folder, exist_ok=True)
    prompts = expert_prompts(story)
    paths = []
    for i, p in enumerate(prompts):
        path = f"{folder}/scene{i + 1}.json"
        settings = {"model_type": VIDEO_MODEL, "prompt": p,
                    "num_inference_steps": 8, "video_length": frames_for(seconds_list[i])}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f)
        paths.append(path)
    return paths
