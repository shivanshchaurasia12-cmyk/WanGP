import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import json, time
import numpy as np
import soundfile as sf
import librosa
import torch
from huggingface_hub import login
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

MODEL_NAME = "ai4bharat/indic-parler-tts"
TAKES = 3  # makes 3 versions of every line and keeps the most normal one
ENDING = "The delivery is natural, lively and conversational, like a real person talking. The recording is clear with no background noise, very close up."

BASE_VOICES = {
    "Narrator": "Divya's voice is warm and friendly, like a storyteller, with a moderate pace.",
    "Aalu": "Rohit speaks with a slightly high pitch and a fast pace.",
    "Baingan": "Aman speaks with a very low pitch and a slow pace.",
    "Tamatar": "Rani speaks with a high pitch and a fast pace.",
    "Gajar": "Rohit speaks with a very high pitch and a soft voice.",
}

EMOTIONS = {
    "calm": "The voice sounds calm and gentle.",
    "happy": "The voice sounds happy, smiling and lively, with a very expressive tone.",
    "excited": "The voice sounds very excited, energetic and animated, with a very expressive tone.",
    "angry": "The voice sounds angry, sharp and loud, with a very expressive tone.",
    "shy": "The voice sounds shy, soft and hesitant, a little slower.",
    "sad": "The voice sounds sad, slow and low.",
    "surprised": "The voice sounds surprised and dramatic, with a very expressive tone.",
    "scared": "The voice sounds scared and trembling, with a very expressive tone.",
    "proud": "The voice sounds proud, confident and strong.",
    "funny": "The voice sounds playful, funny and exaggerated, with a very expressive tone.",
}

# small cartoon pitch change per character (in semitones). Set all to 0 to switch off.
PITCH_SHIFT = {"Narrator": 0, "Aalu": 1, "Baingan": -1.5, "Tamatar": 1, "Gajar": 2}

PAUSE_SAME = 0.18
PAUSE_OTHER = 0.32


def build_description(speaker, emotion):
    emo = EMOTIONS.get(emotion, EMOTIONS["calm"])
    return f"{BASE_VOICES[speaker]} {emo} {ENDING}"


# ---------- audio helpers ----------
def trim_silence(audio, sr):
    trimmed, _ = librosa.effects.trim(audio, top_db=35)
    pad = np.zeros(int(0.04 * sr), dtype=np.float32)
    return np.concatenate([pad, trimmed, pad]).astype(np.float32)


def pick_middle_take(takes):
    order = sorted(range(len(takes)), key=lambda k: len(takes[k]))
    return takes[order[len(order) // 2]]


def normalize(audio, target_rms=0.08):
    rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-9
    audio = audio * (target_rms / rms)
    peak = float(np.max(np.abs(audio))) + 1e-9
    if peak > 0.95:
        audio = audio * (0.95 / peak)
    return audio.astype(np.float32)


def polish(audio, speaker, sr):
    audio = trim_silence(audio, sr)
    steps = PITCH_SHIFT.get(speaker, 0)
    if steps:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps).astype(np.float32)
    return normalize(audio)


# ---------- model ----------
token = os.environ.get("HF_TOKEN", "").strip()
if token:
    login(token=token)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
SR = model.config.sampling_rate


def one_take(text, speaker, emotion, seed):
    d = desc_tokenizer(build_description(speaker, emotion), return_tensors="pt").to(device)
    p = tokenizer(text, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            input_ids=d.input_ids,
            attention_mask=d.attention_mask,
            prompt_input_ids=p.input_ids,
            prompt_attention_mask=p.attention_mask,
        )
    return out.cpu().numpy().squeeze().astype(np.float32)


def speak(text, speaker, emotion, seed):
    takes = []
    for k in range(TAKES * 2):
        if len(takes) == TAKES:
            break
        try:
            audio = one_take(text, speaker, emotion, seed + k)
            if audio.size > SR // 4:
                takes.append(trim_silence(audio, SR))
        except Exception as e:
            print(f"  Voice error ({e}). Trying again...")
            time.sleep(3)
    if not takes:
        raise RuntimeError(f"Could not make voice for: {speaker}: {text}")
    return polish(pick_middle_take(takes), speaker, SR)


def main():
    with open("story.json", encoding="utf-8") as f:
        story = json.load(f)

    os.makedirs("audio", exist_ok=True)
    durations = []
    n = 0

    for i, scene in enumerate(story["scenes"], 1):
        pieces = []
        lines = scene["lines"]
        for j, ln in enumerate(lines):
            n += 1
            emotion = ln.get("emotion", "calm")
            print(f"Scene {i} | {ln['speaker']} ({emotion}): {ln['text']}")
            pieces.append(speak(ln["text"], ln["speaker"], emotion, seed=n * 10))
            if j < len(lines) - 1:
                same = lines[j + 1]["speaker"] == ln["speaker"]
                gap = PAUSE_SAME if same else PAUSE_OTHER
                pieces.append(np.zeros(int(gap * SR), dtype=np.float32))
        pieces.append(np.zeros(int(0.3 * SR), dtype=np.float32))
        audio = np.concatenate(pieces)
        path = f"audio/scene_{i}.wav"
        sf.write(path, audio, SR)
        durations.append(round(len(audio) / SR, 2))
        print("Saved", path, durations[-1], "sec")

    with open("audio/durations.json", "w") as f:
        json.dump(durations, f)
    print("Total seconds:", round(sum(durations), 1))


if __name__ == "__main__":
    main()
