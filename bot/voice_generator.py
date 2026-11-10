import os, json, time
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import login
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

MODEL_NAME = "ai4bharat/indic-parler-tts"
CLEAR = "The recording is very clear audio with no background noise, very close up."

VOICES = {
    "Narrator": f"Divya's voice is calm, warm and friendly, with a moderate pace and a slightly expressive tone. {CLEAR}",
    "Aalu": f"Rohit speaks with a slightly high pitch at a fast pace, in a very cheerful, silly and expressive tone. {CLEAR}",
    "Baingan": f"Aman speaks with a very low pitch at a slow pace, in a calm, deep and wise tone. {CLEAR}",
    "Tamatar": f"Rani speaks with a high pitch at a very fast pace, in an angry but funny and very expressive tone. {CLEAR}",
    "Gajar": f"Rohit speaks with a very high pitch at a fast pace, in a shy, sweet and soft tone. {CLEAR}",
}

PAUSE_SECONDS = 0.35

token = os.environ.get("HF_TOKEN", "").strip()
if token:
    login(token=token)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
SR = model.config.sampling_rate


def speak(text, speaker):
    description = VOICES[speaker]
    d = desc_tokenizer(description, return_tensors="pt").to(device)
    p = tokenizer(text, return_tensors="pt").to(device)
    for attempt in range(1, 4):
        try:
            torch.manual_seed(42)
            with torch.no_grad():
                out = model.generate(
                    input_ids=d.input_ids,
                    attention_mask=d.attention_mask,
                    prompt_input_ids=p.input_ids,
                    prompt_attention_mask=p.attention_mask,
                )
            audio = out.cpu().numpy().squeeze()
            if audio.size > SR // 4:
                return audio
        except Exception as e:
            print(f"Voice error ({e}). Retry {attempt}/3...")
            time.sleep(3)
    raise RuntimeError(f"Could not make voice for: {speaker}: {text}")


def main():
    with open("story.json", encoding="utf-8") as f:
        story = json.load(f)

    os.makedirs("audio", exist_ok=True)
    pause = np.zeros(int(PAUSE_SECONDS * SR), dtype=np.float32)
    durations = []

    for i, scene in enumerate(story["scenes"], 1):
        pieces = []
        for ln in scene["lines"]:
            print(f"Scene {i} | {ln['speaker']}: {ln['text']}")
            pieces.append(speak(ln["text"], ln["speaker"]).astype(np.float32))
            pieces.append(pause)
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
