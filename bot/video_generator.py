import os, glob, json, random, subprocess, shutil, time, queue
from concurrent.futures import ThreadPoolExecutor

PYTHON = "/kaggle/working/env310/bin/python"
WGP_DIR = "/kaggle/working/WanGP"
GPUS = [0, 1]
VIDEO_OUT = "/kaggle/working/videos"
RESOLUTION = "480x832"  # vertical 9:16
NEGATIVE = ("blurry, low quality, distorted, deformed, watermark, text, logo, "
            "flicker, jitter, bad anatomy, overexposed, static, frozen")


# ---------- helpers ----------
def probe(path, entries, stream=None):
    cmd = ["ffprobe", "-v", "error"]
    if stream:
        cmd += ["-select_streams", stream]
    cmd += ["-show_entries", entries, "-of", "csv=p=0", path]
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def get_duration(path):
    try:
        return float(probe(path, "format=duration"))
    except Exception:
        return 0.0


def has_audio(path):
    return probe(path, "stream=index", "a") != ""


def good_enough(path):
    if not path or not os.path.exists(path) or os.path.getsize(path) < 100_000:
        return False
    dur = get_duration(path)
    return dur == 0.0 or dur >= 2  # if ffprobe is missing, size check only


# ---------- core engine (your original, plus quality check) ----------
def _run_one(i, path, out_dir, gpu):
    final = f"{out_dir}/scene{i}.mp4"
    if os.path.exists(final):
        return True
    tmp = f"{out_dir}/tmp{i}"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    os.makedirs(f"{out_dir}/logs", exist_ok=True)
    env = dict(os.environ, MPLBACKEND="agg", CUDA_VISIBLE_DEVICES=str(gpu))
    start = time.time()
    print(f"Scene {i}: started on GPU {gpu}")
    with open(f"{out_dir}/logs/scene{i}.log", "w") as log:
        subprocess.run([PYTHON, "wgp.py", "--process", path, "--output-dir", tmp],
                       cwd=WGP_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    found = sorted(glob.glob(f"{tmp}/*.mp4"))
    if not found or not good_enough(found[0]):
        print(f"Scene {i}: FAILED (see logs/scene{i}.log)")
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    shutil.move(found[0], final)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Scene {i}: done in {int(time.time() - start)}s")
    return True


def make_scene_videos(settings_paths, out_dir="/kaggle/working/clips"):
    os.makedirs(out_dir, exist_ok=True)
    jobs = list(enumerate(settings_paths, 1))
    finals = [f"{out_dir}/scene{i}.mp4" for i, _ in jobs]

    # Scene 1 alone first, so the model downloads only once
    _run_one(jobs[0][0], jobs[0][1], out_dir, GPUS[0])

    # The rest run on both GPUs at the same time
    pool = queue.Queue()
    for g in GPUS:
        pool.put(g)

    def worker(job):
        gpu = pool.get()
        try:
            return _run_one(job[0], job[1], out_dir, gpu)
        finally:
            pool.put(gpu)

    with ThreadPoolExecutor(max_workers=len(GPUS)) as ex:
        list(ex.map(worker, jobs[1:]))

    # Retry failed scenes one by one
    for attempt in range(2):
        failed = [j for j in jobs if not os.path.exists(f"{out_dir}/scene{j[0]}.mp4")]
        if not failed:
            break
        print(f"Retry {attempt + 1}: scenes {[j[0] for j in failed]}")
        for i, path in failed:
            _run_one(i, path, out_dir, GPUS[0])

    missing = [f for f in finals if not os.path.exists(f)]
    if missing:
        raise RuntimeError(f"These scenes failed: {missing}")
    return finals


# ---------- film polish ----------
def run_ffmpeg(src, dst, vf, audio):
    dur = get_duration(src)
    fade_out = max(0.0, dur - 0.5)
    vf = vf + f",fade=t=in:st=0:d=0.4,fade=t=out:st={fade_out:.2f}:d=0.5"
    cmd = ["ffmpeg", "-y", "-i", src, "-vf", vf]
    if audio:
        cmd += ["-af", f"afade=t=in:st=0:d=0.3,afade=t=out:st={fade_out:.2f}:d=0.5,loudnorm",
                "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(dst)


def polish(src, dst):
    """Smooth motion, HD, film colour, fades. Falls back if a step fails."""
    audio = has_audio(src)
    look = ("scale=1080:1920:flags=lanczos,"
            "eq=contrast=1.06:saturation=1.12:gamma=0.98,"
            "unsharp=5:5:0.6:5:5:0.0,vignette=PI/5")
    smooth = "minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:vsbmc=1," + look
    if run_ffmpeg(src, dst, smooth, audio):
        print("[polish] smooth 48fps version made")
        return dst
    print("[polish] smoothing failed, using simple polish")
    if run_ffmpeg(src, dst, look, audio):
        return dst
    print("[polish] polish failed, using original video")
    return src


# ---------- cinematic single video ----------
def find_template():
    paths = glob.glob(f"{WGP_DIR}/**/test1.json", recursive=True)
    paths += glob.glob("/kaggle/working/test1.json")
    paths += glob.glob("/kaggle/working/*/test1.json")
    if not paths:
        raise FileNotFoundError("test1.json not found")
    return paths[0]


def make_video(scene_prompt, attempts=3):
    run_dir = f"{VIDEO_OUT}/run_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    with open(find_template()) as f:
        settings = json.load(f)
    settings["prompt"] = scene_prompt
    settings["negative_prompt"] = NEGATIVE
    settings["resolution"] = RESOLUTION

    for n in range(1, attempts + 1):
        settings["seed"] = random.randint(1, 10**9)
        path = f"{run_dir}/settings{n}.json"
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)
        print(f"[video] attempt {n}/{attempts}")
        if _run_one(n, path, run_dir, GPUS[0]):
            return polish(f"{run_dir}/scene{n}.mp4", f"{run_dir}/final.mp4")
    raise RuntimeError("Video was not made. Check the logs folder.")


if __name__ == "__main__":
    import prompt_generator
    post = prompt_generator.make_post()
    print(post["scene_prompt"])
    print("FINAL VIDEO:", make_video(post["scene_prompt"]))
