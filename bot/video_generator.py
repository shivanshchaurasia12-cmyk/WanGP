import os, glob, subprocess, shutil, time, queue
from concurrent.futures import ThreadPoolExecutor

PYTHON = "/kaggle/working/env310/bin/python"
WGP_DIR = "/kaggle/working/WanGP"
GPUS = [0, 1]


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
    if not found:
        print(f"Scene {i}: FAILED (see logs/scene{i}.log)")
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
