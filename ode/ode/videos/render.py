"""
Render whole episodes of the MATH 331 video series and collect videos and subtitles.

Usage (from this folder, in an environment where Manim is installed)::

    python render.py ep01               # 1080p60 (default)
    python render.py ep01 ep02 -q l     # 480p15 previews, much faster
    python render.py all                # every episode

Video (.mp4) and commentary subtitles (.srt) are copied to ``output/``. Each episode is
rendered in its own temporary media folder, so several renders can run in parallel safely
(sharing one folder lets two LaTeX compilations collide). Any failure stops the script.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
EPISODES = {                                  # key -> (source file, scene with the whole episode)
    "ep01": ("ep01_growth_decay.py", "Episode01"),
    "ep02": ("ep02_logistic.py", "Episode02"),
    "ep03": ("ep03_euler.py", "Episode03"),
}
QUALITY_DIRS = {"l": "480p15", "m": "720p30", "h": "1080p60", "k": "2160p60"}


def render(episode: str, quality: str) -> None:
    source, scene = EPISODES[episode]
    exe = pathlib.Path(sys.executable).with_name("manim")          # the manim next to this Python
    launcher = [str(exe)] if exe.is_file() else [sys.executable, "-m", "manim"]
    with tempfile.TemporaryDirectory() as media_dir:
        cmd = [*launcher, "render", f"-q{quality}", "--progress_bar", "none", "-v", "WARNING",
               "--media_dir", media_dir, str(HERE / source), scene]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        folder = pathlib.Path(media_dir) / "videos" / pathlib.Path(source).stem / QUALITY_DIRS[quality]
        OUTPUT_DIR.mkdir(exist_ok=True)
        for suffix in (".mp4", ".srt"):
            produced = folder / f"{scene}{suffix}"
            if not produced.is_file():
                raise FileNotFoundError(f"Manim finished but {produced} was not produced")
            target = OUTPUT_DIR / f"{episode}_{QUALITY_DIRS[quality]}{suffix}"
            shutil.copyfile(produced, target)
            print(f"-> {target.relative_to(HERE)} ({target.stat().st_size / 1e6:.2f} MB)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes", nargs="+", choices=[*sorted(EPISODES), "all"])
    parser.add_argument("-q", "--quality", default="h", choices=sorted(QUALITY_DIRS),
                        help="l=480p15, m=720p30, h=1080p60 (default), k=2160p60")
    args = parser.parse_args()
    episodes = sorted(EPISODES) if "all" in args.episodes else args.episodes
    for episode in episodes:
        render(episode, args.quality)
    return 0


if __name__ == "__main__":
    sys.exit(main())
