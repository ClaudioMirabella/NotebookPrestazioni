"""
Render a whole episode of the waves series and collect the video and its subtitles.

Usage (from inside this folder, in an environment where Manim is installed)::

    python render.py ep01            # 1080p60 (default)
    python render.py ep01 -q l       # 480p15 preview, much faster

The video and the subtitle file (.srt, the commentary with timings) are copied to
``videos/``. Any failure stops the script with a non-zero exit code.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "videos"
# episode key -> (source file, scene with the whole episode)
EPISODES = {
    "ep01": ("ep01_vectors_complex_wavefunctions.py", "Episode01"),
}
QUALITY_DIRS = {"l": "480p15", "m": "720p30", "h": "1080p60", "k": "2160p60"}


def render(episode: str, quality: str) -> None:
    source, scene = EPISODES[episode]
    exe = pathlib.Path(sys.executable).with_name("manim")   # the manim next to this Python
    launcher = [str(exe)] if exe.is_file() else [sys.executable, "-m", "manim"]
    with tempfile.TemporaryDirectory() as media_dir:
        cmd = [*launcher, "render", f"-q{quality}", "--progress_bar", "none", "-v", "WARNING",
               "--media_dir", media_dir, str(HERE / source), scene]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        folder = pathlib.Path(media_dir) / "videos" / pathlib.Path(source).stem / QUALITY_DIRS[quality]
        OUTPUT_DIR.mkdir(exist_ok=True)
        for suffix in (".mp4", ".srt"):
            produced = folder / f"{scene}{suffix}"
            if not produced.is_file():
                raise FileNotFoundError(f"Manim finished but {produced} was not produced")
            target = OUTPUT_DIR / f"{episode}_{QUALITY_DIRS[quality]}{suffix}"
            shutil.copyfile(produced, target)
            print(f"-> {target.relative_to(HERE)} ({target.stat().st_size / 1e6:.2f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", choices=sorted(EPISODES))
    parser.add_argument("-q", "--quality", default="h", choices=sorted(QUALITY_DIRS),
                        help="l=480p15, m=720p30, h=1080p60 (default), k=2160p60")
    args = parser.parse_args()
    render(args.episode, args.quality)
    return 0


if __name__ == "__main__":
    sys.exit(main())
