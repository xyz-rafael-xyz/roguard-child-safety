"""Encode deterministic 20fps PNG frames; requires imageio-ffmpeg."""
import argparse
import subprocess
from pathlib import Path
import imageio_ffmpeg

p = argparse.ArgumentParser()
p.add_argument('frames', type=Path)
p.add_argument('--output', type=Path, default=Path('media'))
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
source = ['-framerate', '20', '-i', str(a.frames / '%04d.png')]
def run(arguments):
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y'] + arguments, check=True)
run(source + ['-vf', 'scale=1280:720:flags=lanczos', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(a.output / 'roguard-v9-demo.mp4')])
run(source + ['-filter_complex', 'scale=960:540:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle', '-loop', '0', str(a.output / 'roguard-v9-demo-loop.gif')])
