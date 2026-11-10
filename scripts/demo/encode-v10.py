"""Encode the deterministic v10 animation frames (50 fps) to MP4, GIF, and poster."""
import argparse
import shutil
import subprocess
from pathlib import Path
import imageio_ffmpeg

parser=argparse.ArgumentParser()
parser.add_argument('frames',type=Path)
parser.add_argument('--output',type=Path,default=Path('media'))
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
source=['-framerate','50','-i',str(args.frames/'%04d.png')]
def run(extra):
 subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y']+source+extra,check=True)
# MP4 keeps every frame: 50 fps for smooth springs and fades.
run(['-vf','scale=1280:720:flags=lanczos','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-r','50','-movflags','+faststart',str(args.output/'roguard-v10-demo.mp4')])
# GIF takes every second frame: 25 fps is an exact 4 cs delay, so timing stays even.
run(['-filter_complex','select=not(mod(n\\,2)),setpts=N/25/TB,scale=960:540:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle','-r','25','-loop','0',str(args.output/'roguard-v10-demo-loop.gif')])
gif=args.output/'roguard-v10-demo-loop.gif'
if shutil.which('gifsicle'):
 # light lossy pass keeps the GIF well under GitHub's inline-image comfort zone
 subprocess.run(['gifsicle','-O3','--lossy=30','-b',str(gif)],check=True)
shutil.copyfile(args.frames/'0140.png',args.output/'roguard-v10-poster.png')
print('Encoded v10 MP4, GIF, and poster')
