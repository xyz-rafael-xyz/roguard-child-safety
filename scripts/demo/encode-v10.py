"""Encode the deterministic v10 animation frames to MP4, GIF, and poster."""
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
source=['-framerate','20','-i',str(args.frames/'%04d.png')]
def run(extra):
 subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y']+source+extra,check=True)
run(['-vf','scale=1280:720:flags=lanczos','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output/'roguard-v10-demo.mp4')])
run(['-filter_complex','scale=960:540:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle','-loop','0',str(args.output/'roguard-v10-demo-loop.gif')])
shutil.copyfile(args.frames/'0032.png',args.output/'roguard-v10-poster.png')
print('Encoded v10 MP4, GIF, and poster')
