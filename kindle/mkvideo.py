# mkvideo.py RAWDIR OUTDIR   Turn a kindle/rec.sh recording into 30 fps PNG frames. The waits are sped up
# (SPEED) and every sped-up frame says so in the corner. Needs Pillow. Then:
#   ffmpeg -framerate 30 -i OUTDIR/o%05d.png -c:v libx264 -pix_fmt yuv420p -crf 20 kindle-writes.mp4
# Get the frames off the Kindle and unpack them first:
#   ssh ... 'cd /mnt/base-us/kwrec && tar cf - f*.gz times marks' | tar xf - -C RAWDIR
#   for f in RAWDIR/f*.gz; do { printf 'P5\n1072 1448\n255\n'; gzip -dc $f; } > ${f%.gz}.pgm; done
import sys, os, bisect
from PIL import Image, ImageDraw, ImageFont
raw, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
times = [(int(a), float(b)) for a, b in (l.split() for l in open(f"{raw}/times"))]
marks = dict((a, float(b)) for a, b in (l.split() for l in open(f"{raw}/marks")))
SPEED = {"title": 1, "kw": 6, "page": 0.6, "curve": 3, "proof": 1}
order = ["title", "kw", "page", "curve", "proof", "end"]
# Map output time -> real device time, segment by segment.
segs, t_out = [], 0.0
for a, b in zip(order, order[1:]):
    d = (marks[b] - marks[a]) / SPEED[a]
    segs.append((t_out, t_out + d, marks[a], SPEED[a])); t_out += d
TAIL, FPS = 2.0, 30
font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
real_ts = [t for _, t in times]
cache = {}
for i in range(int((t_out + TAIL) * FPS)):
    t = i / FPS
    seg = next((s for s in segs if s[0] <= t < s[1]), segs[-1])
    real = min(seg[2] + (t - seg[0]) * seg[3], marks["end"])
    n = times[max(0, bisect.bisect_right(real_ts, real) - 1)][0]
    key = (n, seg[3])
    if key not in cache:
        im = Image.open(f"{raw}/f{n:04d}.pgm").convert("L")
        if seg[3] > 1 and t < t_out:
            ImageDraw.Draw(im).text((1072 - 40, 1448 - 40), f"{seg[3]:g}x speed", font=font, fill=110, anchor="rb")
        cache = {key: im}
    cache[key].save(f"{out}/o{i:05d}.png")
print(f"{t_out + TAIL:.1f} s, {i + 1} frames")
