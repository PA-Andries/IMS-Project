"""Make a fully self-contained copy of the slide deck.

Embeds the chart images (and Chart.js) directly into the HTML, so the deck
works anywhere: preview panes, a browser opened from any folder, e-mail, and
offline PDF export.

    python build_deck.py
    -> creates anomaly-deck-standalone.html
"""
import base64
import mimetypes
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "anomaly-deck.html"
OUT = HERE / "anomaly-deck-standalone.html"

html = SRC.read_text(encoding="utf-8")

# 1) Inline every local PNG (src="outputs/....png") as a base64 data URI.
def embed_img(m):
    rel = m.group(2)
    path = HERE / rel
    if not path.exists():
        print("  ! missing:", rel)
        return m.group(0)
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    print(f"  embedded {rel} ({path.stat().st_size // 1024} KB)")
    return f'{m.group(1)}data:{mime};base64,{data}{m.group(3)}'

html = re.sub(r'(src=")(outputs/[^"]+\.png)(")', embed_img, html)

# 2) Inline Chart.js so Fig. 3 also works offline (best-effort).
m = re.search(r'<script src="(https://[^"]*chart[^"]*)"></script>', html, re.I)
if m:
    try:
        js = urllib.request.urlopen(m.group(1), timeout=15).read().decode("utf-8")
        html = html.replace(m.group(0), f"<script>{js}</script>")
        print("  inlined Chart.js")
    except Exception as e:
        print("  (kept Chart.js on CDN - needs internet):", e)

OUT.write_text(html, encoding="utf-8")
print(f"Wrote {OUT.name} ({OUT.stat().st_size // 1024} KB)")
