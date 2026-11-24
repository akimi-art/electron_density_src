# 必要なら最初に実行
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from pathlib import Path
from io import BytesIO
import random
import requests
from astropy.io import fits

base_url = "https://das.sdss.org/spectro/1d_26/0644/1d/"
out_dir = Path("sdss_dr7_plate0644_galaxies")
out_dir.mkdir(exist_ok=True)

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

# ディレクトリ一覧を取得
response = requests.get(base_url)
response.raise_for_status()

parser = LinkParser()
parser.feed(response.text)

files = [
    urljoin(base_url, link)
    for link in parser.links
    if Path(urlparse(link).path).name.startswith("spSpec-")
    and Path(urlparse(link).path).suffix.lower() in {".fit", ".fits"}
]

# ファイル順に偏らないよう、候補をシャッフル
random.seed(42)
random.shuffle(files)

session = requests.Session()
saved = 0

for url in files:
    r = session.get(url)
    if r.status_code != 200:
        continue

    try:
        with fits.open(BytesIO(r.content), memmap=False) as hdul:
            spec_class = int(hdul[0].header.get("SPEC_CLN", -1))
    except Exception:
        continue

    if spec_class == 2:  # SPEC_GALAXY
        filename = Path(urlparse(url).path).name
        (out_dir / filename).write_bytes(r.content)
        saved += 1
        print(f"{saved:3d}/100  {filename}")

        if saved == 100:
            break

print(f"保存数: {saved}  保存先: {out_dir.resolve()}")