"""Fail deployment early if required portable assets were not committed."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
for name in ('models/hog_svm.npz', 'assets/DejaVuSans.ttf', 'assets/FONT_LICENSE.txt', 'public/index.html', 'public/app.js', 'public/style.css'):
    path = root / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit('Missing deployment asset: ' + name)
print('Deployment assets OK')
