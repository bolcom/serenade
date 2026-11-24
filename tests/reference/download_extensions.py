"""Download pinned validation-only sources; no third-party Python packages needed."""
from pathlib import Path
from urllib.request import urlopen

ROOT = Path('target/reference-upstream')
ROOT.mkdir(parents=True, exist_ok=True)
SOURCES = {
    'scknn.py': 'https://raw.githubusercontent.com/rn5l/session-rec/5dcd583cbd8d44703a5248b9a308945f24b91390/backup/algorithms/aware_backup/knn_aware/scknn.py',
}
for path in ['wsknn/model/wsknn.py', 'wsknn/weighting/session_weighting.py',
             'wsknn/weighting/item_weighting.py', 'wsknn/weighting/weighting.py',
             'LICENSE.md']:
    SOURCES['wsknn-' + path.replace('/', '-')] = (
        'https://raw.githubusercontent.com/nokaut/wsknn/'
        'bd06b567a15df256503ed3aa0de607712ab912e3/' + path)
SOURCES['wsknn-calc.py'] = ('https://raw.githubusercontent.com/nokaut/wsknn/'
                           'bd06b567a15df256503ed3aa0de607712ab912e3/wsknn/utils/calc.py')
for name, url in SOURCES.items():
    with urlopen(url) as response:
        (ROOT / name).write_bytes(response.read())
    print(name)
