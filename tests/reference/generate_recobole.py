"""Generate parity fixtures using unmodified, pinned RecBole metric methods.

Requires NumPy. Run from the repository root. Downloads metrics.py into .baseline.
Only the four metric classes are AST-loaded; PyTorch/RecBole are not required.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import numpy as np

COMMIT = "2d7158418d1181cd2b9f4fca94923d0aa96bf2ad"
URL = f"https://raw.githubusercontent.com/RUCAIBox/RecBole/{COMMIT}/recbole/evaluator/metrics.py"
source = urllib.request.urlopen(URL).read()
cache = Path(".baseline/recobole_metrics.py")
cache.parent.mkdir(exist_ok=True)
cache.write_bytes(source)
# RecBole still uses the removed np.float alias. Restore it without changing methods.
np.float = float
scope = dict(np=np, Counter=Counter, AbstractMetric=object, TopkMetric=object,
             EvaluatorType=SimpleNamespace(RANKING="ranking"))
classes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
           and node.name in {"MAP", "GiniIndex", "ShannonEntropy", "TailPercentage"}]
exec(compile(ast.Module(body=classes, type_ignores=[]), URL, "exec"), scope)
metrics = {name: cls.__new__(cls) for name, cls in scope.items() if name in
           {"MAP", "GiniIndex", "ShannonEntropy", "TailPercentage"}}
metrics["TailPercentage"].tail = 0.1

fixtures = []
for name, counts, predictions, targets in [
    ("unequal_exposure", [1, 1, 2, 3, 5, 8, 8, 9, 10, 11],
     [[1, 3, 5, 7], [1, 2, 3, 4], [1, 5, 8, 9]], [[3, 7, 9], [2], [1, 2, 8, 9, 10]]),
    ("uniform_exposure", [1, 1, 1, 1],
     [[1, 2, 3, 4], [4, 3, 2, 1]], [[2, 4], [1, 3]]),
    ("single_exposed_item", [1, 3, 4], [[2], [2], [2]], [[2], [1], [1, 3]]),
    ("twenty_items_tied_tail", [1] * 20,
     [[1, 2, 3, 4], [2, 1, 5, 6], [20, 19, 18, 17]], [[1, 3], [7], [20]]),
]:
    training = [i for i, count in enumerate(counts, 1) for _ in range(count)]
    matrix = np.array(predictions)
    positive = np.array([[i in set(t) for i in row] for row, t in zip(predictions, targets)])
    lengths = np.array([len(set(t)) for t in targets])
    ap = metrics["MAP"].metric_info(positive, lengths).mean(axis=0)
    tail = metrics["TailPercentage"]
    tail_values = tail.metric_info(tail.get_tail(matrix, Counter(training))).mean(axis=0)
    for k in range(1, matrix.shape[1] + 1):
        fixtures.append(dict(name=f"{name}@{k}", training=training,
                             recommendations=predictions, targets=targets, k=k,
                             expected=[float(ap[k-1]),
                                       float(metrics["GiniIndex"].get_gini(matrix[:, :k], len(counts))),
                                       float(metrics["ShannonEntropy"].get_entropy(matrix[:, :k])),
                                       float(tail_values[k-1])]))
Path("tests/reference/recobole.json").write_text(json.dumps(
    dict(source=URL, sha256=hashlib.sha256(source).hexdigest(), fixtures=fixtures), indent=2) + "\n")
print(f"Wrote {len(fixtures)} fixtures from RecBole {COMMIT}")
