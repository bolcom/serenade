"""Generate scoring fixtures by executing unmodified upstream class methods.

Download the four pinned sources listed in README.md alongside this script first.
Only class definitions are loaded, avoiding pandas/numpy dependencies; scoring
methods themselves are not translated or patched.
"""
import ast
import math
import time
from types import SimpleNamespace
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
SEQUENCES = [[1, 2, 1, 4], [2, 3, 2, 5], [1, 3, 6], [1, 2, 1, 4]]
TIMES = [100, 86500, 172900, 100]
CURRENT = [1, 2, 1, 3]
NOW = 216100
idf = {item: math.log(len(SEQUENCES) / count)
       for item, count in Counter(i for s in SEQUENCES for i in s).items()}
rows = []
for name in ['sknn', 'vsknn', 'stan', 'vstan']:
    tree = ast.parse((ROOT / (name + '.py')).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    namespace = dict(time=SimpleNamespace(clock=time.perf_counter), sqrt=math.sqrt, exp=math.exp, log10=math.log10)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), name + '.py', 'exec'), namespace)
    model_type = namespace[cls.name]
    options = [{}]
    if name == 'sknn':
        options += [dict(similarity=s) for s in ['cosine', 'binary', 'tanimoto']]
        options += [dict(remind=True), dict(pop_boost=2)]
    if name == 'vsknn':
        options += [dict(weighting=w, weighting_score=w) for w in ['same', 'linear', 'log', 'quadratic']]
        options += [dict(similarity='cosine'), dict(weighting_time=True),
                    dict(idf_weighting=2), dict(idf_weighting_session=True), dict(remind=False)]
    if name in ['stan', 'vstan']:
        options += [dict(lambda_spw=None, lambda_snh=None, lambda_inh=None), dict(remind=False)]
    if name == 'vstan':
        options += [dict(lambda_ipw=None, lambda_idf=None)]
    for kwargs in options:
        for k in [2, 10]:
            model = model_type(k=k, **{key: value for key, value in kwargs.items() if key != 'lambda_snh' or value is not None})
            if 'lambda_snh' in kwargs and kwargs['lambda_snh'] is None:
                model.lambda_snh = None
            model.session_item_map = {i: (s if name in ['stan', 'vstan'] else set(s)) for i, s in enumerate(SEQUENCES)}
            model.session_time = dict(enumerate(TIMES))
            model.idf = idf
            if name == 'sknn':
                neighbors = model.calc_similarity(set(CURRENT), list(range(4)))
            elif name == 'vsknn':
                neighbors = model.calc_similarity(CURRENT, list(range(4)), [], NOW)
            else:
                neighbors = model.calc_similarity(CURRENT, list(range(4)), NOW)
            neighbors = sorted(neighbors, key=lambda n: n[1], reverse=True)[:k]
            scores = model.score_items(neighbors) if name == 'sknn' else model.score_items(neighbors, CURRENT, NOW)
            if name == 'sknn':
                # Execute the original predict_next post-scoring blocks verbatim.
                predict = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'predict_next')
                for node in predict.body:
                    if isinstance(node, ast.If) and ast.unparse(node.test) in ['self.remind', 'self.pop_boost > 0']:
                        model.session_items = CURRENT
                        scope = dict(self=model, scores=scores, neighbors=neighbors)
                        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), name + '.py', 'exec'), namespace, scope)
                        scores = scope['scores']
            def attr(key, default):
                v = getattr(model, key, default)
                if key == 'idf_weighting' and isinstance(v, bool): v = int(v)
                if key == 'lambda_snh' and v is not None: v /= 86400
                return 'none' if v is None else str(v).lower()
            fields = [name] + [attr(key, default) for key, default in [
                ('similarity', 'cosine'), ('weighting', 'div'), ('weighting_score', 'div'),
                ('weighting_time', False), ('idf_weighting', 0), ('idf_weighting_session', False),
                ('remind', True), ('lambda_spw', None), ('lambda_snh', None),
                ('lambda_inh', None), ('lambda_ipw', None), ('lambda_idf', None)]]
            fields += [str(k), ','.join(map(str, CURRENT)),
                       ','.join(f'{i}:{v:.17g}' for i, v in neighbors),
                       ','.join(f'{i}:{v:.17g}' for i, v in sorted(scores.items()))]
            fields += [attr('pop_boost', 0)]
            # Canonical score order: ties use item ID only for deterministic fixture output.
            fields += [','.join(str(i) for i, _ in sorted(scores.items(), key=lambda p: (-p[1], p[0])) if i != CURRENT[-1])]
            rows.append('\t'.join(fields))
(ROOT / 'scores.tsv').write_text('\n'.join(rows) + '\n')
print(f'Generated {len(rows)} scoring cases')
