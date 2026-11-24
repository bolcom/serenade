"""Execute pinned upstream methods without installing their optional dependencies.

Sources: target/reference-upstream; see docs/session-knn-validation.md.
Run from repository root: python tests/reference/generate_extensions.py
"""
import ast
import math
import typing
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('target/reference-upstream')
SEQUENCES = [[1, 2, 1, 4], [2, 3, 2, 5], [1, 3, 6], [1, 2, 1, 4]]


def load(path, namespace, kind):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, kind)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)


def models():
    ns = dict(vars(typing), np=SimpleNamespace(ndarray=list), log10=math.log10,
              sqrt=math.sqrt, time=SimpleNamespace(clock=time.perf_counter))
    load('scknn.py', ns, ast.ClassDef)
    for file in ['wsknn-wsknn-weighting-session_weighting.py',
                 'wsknn-wsknn-weighting-item_weighting.py',
                 'wsknn-wsknn-weighting-weighting.py', 'wsknn-calc.py']:
        load(file, ns, ast.FunctionDef)
    load('wsknn-wsknn-model-wsknn.py', ns, ast.ClassDef)
    return ns


if __name__ == '__main__':
    ns = models()
    rows = []
    for algorithm in ['ssknn', 'wsknn']:
        for weighting in (['same', 'linear', 'div', 'log', 'quadratic']
                          if algorithm == 'ssknn' else ['linear', 'log', 'quadratic']):
            for ranking in (['div'] if algorithm == 'ssknn' else ['linear', 'inv', 'log', 'quadratic']):
                for current in [[1, 2, 1, 3], [1], [999, 1], [1, 999]]:
                    if algorithm == 'ssknn':
                        model = ns['SeqContextKNN'](k=10, weighting=weighting)
                        model.session_item_map = dict(enumerate(map(set, SEQUENCES)))
                        neighbors = model.calc_similarity(set(current), list(range(4)))
                        scores = model.score_items(neighbors, current)
                    else:
                        model = ns['WSKNN'].__new__(ns['WSKNN'])
                        model.weighting_function = weighting
                        model.ranking_strategy = ranking
                        model.return_events_from_session = True
                        model.session_item_map = {i: [s, []] for i, s in enumerate(SEQUENCES)}
                        neighbors = model._calculate_similarity(current, list(range(4)))
                        neighbors = [n for n in neighbors if n[1] > 0]
                        scores = dict(model._rank_items(neighbors, [current]))
                    neighbors = sorted(neighbors, key=lambda p: -p[1])
                    fields = [algorithm, 'jaccard' if algorithm == 'ssknn' else 'vec',
                              weighting, 'div' if ranking == 'inv' else ranking,
                              'false', '0', 'false', str(algorithm == 'wsknn').lower(),
                              *(['none'] * 5), '10', ','.join(map(str, current)),
                              ','.join(f'{i}:{v:.17g}' for i, v in neighbors),
                              ','.join(f'{i}:{v:.17g}' for i, v in sorted(scores.items())), '0',
                              ','.join(str(i) for i, v in sorted(scores.items(), key=lambda p: (-p[1], p[0])) if i != current[-1])]
                    rows.append('\t'.join(fields))
    Path('tests/reference/extensions.tsv').write_text('\n'.join(rows) + '\n')
    print(f'Generated {len(rows)} upstream cases')
