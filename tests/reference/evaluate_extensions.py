"""Generate original Python predictions; Rust evaluates these with its own metrics.

python tests/reference/evaluate_extensions.py TRAIN VALID OUTPUT.json
"""
import csv
import json
import sys
from collections import defaultdict
from operator import itemgetter
from generate_extensions import models


def read(path):
    sessions = defaultdict(list)
    with open(path) as source:
        for row in csv.DictReader(source, delimiter='\t'):
            sessions[int(row['SessionId'])].append((int(row['ItemId']), float(row['Time'])))
    return {sid: sorted(events, key=itemgetter(1)) for sid, events in sessions.items()}


train, valid = read(sys.argv[1]), read(sys.argv[2])
ns = models()
ns['itemgetter'] = itemgetter
postings = defaultdict(set)
for sid, events in train.items():
    for item, _ in events:
        postings[item].add(sid)
output = {}
for algorithm in ['ssknn', 'wsknn']:
    if algorithm == 'ssknn':
        model = ns['SeqContextKNN'](k=100, sample_size=500, sampling='recent')
        model.session_item_map = {sid: set(i for i, _ in events) for sid, events in train.items()}
        model.session_time = {sid: events[-1][1] for sid, events in train.items()}
    else:
        model = ns['WSKNN'].__new__(ns['WSKNN'])
        model.session_item_map = {sid: [list(i for i, _ in events), list(t for _, t in events)] for sid, events in train.items()}
        model.possible_neighbors_sample_size = 500
        model.weighting_function = 'linear'
        model.ranking_strategy = 'linear'
        model.return_events_from_session = True
    predictions = []
    for sid, events in sorted(valid.items()):
        items = [i for i, _ in events]
        for state in range(1, len(items)):
            current = items[max(0, state-10):state]
            candidates = set().union(*(postings[i] for i in current))
            if algorithm == 'ssknn':
                candidates = model.most_recent_sessions(candidates, 500)
                neighbors = model.calc_similarity(set(current), candidates)
                neighbors.sort(key=itemgetter(1), reverse=True)
                scores = model.score_items(neighbors[:100], current)
            else:
                candidates = model._sampling_recent(candidates)
                neighbors = model._calculate_similarity(current, candidates)
                neighbors.sort(key=itemgetter(1), reverse=True)
                scores = dict(model._rank_items(neighbors[:100], [current]))
            ranked = [i for i, _ in sorted(scores.items(), key=lambda p: (-p[1], p[0])) if i != current[-1]][:20]
            predictions.append([ranked, items[state:]])
    output[algorithm] = predictions
    print(algorithm, len(predictions), 'prediction points')
with open(sys.argv[3], 'w') as destination:
    json.dump(output, destination)
