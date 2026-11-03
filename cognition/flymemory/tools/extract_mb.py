"""Extract the MaleCNS v1.0 mushroom-body subgraph (KC/DAN/MBON/APL/DPM + top KC inputs).
Sources (CC-BY, Berg et al. 2025): https://male-cns.janelia.org/download/
  body-annotations-male-cns-v1.0-minconf-0.5.feather (14MB)
  connectome-weights-male-cns-v1.0-minconf-0.5.feather (1.05GB, deleted after slicing)
Output: /tmp/opencode/mb_slice/ (nodes, edges, summary.json) — KBs.
"""
import json
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

ANNOT = '/tmp/opencode/body-annotations-male-cns-v1.0-minconf-0.5.feather'
WEIGHTS = '/tmp/opencode/connectome-weights-male-cns-v1.0-minconf-0.5.feather'
OUT = '/tmp/opencode/mb_slice'

a = pd.read_feather(ANNOT)
t = a['type'].fillna('')
roles = {}
roles['KC'] = set(a.loc[t.str.startswith('KC'), 'bodyId'])
roles['MBON'] = set(a.loc[t.str.startswith('MBON'), 'bodyId'])
dan = t.str.startswith('PAM') | t.str.startswith('PPL1') | t.str.startswith('DAN-')
roles['DAN'] = set(a.loc[dan, 'bodyId'])
roles['APLDPM'] = set(a.loc[t.isin(['APL', 'DPM']), 'bodyId'])
mb_ids = sorted(set().union(*roles.values()))
print('MB bodies:', {k: len(v) for k, v in roles.items()}, 'total:', len(mb_ids))

# MB-touching edges via pyarrow filter pushdown (avoids full 151M-row pandas load)
filt = (pc.field('body_pre').isin(mb_ids)) | (pc.field('body_post').isin(mb_ids))
edges = ds.dataset(WEIGHTS, format='ipc').to_table(filter=filt).to_pandas()
print('MB-touching edges:', len(edges))
mbset = set(mb_ids)
mb_edges = edges[edges['body_pre'].isin(mbset) & edges['body_post'].isin(mbset)].copy()
print('MB-internal edges:', len(mb_edges))

# Top presynaptic partners of KCs (empirical PN/input discovery, no name guessing)
kc_in = edges[edges['body_post'].isin(roles['KC']) & ~edges['body_pre'].isin(mbset)]
by_type = kc_in.merge(a[['bodyId', 'type']].rename(columns={'bodyId': 'body_pre', 'type': 'pre_type'}),
                      on='body_pre', how='left')
agg = by_type.groupby('pre_type')['weight'].sum().sort_values(ascending=False)
print('top KC input types:'); print(agg.head(15).to_string())
top_in_types = list(agg.head(40).index)
top_in_ids = set(a.loc[a['type'].isin(top_in_types), 'bodyId'])
roles['INPUT'] = top_in_ids
print('INPUT bodies:', len(top_in_ids))

import os
os.makedirs(OUT, exist_ok=True)
id2role = {}
for role, ids in roles.items():
    for i in ids:
        id2role.setdefault(i, role)
nodes = a[a['bodyId'].isin(set().union(*roles.values()))][
    ['bodyId', 'type', 'instance', 'superclass', 'somaSide']].copy()
nodes['mb_role'] = nodes['bodyId'].map(id2role)
nodes.to_parquet(f'{OUT}/mb_nodes.parquet', index=False)
node_ids = set(nodes['bodyId'])
keep = edges[edges['body_pre'].isin(node_ids) &
             edges['body_post'].isin(node_ids) &
             (edges['weight'] >= 5)].copy()  # paper-standard threshold: drops minconf-0.5 noise
keep['role_pre'] = keep['body_pre'].map(id2role)
keep['role_post'] = keep['body_post'].map(id2role)
keep.to_parquet(f'{OUT}/mb_edges.parquet', index=False)

def mat(r1, r2):
    sub = keep[(keep['role_pre'] == r1) & (keep['role_post'] == r2)]
    return {'edges': int(len(sub)), 'synapses': int(sub['weight'].sum()),
            'mean_w': float(sub['weight'].mean()) if len(sub) else 0.0}
summary = {
    'provenance': {
        'dataset': 'male-cns:v1.0 (minconf-0.5)', 'paper': 'Berg et al. 2025',
        'license': 'CC-BY', 'portal': 'https://male-cns.janelia.org/download/',
        'repos': ['flyconnectome/2025malecns', 'connectome-neuprint/neuprint-python',
                  'natverse/malecns (R reference)', 'janelia-flyem/male-cns'],
        'note': 'Slice: KC/MBON/DAN/APL/DPM bodies + top-40 KC input types, weight>=5. '
                'Full 1.05GB weights file sliced locally, not shipped.',
    },
    'nodes': {k: len(v) for k, v in roles.items()},
    'mats': {'KC->MBON': mat('KC', 'MBON'), 'DAN->MBON': mat('DAN', 'MBON'),
             'DAN->KC': mat('DAN', 'KC'), 'APLDPM->KC': mat('APLDPM', 'KC'),
             'INPUT->KC': mat('INPUT', 'KC'), 'KC->KC': mat('KC', 'KC'),
             'MBON->MBON': mat('MBON', 'MBON'), 'MBON->DAN': mat('MBON', 'DAN')},
}
json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=2)
print(json.dumps(summary, indent=2))
