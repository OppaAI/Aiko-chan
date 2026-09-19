"""Extract the MaleCNS v1.0 central-complex subgraph for Aiko's compass layer.
Core ring: EPG/EPGt (compass), PEN (angular velocity), PEG, PFN/PFL (FB steering),
ExR/ER (ring inputs, incl. ER5/R5 sleep-drive proxy), + empirical top EPG inputs.
Threshold weight>=5 (paper standard). Output: /tmp/opencode/cx_slice/ (KBs).
"""
import json
import os
import re
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

ANNOT = '/tmp/opencode/body-annotations-male-cns-v1.0-minconf-0.5.feather'
WEIGHTS = '/tmp/opencode/cx-weights.feather'
OUT = '/tmp/opencode/cx_slice'

a = pd.read_feather(ANNOT)
t = a['type'].fillna('')
roles = {}
roles['EPG'] = set(a.loc[t.str.startswith('EPG'), 'bodyId'])
roles['PEN'] = set(a.loc[t.str.startswith('PEN'), 'bodyId'])
roles['PEG'] = set(a.loc[t.str.startswith('PEG'), 'bodyId'])
pf = t.str.startswith('PFN') | t.str.startswith('PFL') | t.str.startswith('PFd') | t.str.startswith('PFv') | t.str.startswith('PFm')
roles['PF'] = set(a.loc[pf, 'bodyId'])
ring = t.str.startswith('ExR') | ((t.str.startswith('ER')) & (~t.str.startswith('ERROR')))
roles['RING'] = set(a.loc[ring, 'bodyId'])
cx_ids = sorted(set().union(*roles.values()))
print('CX bodies:', {k: len(v) for k, v in roles.items()}, 'total:', len(cx_ids))

filt = (pc.field('body_pre').isin(cx_ids)) | (pc.field('body_post').isin(cx_ids))
edges = ds.dataset(WEIGHTS, format='ipc').to_table(filter=filt).to_pandas()
print('CX-touching edges:', len(edges))

# Empirical top inputs to EPG (visual/feature drive discovery)
epg_in = edges[edges['body_post'].isin(roles['EPG']) & ~edges['body_pre'].isin(set(cx_ids))]
by_type = epg_in.merge(a[['bodyId', 'type']].rename(columns={'bodyId': 'body_pre', 'type': 'pre_type'}),
                       on='body_pre', how='left')
agg = by_type.groupby('pre_type')['weight'].sum().sort_values(ascending=False)
print('top EPG input types:'); print(agg.head(15).to_string())
roles['INPUT'] = set(a.loc[a['type'].isin(agg.head(40).index), 'bodyId'])

os.makedirs(OUT, exist_ok=True)
id2role = {}
for role, ids in roles.items():
    for i in ids:
        id2role.setdefault(i, role)
nodes = a[a['bodyId'].isin(set().union(*roles.values()))][
    ['bodyId', 'type', 'instance', 'superclass', 'somaSide']].copy()
nodes['cx_role'] = nodes['bodyId'].map(id2role)
nodes.to_parquet(f'{OUT}/cx_nodes.parquet', index=False)
node_ids = set(nodes['bodyId'])
keep = edges[edges['body_pre'].isin(node_ids) & edges['body_post'].isin(node_ids)
             & (edges['weight'] >= 5)].copy()
keep['role_pre'] = keep['body_pre'].map(id2role)
keep['role_post'] = keep['body_post'].map(id2role)
keep.to_parquet(f'{OUT}/cx_edges.parquet', index=False)

# Real wedge positions from EPG instance suffixes (_R1..8/_L1..8).
def wedge(inst):
    m = re.search(r'_([RL])(\d+)\s*$', str(inst or ''))
    if not m:
        return None
    side, k = m.group(1), int(m.group(2))
    return (k - 1) + (8 if side == 'L' else 0)  # 16 wedges around the ring
epg = nodes[nodes['cx_role'] == 'EPG'].copy()
epg['wedge'] = epg['instance'].map(wedge)
print('EPG wedge coverage:', epg['wedge'].value_counts().sort_index().to_dict())
print('EPG without wedge parse:', int(epg['wedge'].isna().sum()))

def mat(r1, r2):
    sub = keep[(keep['role_pre'] == r1) & (keep['role_post'] == r2)]
    return {'edges': int(len(sub)), 'synapses': int(sub['weight'].sum()),
            'mean_w': float(sub['weight'].mean()) if len(sub) else 0.0}
summary = {
    'provenance': {
        'dataset': 'male-cns:v1.0 (minconf-0.5, weight>=5)', 'paper': 'Berg et al. 2025',
        'license': 'CC-BY', 'portal': 'https://male-cns.janelia.org/download/',
        'note': 'Slice: EPG/PEN/PEG/PF/RING bodies + top-40 EPG input types. '
                'No dFB-annotated sleep neurons in v1.0 types; ER5/R5 used as '
                'sleep-drive proxy (Liu et al. 2016). Full weights sliced locally.',
    },
    'nodes': {k: len(v) for k, v in roles.items()},
    'mats': {'ER_RING->EPG': mat('RING', 'EPG'), 'PEN->EPG': mat('PEN', 'EPG'),
             'PEG->EPG': mat('PEG', 'EPG'), 'EPG->EPG': mat('EPG', 'EPG'),
             'EPG->PF': mat('EPG', 'PF'), 'PF->PF': mat('PF', 'PF'),
             'INPUT->EPG': mat('INPUT', 'EPG'), 'RING->RING': mat('RING', 'RING')},
}
json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=2)
print(json.dumps(summary, indent=2))
