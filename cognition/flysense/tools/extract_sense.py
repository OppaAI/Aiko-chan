"""Extract MaleCNS v1.0 sensory slices for Aiko's grafts (weight>=5).
1. AL: ORN + lLN/vLN + glomerular PNs -> per-glomerulus gain stats (JSON).
2. Motion: T4/T5 + HS/VS LPTCs kept as bodies; Mi/Tm fan-in aggregated.
3. DN: descending-neuron counts + output-target stats for gating priors.
"""
import json
import os
import re
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

ANNOT = '/tmp/opencode/body-annotations-male-cns-v1.0-minconf-0.5.feather'
WEIGHTS = '/tmp/opencode/sense-weights.feather'
OUT = '/tmp/opencode/sense_slice'
os.makedirs(OUT, exist_ok=True)

a = pd.read_feather(ANNOT)
t = a['type'].fillna('')
id2type = dict(zip(a['bodyId'], t))

def bodies(mask):
    return set(a.loc[mask, 'bodyId'])

ORN = bodies(t.str.startswith('ORN'))
LN = bodies(t.str.contains('LN') & ~t.str.contains('PN'))
ALPN = bodies(t.str.endswith('lPN') | t.str.endswith('adPN'))
T4 = bodies(t.str.startswith('T4'))
T5 = bodies(t.str.startswith('T5'))
LPTC = bodies(t.str.startswith('HS') | t.str.startswith('VS'))
DN = bodies(t.str.startswith('DN'))

print(f'ORN={len(ORN)} LN={len(LN)} ALPN={len(ALPN)} T4={len(T4)} T5={len(T5)} LPTC={len(LPTC)} DN={len(DN)}')

def glom(typ):
    m = re.match(r'^(?:ORN_|.*?[_\s])?([A-Z0-9]+[a-z]?\d*[a-z]?)_?(?:lPN|adPN)?$', str(typ))
    return m.group(1) if m else None

# --- 1. AL per-glomerulus stats ---
keep_al = sorted(ORN | LN | ALPN)
filt = (pc.field('body_pre').isin(keep_al)) & (pc.field('body_post').isin(keep_al))
al = ds.dataset(WEIGHTS, format='ipc').to_table(filter=filt).to_pandas()
al = al[al['weight'] >= 5]
al['tpre'] = al['body_pre'].map(id2type)
al['tpost'] = al['body_post'].map(id2type)
al['gpre'] = al['tpre'].map(glom)
al['gpost'] = al['tpost'].map(glom)
# simpler robust aggregates:
orn2pn = al[al['tpre'].str.startswith('ORN') & al['tpost'].str.endswith(('lPN', 'adPN'))]
ln2pn = al[(al['tpre'].str.contains('LN')) & (~al['tpre'].str.contains('PN')) & al['tpost'].str.endswith(('lPN', 'adPN'))]
ln2orn = al[(al['tpre'].str.contains('LN')) & (~al['tpre'].str.contains('PN')) & al['tpost'].str.startswith('ORN')]
per_glom = {}
pn_types = t[t.str.endswith('lPN') | t.str.endswith('adPN')].unique()
for pt in pn_types:
    g = glom(pt)
    if not g:
        continue
    e_op = orn2pn[orn2pn['tpost'] == pt]
    e_lp = ln2pn[ln2pn['tpost'] == pt]
    per_glom.setdefault(g, {'orn_syn': 0, 'ln_syn': 0, 'pn_types': set(), 'pn_bodies': set()})
    per_glom[g]['orn_syn'] += int(e_op['weight'].sum())
    per_glom[g]['ln_syn'] += int(e_lp['weight'].sum())
    per_glom[g]['pn_types'].add(pt)
    per_glom[g]['pn_bodies'].update(e_op['body_pre'].tolist() and a.loc[a['type'] == pt, 'bodyId'].tolist())
per_glom_out = {g: {'orn_syn': v['orn_syn'], 'ln_syn': v['ln_syn'],
                    'n_pn_types': len(v['pn_types']), 'n_pn_bodies': len(v['pn_bodies']),
                    'inhibition_ratio': round(v['ln_syn'] / max(1, v['orn_syn'] + v['ln_syn']), 4)}
                for g, v in sorted(per_glom.items())}
json.dump({'glomeruli': per_glom_out,
           'n_orn': len(ORN), 'n_ln': len(LN), 'n_alpn': len(ALPN),
           'provenance': 'male-cns:v1.0 minconf-0.5 w>=5, CC-BY Berg et al. 2025'},
          open(f'{OUT}/al_gains.json', 'w'), indent=1)
print('glomeruli:', len(per_glom_out))

# --- 2. Motion: keep T4/T5+LPTC bodies; aggregate Mi/Tm fan-in + T4/T5->LPTC ---
keep_mo = sorted(T4 | T5 | LPTC)
filt = (pc.field('body_pre').isin(keep_mo)) | (pc.field('body_post').isin(keep_mo))
mo = ds.dataset(WEIGHTS, format='ipc').to_table(filter=filt).to_pandas()
mo = mo[mo['weight'] >= 5]
mo['tpre'] = mo['body_pre'].map(id2type)
mo['tpost'] = mo['body_post'].map(id2type)
in_mo = set(keep_mo)
mo_in = mo[mo['body_post'].isin(in_mo) & ~mo['body_pre'].isin(in_mo)]
fanin = mo_in.groupby(mo_in['tpost'].map(lambda x: x[:2] if isinstance(x, str) and x[:2] in ('T4', 'T5') else 'other')).agg(
    pre_types=('tpre', 'nunique'), pre_bodies=('body_pre', 'nunique'),
    edges=('weight', 'size'), syn=('weight', 'sum'))
print(fanin.to_string())
t4lptc = mo[mo['tpre'].str.startswith(('T4', 'T5')) & mo['tpost'].isin(LPTC)]
lp = t4lptc.groupby(['tpre', 'tpost'])['weight'].sum()
lptc_gain = {}
for (pre, post), w in lp.items():
    lptc_gain.setdefault(post, {})[pre[:3]] = lptc_gain.setdefault(post, {}).get(pre[:3], 0) + int(w)
lptc_gain = {k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in lptc_gain.items()}
keep_edges = mo[mo['body_pre'].isin(in_mo) & mo['body_post'].isin(in_mo)][['body_pre', 'body_post', 'weight']]
keep_nodes = a[a['bodyId'].isin(in_mo)][['bodyId', 'type', 'instance', 'superclass', 'somaSide']]
keep_edges.to_parquet(f'{OUT}/motion_edges.parquet', index=False)
keep_nodes.to_parquet(f'{OUT}/motion_nodes.parquet', index=False)
json.dump({'fanin_by_target': fanin.to_dict(), 'lptc_gain': lptc_gain,
           'n_t4': len(T4), 'n_t5': len(T5), 'n_lptc': len(LPTC),
           'provenance': 'male-cns:v1.0 minconf-0.5 w>=5, CC-BY Berg et al. 2025'},
          open(f'{OUT}/motion_stats.json', 'w'), indent=1)

# --- 3. DN stats ---
dn_types = t[t.str.startswith('DN')].value_counts()
print('DN bodies:', int(t.str.startswith('DN').sum()), 'types:', len(dn_types))
json.dump({'n_dn': int(t.str.startswith('DN').sum()), 'n_dn_types': len(dn_types),
           'top_dn_types': dn_types.head(20).to_dict(),
           'provenance': 'male-cns:v1.0 annotations, CC-BY Berg et al. 2025'},
          open(f'{OUT}/dn_stats.json', 'w'), indent=1)
print('done')
