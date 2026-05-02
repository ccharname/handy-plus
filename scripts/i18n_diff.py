#!/usr/bin/env python3
"""Compare locale JSON files against en SSOT, report missing keys."""
import json, os, sys
from pathlib import Path

LOCALES_DIR = Path('/Users/zhengma/Developer/handy/src/i18n/locales')
SSOT = 'en'

def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out

ssot_data = flatten(json.load(open(LOCALES_DIR / SSOT / 'translation.json')))
ssot_keys = set(ssot_data.keys())

print(f'# i18n diff report (SSOT: {SSOT}, total keys: {len(ssot_keys)})')
print()
print('| Locale | Missing | Extra | Coverage |')
print('|---|---|---|---|')

for loc_dir in sorted(LOCALES_DIR.iterdir()):
    if loc_dir.name == SSOT or not loc_dir.is_dir():
        continue
    json_file = loc_dir / 'translation.json'
    if not json_file.exists():
        print(f'| {loc_dir.name} | (no translation.json) | - | 0% |')
        continue
    loc_data = flatten(json.load(open(json_file)))
    loc_keys = set(loc_data.keys())
    missing = ssot_keys - loc_keys
    extra = loc_keys - ssot_keys
    coverage = (len(ssot_keys) - len(missing)) / len(ssot_keys) * 100
    print(f'| {loc_dir.name} | {len(missing)} | {len(extra)} | {coverage:.1f}% |')

print()
print('## Detailed missing keys per locale')
for loc_dir in sorted(LOCALES_DIR.iterdir()):
    if loc_dir.name == SSOT or not loc_dir.is_dir():
        continue
    json_file = loc_dir / 'translation.json'
    if not json_file.exists(): continue
    loc_data = flatten(json.load(open(json_file)))
    missing = ssot_keys - set(loc_data.keys())
    if missing:
        print(f'\n### {loc_dir.name} ({len(missing)} missing)')
        for k in sorted(missing)[:30]:
            print(f'- `{k}`')
        if len(missing) > 30:
            print(f'- ...{len(missing) - 30} more')
