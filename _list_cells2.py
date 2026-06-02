import json

with open('notebooks/comparisons/rtdert_vs_yolo26.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])[:60].replace('\n', ' ')
    metadata = c.get('metadata', {})
    cell_id = metadata.get('vscode', {}).get('cellId', c.get('id', 'no-id'))
    print(f"Cell {i+1:2d}: {cell_id[:30]:30s} | {src}")
