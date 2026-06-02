import json

with open('notebooks/comparisons/rtdert_vs_yolo26.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])[:80].replace('\n', ' ')
    cell_id = c.get('id', 'no-id')
    print(f"Cell {i+1:2d} ({c['cell_type']:8s}) {cell_id[:20]:20s}: {src}")
