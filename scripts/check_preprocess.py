import os, json, traceback
from segmentation.rf_classifier import extract_cluster_features

DATASET = 'dataset'
MAX_PER_LABEL = 10
out = {'summary':{}, 'details':[]}

for label in sorted(os.listdir(DATASET)):
    label_path = os.path.join(DATASET, label)
    if not os.path.isdir(label_path):
        continue

    files = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]
    files = files[:MAX_PER_LABEL]
    out['summary'][label] = {'attempted': len(files), 'succeeded': 0, 'failed': 0}

    for fname in files:
        path = os.path.join(label_path, fname)
        try:
            feats = extract_cluster_features(path, K=10)
            out['details'].append({'file': path, 'label': label, 'num_clusters': len(feats)})
            out['summary'][label]['succeeded'] += 1
            print(f"OK: {path} -> clusters={len(feats)}")
        except Exception as e:
            out['details'].append({'file': path, 'label': label, 'error': str(e)})
            out['summary'][label]['failed'] += 1
            print(f"ERR: {path} -> {e}")
            traceback.print_exc()

with open('preprocess_check.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)

print('\nDONE. Wrote preprocess_check.json')
