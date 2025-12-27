import json
from segmentation.rf_classifier import build_training_set

X, y = build_training_set()
info = {
    'X_shape': getattr(X, 'shape', None),
    'y_shape': getattr(y, 'shape', None),
    'unique_labels': list(set(y.tolist())) if hasattr(y, 'tolist') else []
}
with open('training_info.json', 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2)
print('dumped training_info.json')
