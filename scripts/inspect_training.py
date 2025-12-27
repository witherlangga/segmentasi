from segmentation.rf_classifier import build_training_set, train_model
import numpy as np

print('Starting inspect')
X, y = build_training_set()
print('X shape:', getattr(X, 'shape', None))
print('y shape:', getattr(y, 'shape', None))
if len(y) > 0:
    print('unique labels:', np.unique(y))
    print('per-label counts:')
    labels, counts = np.unique(y, return_counts=True)
    for l,c in zip(labels, counts):
        print(l, c)
else:
    print('No training samples found')

# Try to train and capture exceptions
try:
    print('\nRunning train_model()...')
    train_model()
    print('train_model() finished')
except Exception as e:
    import traceback
    print('train_model() failed')
    traceback.print_exc()
