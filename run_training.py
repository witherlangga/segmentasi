import traceback
from segmentation.rf_classifier import train_model

print('--- RUN_TRAINING.PY START ---', flush=True)
try:
    train_model(force=True)
    print('--- TRAINING COMPLETED SUCCESSFULLY ---', flush=True)
except Exception:
    print('--- TRAINING FAILED ---', flush=True)
    traceback.print_exc()


