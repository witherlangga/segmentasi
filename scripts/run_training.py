import traceback
from segmentation.rf_classifier import train_model

print('--- RUN TRAINING SCRIPT START ---')
try:
    train_model(force=True)
    print('--- TRAINING FINISHED SUCCESSFULLY ---')
except Exception as e:
    print('--- TRAINING FAILED ---')
    traceback.print_exc()
