import time

# ███ General settings and hyperparameters ███
BATCH_SIZE = 128
# NUM_EPOCHS = 200
NUM_EPOCHS = 50
LEARNING_RATE = 0.005
NUM_WORKERS = 0
# NROWS = None
NROWS = 100000
NUM_BINS = 10
HIDDEN_UNITS = (120, 240, 120, 12)

def run_id():
    return f"{time.strftime('%Y%m%d')}_{NROWS}_{'_'.join(map(str, HIDDEN_UNITS))}"
