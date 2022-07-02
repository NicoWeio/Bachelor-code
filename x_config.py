import time

# ███ General settings and hyperparameters ███
BATCH_SIZE = 2**6
# NUM_EPOCHS = 200
NUM_EPOCHS = 5
LEARNING_RATE = 0.005
NUM_WORKERS = 48 # no speedup compared to 0
# NROWS = None # = all rows
NROWS = 100000
NUM_BINS = 10
HIDDEN_UNITS = (120, 240, 120, 12)

def run_id():
    return f"{time.strftime('%Y%m%d')}_{NROWS}_{'_'.join(map(str, HIDDEN_UNITS))}"
