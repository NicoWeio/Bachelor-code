import time

# ███ General settings and hyperparameters ███
BATCH_SIZE = 2**6
# BATCH_SIZE = 2**15 # Test, lol
# NUM_EPOCHS = 200
NUM_EPOCHS = 3
NUM_DSEA_ITERATIONS = 3
LEARNING_RATE = 0.005
NUM_WORKERS = 48 # no speedup compared to 0
# NROWS = None # = all rows
NROWS = 100000 # reasonable subset
# NROWS = 1000 # very small subset
NUM_BINS = 10
HIDDEN_UNITS = (120, 240, 120, 12)

def run_id():
    return f"{time.strftime('%Y%m%d')}_{NROWS}_{'_'.join(map(str, HIDDEN_UNITS))}"
