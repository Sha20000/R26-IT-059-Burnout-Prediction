import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Configuration

SEED = 42 
PROCESSED_PATH = "../data/processed/"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

np.random.seed(SEED)