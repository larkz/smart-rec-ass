import numpy as np

LEVEL_MAPPING_DIC = {'Internship':0, 'Entry Level':1, 'Mid Level':2, 'Senior Level':3}
INVERT_LVL_DIC = {value: key for key, value in LEVEL_MAPPING_DIC.items()}

TEST_SIZE = 0.2
RANDOM_SEED = 99

N_ESTIMATORS = 2
MAX_DEPTH = 20
LR = 0.03
SUBSAMPLE = 0.7
COLSAMP = 0.6