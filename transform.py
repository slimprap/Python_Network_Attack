import pandas as pd
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base

def data_importer():
    TRAIN_SET = pd.read_csv('data/UNSW_NB15_testing-set.csv')
    TEST_SET = pd.read_csv('data/UNSW_NB15_training-set.csv')
    dtype = dtypes.float32
    reshape = True
    df = pd.DataFrame(np.random.randn(len(TRAIN_SET), 2))
    mask = np.random.rand(len(df)) < 0.8

    train = TRAIN_SET[mask]
    validation = TRAIN_SET[~mask]
    test = TEST_SET

    return base.Datasets(train=train, validation=validation, test=test)

