import pandas as pd
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

# function  for Data pre-processing method according to IDS
def prepare_feature(data):
    #data = pd.read_csv('data/UNSW_NB15_training-set.csv')
    max_feature = dict()
    min_feature = dict()
    # can change in exception if you don't want this process on any feature
    # exception = ['label', 'proto', 'service', 'state', 'attack_cat']
    exception = ['label']
    for col in data.columns:
        max_feature[col] = data[col].max()
        min_feature[col] = data[col].min()

    for col in data.columns:
        if col in exception:
            continue
        data[col] = data[col] - min_feature[col]
        data[col] = data[col] / (max_feature[col] - min_feature[col])

    return data

# Need to find out actual number of classes from data set
def extract_labels(data, one_hot=False, num_classes=2):
    # label from label column of Data
    labels = data['label'].to_numpy()
    if one_hot:
        # labels = to_categorical(labels)
         return dense_to_one_hot(labels, num_classes)
    return labels
def extract_labels_attack(data, one_hot=False, num_classes=2):
    # label from label column of Data
    labels = data['label'].to_numpy()
    label_normal=labels[labels==0]
    label_attack = labels[labels != 0]
    if one_hot:
        # labels = to_categorical(labels)
        label_normal=dense_to_one_hot(label_normal, num_classes)
        label_attack = dense_to_one_hot(label_attack, num_classes)
    return label_normal, label_attack

# Extract feature + label values and replace any non numerical feature value with numerical value with split attack_cat
def extract_features_attack(data):
    proto_unique = data['proto'].unique()
    service_unique = data['service'].unique()
    state_unique = data['state'].unique()
    attack_unique = data['attack_cat'].unique()
    # print("attack_unique", attack_unique)
    data.pop('label')
    data.pop('id')
    for i in range(len(proto_unique)):
        data = data.replace(proto_unique[i], i)
    for i in range(len(service_unique)):
        data = data.replace(service_unique[i], i)
    for i in range(len(state_unique)):
        data = data.replace(state_unique[i], i)
    for i in range(len(attack_unique)):
        data = data.replace(attack_unique[i], i)
    data_normal = data[data['attack_cat'] == 0]
    data_attack = data[data['attack_cat'] != 0]
    return data, data_normal, data_attack
# Extract feature values and replace any non numerical feature value with numerical value here
def extract_features(data):
    proto_unique = data['proto'].unique()
    service_unique = data['service'].unique()
    state_unique = data['state'].unique()
    attack_unique = data['attack_cat'].unique()
    data.pop('label')
    data.pop('id')
    for i in range(len(proto_unique)):
        data = data.replace(proto_unique[i], i)
    for i in range(len(service_unique)):
        data = data.replace(service_unique[i], i)
    for i in range(len(state_unique)):
        data = data.replace(state_unique[i], i)
    for i in range(len(attack_unique)):
        data = data.replace(attack_unique[i], i)
    return data

def extract_labels_multi(data, one_hot=False, num_classes=10):
    # label from label column of Data
    labels = data['attack_cat'].to_numpy()
    if one_hot:
        # labels = to_categorical(labels)
         return dense_to_one_hot(labels, num_classes)
    data.pop('attack_cat')
    return labels
def extract_labels_attack_multi(data, one_hot=False, num_classes=10):
    # label from label column of Data
    labels = data['attack_cat'].to_numpy()
    label_normal=labels[labels==0]
    label_attack = labels[labels != 0]
    if one_hot:
        # labels = to_categorical(labels)
        label_normal=dense_to_one_hot(label_normal, num_classes)
        label_attack = dense_to_one_hot(label_attack, num_classes)
    data.pop('attack_cat')
    return label_normal, label_attack

def data_importer_GAN(one_hot=False):
    TRAIN_SET = pd.read_csv('data/UNSW_NB15_testing-set.csv')
    TEST_SET = pd.read_csv('data/UNSW_NB15_training-set.csv')
    total_sample = TRAIN_SET.shape[0]
    # print("total_sample", total_sample)
    # trainStart = total_sample // 2
    # TRAIN_SET = TRAIN_SET.iloc[:trainStart, :]
    # print("GAN train sample", TRAIN_SET.shape)
    # print(TRAIN_SET.head())
    dtype = dtypes.float32
    df = pd.DataFrame(np.random.randn(len(TRAIN_SET), 2))
    # mask = np.random.rand(len(df)) < 0.8

    # ACTUAL_TRAIN_SET = TRAIN_SET[mask]
    ACTUAL_TRAIN_SET = TRAIN_SET
    # print(ACTUAL_TRAIN_SET.head())
    # VALIDATION_SET = TRAIN_SET[~mask]

    train_labels = extract_labels(ACTUAL_TRAIN_SET, one_hot=one_hot)
    train_samples = extract_features(ACTUAL_TRAIN_SET)
    test_labels_normal, test_labels_attack = extract_labels_attack(TEST_SET, one_hot=one_hot)
    test_samples_normal, test_samples_attack = extract_features_attack(TEST_SET)
    print(test_labels_attack[0:5])

    # test_labels = extract_labels(TEST_SET, one_hot=one_hot)
    # test_samples = extract_features(TEST_SET)

    test_normal = DataSet(test_samples_normal, test_labels_normal, dtype=dtype)
    test_attack = DataSet(test_samples_attack,
                         test_labels_attack,
                         dtype=dtype)
    train = DataSet(train_samples, train_labels, dtype=dtype)

    return base.Datasets(train=train, validation=test_attack, test=test_normal)
    # return base.Datasets(train=train, validation=validation)

def data_importer_IDS(Evaluate=False):
    TRAIN_SET = pd.read_csv('data/UNSW_NB15_testing-set.csv')
    TEST_SET = pd.read_csv('data/UNSW_NB15_training-set.csv')
    total_sample=TRAIN_SET.shape[0]
    print("total training sample", total_sample)
    print("total test sample", TEST_SET.shape[0])
    # trainStart=total_sample//2
    # TRAIN_SET=TRAIN_SET.iloc[trainStart:,:]
    # print("IDS train sample", TRAIN_SET.shape)
    # print(TRAIN_SET.head())
    dtype = dtypes.float32
    if Evaluate==False:
        df = pd.DataFrame(np.random.randn(len(TRAIN_SET), 2))
        mask = np.random.rand(len(df)) < 0.8

        ACTUAL_TRAIN_SET = TRAIN_SET[mask]
        # print(ACTUAL_TRAIN_SET.head())
        VALIDATION_SET = TRAIN_SET[~mask]

        # validation_labels = extract_labels(VALIDATION_SET, one_hot=True)
        # validation_samples = extract_features(VALIDATION_SET)
        #
        # test_labels = extract_labels(TEST_SET, one_hot=True)
        # test_samples = extract_features(TEST_SET)
        validation_samples = extract_features(VALIDATION_SET)
        validation_labels = extract_labels_multi(validation_samples, one_hot=True)

        test_samples = extract_features(TEST_SET)
        test_labels = extract_labels_multi(test_samples, one_hot=True)

        validation = DataSet(validation_samples,
                             validation_labels,
                             dtype=dtype)
        test = DataSet(test_samples, test_labels, dtype=dtype)
    else:
        ACTUAL_TRAIN_SET=TRAIN_SET

        # test_labels_normal, test_labels_attack = extract_labels_attack(TEST_SET, one_hot=True)
        # test_samples_normal, test_samples_attack = extract_features_attack(TEST_SET)

        test_samples_full, test_samples_normal, test_samples_attack = extract_features_attack(TEST_SET)
        test_labels_normal, test_labels_attack = extract_labels_attack_multi(test_samples_full, one_hot=True)
        validation = DataSet(test_samples_attack,
                             test_labels_attack,
                             dtype=dtype)
        test = DataSet(test_samples_normal, test_labels_normal, dtype=dtype)

    # train_labels = extract_labels(ACTUAL_TRAIN_SET, one_hot=True)
    # train_samples = extract_features(ACTUAL_TRAIN_SET)

    train_samples = extract_features(ACTUAL_TRAIN_SET)
    train_labels = extract_labels_multi(train_samples,one_hot=True)
    # print(train_labels.head())
    train = DataSet(train_samples, train_labels, dtype=dtype)
    return base.Datasets(train=train, validation=validation, test=test)

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

class DataSet(object):

    def __init__(self,
                 samples,
                 labels,
                 one_hot=False,
                 dtype=dtypes.float32):
        """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)

        assert samples.shape[0] == labels.shape[0], (
                'samples.shape: %s labels.shape: %s' % (samples.shape, labels.shape))
        self._num_examples = samples.shape[0]
        # print("self._num_examples", self._num_examples)
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            samples = samples.astype(np.float32)
            samples = np.multiply(samples, 1.0 / 255.0)
        self._samples = samples
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def samples(self):
        return self._samples

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            # perm = np.arange(self._num_examples)
            # np.random.shuffle(perm)
            # sample_shape=self._samples.shape
            # self._samples = self._samples[perm]
            # label_shape = self._labels.shape
            # self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._samples[start:end], self._labels[start:end]
