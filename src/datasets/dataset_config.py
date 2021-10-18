import os
from os.path import join
import re
import sys

# Local imports
sys.path.append(os.path.join(sys.path[0], "./../../"))
sys.path.append(os.path.join(sys.path[0], "./../"))
from env import env_image_lists

_BASE_LIST_PATH = env_image_lists

_BASE_REF_DSET_PATH = join(_BASE_LIST_PATH, "reference")
_BASE_TEST_DSET_PATH = join(_BASE_LIST_PATH, "target")
_DSET_MEAN_STD_FILE = join(_BASE_LIST_PATH, "datasets_stats.txt")
_BASE_TEST_HALF_DSET_PATH = join(_BASE_LIST_PATH, "target_half")


def get_dset_mean_std(dataset):
    """
    Retrieving normalizing constants of a given dataset.
    """
    datasets_mean_std_file = open(_DSET_MEAN_STD_FILE, 'r').readlines()

    for line in datasets_mean_std_file:
        line = line.strip().split(':')
        dataset_name, dataset_stat = line[0], line[1]
        if dataset_name == dataset:
            dataset_stat = dataset_stat.split(';')
            dataset_mean = [float(e) for e in re.findall(r'\d+\.\d+', dataset_stat[0])]
            dataset_std = [float(e) for e in re.findall(r'\d+\.\d+', dataset_stat[1])]
            return dataset_mean, dataset_std

    print('Invalid normalization dataset name')
    exit(-1)


# Defining target datasets
dataset_config = {
    '0mixed100': {
        'path': join(_BASE_TEST_DSET_PATH, '0mixed100'),
        'resize': None,
        'crop': 224,
        'flip': True,
        'normalize': get_dset_mean_std('0mixed100'),
        'validation': 0.1,
        'num_classes': 100
    },
    '1bird100': {
        'path': join(_BASE_TEST_DSET_PATH, '1bird100'),
        'resize': None,
        'crop': 224,
        'flip': True,
        'normalize': get_dset_mean_std('1bird100'),
        'validation': 0.1,
        'num_classes': 100
    },
    'cif100': {
        'path': join(_BASE_TEST_DSET_PATH, 'cifar100'),
        'resize': None,
        'crop': 224,
        'flip': True,
        'normalize': get_dset_mean_std('cifar100'),
        'validation': 0.1,
        'num_classes': 100
    },
    'food100': {
        'path': join(_BASE_TEST_DSET_PATH, 'food100'),
        'resize': None,
        'crop': 224,
        'flip': True,
        'normalize': get_dset_mean_std('food100'),
        'validation': 0.1,
        'num_classes': 100
    },
    'places100': {
        'path': join(_BASE_TEST_DSET_PATH, 'places100'),
        'resize': None,
        'crop': 224,
        'flip': True,
        'normalize': get_dset_mean_std('ilsvrc'),
        'validation': 0.1,
        'num_classes': 100
    },
    'places100_half': {
        'path': join(_BASE_TEST_HALF_DSET_PATH, 'places100'),
        'resize': None,
        'crop': 224,
        'flip': True,
        'normalize': get_dset_mean_std('ilsvrc'),
        'validation': 0.1,
        'num_classes': 100
    },

    # Mock datasets for testing
    'mock_ref': {
        'path': join(_BASE_REF_DSET_PATH, 'mock_dataset'),
        'resize': None,
        'crop': 224,
        'flip': True,
        'normalize': get_dset_mean_std('cifar100'),
        'validation': 0.1,
        'num_classes': 10
    },
    'mock_target': {
        'path': join(_BASE_TEST_DSET_PATH, 'mock_dataset'),
        'resize': None,
        'crop': 224,
        'flip': True,
        'normalize': get_dset_mean_std('cifar100'),
        'validation': 0.1,
        'num_classes': 10
    },
}

# Defining reference datasets
# Reference datasets are sampled without replacement from ImageNet
for k in range(1, 11):
    dataset_config['%dmixed100' % k] = \
        {
            'path': join(_BASE_REF_DSET_PATH, '%dmixed100' % k),
            'resize': None,
            'crop': 224,
            'flip': True,
            'normalize': get_dset_mean_std('%dmixed100' % k),
            'validation': 0.3,
            'num_classes': 100
        }

# Add missing keys:
for dset in dataset_config.keys():
    for k in ['resize', 'pad', 'crop', 'normalize', 'class_order', 'extend_channel']:
        if k not in dataset_config[dset].keys():
            dataset_config[dset][k] = None
    if 'flip' not in dataset_config[dset].keys():
        dataset_config[dset]['flip'] = False
