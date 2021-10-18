from torch.utils import data
import torchvision.transforms as transforms

from . import base_dataset as basedat
from .dataset_config import dataset_config


def get_loaders(datasets, num_tasks, nc_first_task, batch_size, num_workers,
                pin_memory, force_order=True, val_only=False):
    """Apply transformations to Datasets and create the DataLoaders for each task"""

    trn_load, val_load, tst_load = [], [], []
    taskcla = []
    dataset_offset = 0
    for idx_dataset, cur_dataset in enumerate(datasets, 0):
        # get configuration for current dataset
        dc = dataset_config[cur_dataset]

        # fixing class order
        if force_order:
            class_order = list(range(dc['num_classes']))
        else:
            class_order = dc['class_order']

        # transformations
        trn_transform, tst_transform = get_transforms(resize=dc['resize'],
                                                      pad=dc['pad'],
                                                      crop=dc['crop'],
                                                      flip=dc['flip'],
                                                      normalize=dc['normalize'],
                                                      extend_channel=dc['extend_channel'])

        # datasets
        trn_dset, val_dset, tst_dset, curtaskcla = get_datasets(cur_dataset, dc['path'], num_tasks, nc_first_task,
                                                                validation=dc['validation'],
                                                                trn_transform=trn_transform,
                                                                tst_transform=tst_transform,
                                                                class_order=class_order,
                                                                val_only=val_only)

        # apply offsets in case of multiple datasets
        if idx_dataset > 0:
            for tt in range(num_tasks):
                trn_dset[tt].labels = [elem + dataset_offset for elem in trn_dset[tt].labels]
                val_dset[tt].labels = [elem + dataset_offset for elem in val_dset[tt].labels]
                tst_dset[tt].labels = [elem + dataset_offset for elem in tst_dset[tt].labels]
        dataset_offset = dataset_offset + sum([tc[1] for tc in curtaskcla])

        # reassign class idx for multiple dataset case
        curtaskcla = [(tc[0] + idx_dataset * num_tasks, tc[1]) for tc in curtaskcla]

        # extend final taskcla list
        taskcla.extend(curtaskcla)

        # loaders
        for tt in range(num_tasks):
            trn_load.append(data.DataLoader(trn_dset[tt], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                            pin_memory=pin_memory))
            val_load.append(data.DataLoader(val_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory))
            tst_load.append(data.DataLoader(tst_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory))

    # Debugging outputs
    print("Train set size/task: %d" % len(trn_load[0]))
    print("Val set size/task: %d"   % len(val_load[0]))
    print("Test set size/task: %d"  % len(tst_load[0]))

    return trn_load, val_load, tst_load, taskcla


def get_datasets(dataset, path, num_tasks, nc_first_task, validation,
                 trn_transform, tst_transform,
                 class_order=None, val_only=False):
    """Extract datasets and create Dataset class"""

    trn_dset, val_dset, tst_dset = [], [], []

    # read data paths and compute splits -- path needs to have a train.txt and a test.txt with image-label pairs
    all_data, taskcla, class_indices = basedat.get_data(path, num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                        validation=validation, shuffle_classes=class_order is None,
                                                        class_order=class_order,
                                                        val_only=val_only)

    # set dataset type
    Dataset = basedat.BaseDataset

    # get datasets, apply correct label offsets for each task
    offset = 0
    for task in range(num_tasks):
        all_data[task]['trn']['y'] = [label + offset for label in all_data[task]['trn']['y']]
        all_data[task]['val']['y'] = [label + offset for label in all_data[task]['val']['y']]
        all_data[task]['tst']['y'] = [label + offset for label in all_data[task]['tst']['y']]

        trn_dset.append(Dataset(all_data[task]['trn'], trn_transform, class_indices))
        val_dset.append(Dataset(all_data[task]['val'], tst_transform, class_indices))
        tst_dset.append(Dataset(all_data[task]['tst'], tst_transform, class_indices))
        offset += taskcla[task][1]

    return trn_dset, val_dset, tst_dset, taskcla


def get_transforms(resize, pad, crop, flip, normalize, extend_channel):
    """Unpack transformations and apply to train or test splits"""

    trn_transform_list = []
    tst_transform_list = []

    trn_transform_list += [transforms.RandomResizedCrop(224)]
    trn_transform_list += [transforms.RandomHorizontalFlip()]
    trn_transform_list += [transforms.ToTensor()]

    tst_transform_list += [transforms.Resize(256)]
    tst_transform_list += [transforms.CenterCrop(224)]
    tst_transform_list += [transforms.ToTensor()]

    # normalization
    trn_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))
    tst_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))

    return transforms.Compose(trn_transform_list), transforms.Compose(tst_transform_list)
