## Datasets

This folder contains all image splits used in the paper.
First extract all lists by running the following command:

```bash
cd ./datasets/
tar -xf image_lists.tar.xz
```

Datasets with a name ending in <code>"mixed100"</code>, and the <code>"1bird100"</code> dataset are sampled from __ImageNet__.
We also provide image lists for __CIFAR-100__, __Food-100__ and __Places-100__. As described in the WACV-2022 paper, __Food-100__ and __Places-100__ are sampled from __Food-101__ and __Places-365__ respectively.

For all image lists, the string <code>"/DATASET_PATH/"</code> must be replaced by the absolute path to each dataset in all image lists.
As an example for the __CIFAR-100__ lists, this can be done as follows:

```bash
cd ./datasets/target/cifar100/
grep -rl '/DATASET_PATH/' . | xargs sed -i 's\/path_to_cifar100/\/DATASET_PATH/\g'
```

This command is recursive and can be applied directly to a whole folder. For the reference lists:

```bash
cd ./datasets/reference/
grep -rl '/DATASET_PATH/' . | xargs sed -i 's\/path_to_imagenet/\/DATASET_PATH/\g'
```

Make sure that the new paths are pointing to existing files after executing this command.
