import os
import os.path
import sys
import torch.utils.data as data
from PIL import Image


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


class ImagesFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

        Args:
            transform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
            target_transform (callable, optional): A function/transform that takes
                in the target and transforms it.
         Attributes:
            classes (list): List of the class names.
            samples (list): List of (sample path, class_index) tuples
            targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, images_list_file, transform=None,
                 target_transform=None, return_path=False):
        self.return_path = return_path
        images_list_file = open(images_list_file, 'r').readlines()
        samples = []
        for e in images_list_file:
            e = e.strip()
            image_path = e.split()[0]
            try:
                assert (os.path.exists(image_path))
            except AssertionError:
                print('Cant find ' + image_path)
                sys.exit(-1)
            image_class = int(e.split()[-1])
            samples.append((image_path, image_class))

        if len(samples) == 0:
            raise (RuntimeError("No image found"))

        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.classes = list(set([e[1] for e in samples]))
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_path:
            return (sample, target), self.samples[index][0]
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
