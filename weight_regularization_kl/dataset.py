from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms


class DataLoaderWrapper(DataLoader):
    def __init__(self, dataset, is_train, batch_size, num_workers=2):
        super().__init__(
                dataset,
                batch_size=batch_size, shuffle=is_train, num_workers=num_workers)


class BasicDataLoader(DataLoaderWrapper):
    def __init__(self, params, is_train, batch_size, data_root, normalize):
        augments = list()

        if is_train:
            augments.append(transforms.RandomHorizontalFlip())

        augments.append(transforms.ToTensor())

        if normalize:
            augments.append(transforms.Normalize(params.mean, params.std))

        data = params.dataset(
                root=data_root, train=is_train, download=True,
                transform=transforms.Compose(augments))

        super().__init__(data, is_train, batch_size)


class CIFAR10(object):
    dataset = torchvision.datasets.CIFAR10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    in_channels = 3
    size = 32
    num_classes = 10

    def get_data(data_root, is_train, batch_size, normalize=True):
        return BasicDataLoader(CIFAR10, is_train, batch_size, data_root, normalize)
