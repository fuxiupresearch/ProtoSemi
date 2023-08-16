import numpy as np
import PIL
from torchvision import datasets, transforms
from .cifar import CIFAR10, CIFAR100

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

train_cifar10_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

test_cifar10_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_cifar100_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
)

test_cifar100_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
)


def build_transform(is_train, args):
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == "animal10n":
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def input_dataset(dataset, dataset_dir, noise_type, noise_path, is_human, args=None):
    if dataset == "cifar10":
        train_dataset = CIFAR10(
            root=dataset_dir,
            download=True,
            train=True,
            transform=train_cifar10_transform,
            noise_type=noise_type,
            noise_path=noise_path,
            is_human=is_human,
        )
        test_dataset = CIFAR10(
            root=dataset_dir,
            download=False,
            train=False,
            transform=test_cifar10_transform,
            noise_type=noise_type,
        )
        num_classes = 10
        num_training_samples = 50000
    elif dataset == "cifar100":
        train_dataset = CIFAR100(
            root=dataset_dir,
            download=True,
            train=True,
            transform=train_cifar100_transform,
            noise_type=noise_type,
            noise_path=noise_path,
            is_human=is_human,
        )
        test_dataset = CIFAR100(
            root=dataset_dir,
            download=False,
            train=False,
            transform=test_cifar100_transform,
            noise_type=noise_type,
        )
        num_classes = 100
        num_training_samples = 50000
    elif dataset == "animal10n":
        num_classes = 10
        num_training_samples = 50000
        train_transform = build_transform(is_train=True, args=args)
        test_transform = build_transform(is_train=False, args=args)
        train_dataset = datasets.ImageFolder(
            dataset_dir + "/Animal10N/train", transform=train_transform
        )
        test_dataset = datasets.ImageFolder(
            dataset_dir + "/Animal10N/test", transform=test_transform
        )
    return train_dataset, test_dataset, num_classes, num_training_samples
