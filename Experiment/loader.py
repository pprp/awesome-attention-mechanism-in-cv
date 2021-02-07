import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_train_dataloader(rootdir, distributed=False, workers=8, batch_size=64):
    traindir = os.path.join(rootdir, "train")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(traindir, train_transform)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=workers,
                                               pin_memory=True,
                                               sampler=train_sampler)
    return train_loader


def get_val_dataloader(rootdir, distributed=False, workers=8, batch_size=64):
    valdir = os.path.join(rootdir, "val")
    val_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_dataset = datasets.ImageFolder(valdir, val_transformer)
    if distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=val_sampler)
    return val_loader


def get_train_val_dataloader(rootdir, distributed=False, workers=8, batch_size=64):
    # Data loading code
    traindir = os.path.join(rootdir, 'train')
    valdir = os.path.join(rootdir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    datasets.CIFAR10("./data", train=True, transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])))

    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(
            train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, val_loader, train_sampler

if __name__ == "__main__":
    val_loader = get_val_dataloader('E:/imagenet_data')

    for i,(a,b) in enumerate(val_loader):
        print(a.shape, b.shape)