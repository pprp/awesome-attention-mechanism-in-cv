import torch
import torchvision
from torch.utils.data import DataLoader
import models
from loader import get_val_dataloader,get_train_dataloader
import argparse
from utils import accuracy
import torch.nn as nn
from collections import OrderedDict
from utils import accuracy


def get_parser():
    parser = argparse.ArgumentParser('testing')
    parser.add_argument('--rootdir', type=str,
                        default="E:/imagenet_data", help='root dir of imagenet')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=64, help='batch size')
    parser.add_argument('--arch', type=str,
                        default='resnet50', help='CNN architecture')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use')
    parser.add_argument('--distributed', type=bool,
                        default=False, help='multi gpu inference')
    parser.add_argument('-j', '--workers', type=int,
                        default=8, help='num of workers')
    parser.add_argument('--load_path', type=str,
                        default='checkpoints/epoch_90_resnet50.pt', help='weights path')
    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    val_loader = get_train_dataloader(
        args.rootdir, args.distributed, args.workers, args.batch_size)

    # model new
    model = models.resnet50()

    # checkpoint load
    if args.gpu is None:
        # cpu
        print("=> loading checkpoint '{}'".format(args.load_path))
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # gpu
        print("=> loading checkpoint '{}'".format(args.load_path))
        torch.cuda.set_device(args.gpu)
        checkpoint = torch.load(
            args.load_path, map_location='cuda:{}'.format(args.gpu))
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model = nn.DataParallel(model)  # 数据并行
        model = model.cuda(args.gpu)

    model.eval()

    total = len(val_loader)

    #####################TEST ####################
    # for i, (a, b) in enumerate(val_loader):
    #     print(a.shape, b.shape)
    ##############################################

    with torch.no_grad():
        correct_1 = 0
        correct_5 = 0
        n_sample = 0

        for i, (images, targets) in enumerate(val_loader):
            print('\r', i, end="")
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                targets = targets.cuda(args.gpu, non_blocking=True)

            output = model(images)

            acc1, acc5 = accuracy(output, targets, topk=(1, 5))

            correct_1 += acc1
            correct_5 += acc5
            n_sample += 1

        print("TOP 1:", correct_1/n_sample)
        print("TOP 5", correct_5/n_sample)
        print("Parameter numbers: {}".format(sum(p.numel()
                                                 for p in model.parameters())))

        # acc1, acc5 = accuracy(output, target, topk=(1, 5))


if __name__ == "__main__":
    main()
