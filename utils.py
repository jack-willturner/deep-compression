from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import GPUtil

__all__ = [
    "AverageMeter",
    "get_cifar_loaders",
    "load_model",
    "get_error",
    "get_no_params",
    "train",
    "validate",
    "finetune",
    "Cutout",
    "select_devices",
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
error_history = []


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_cifar_loaders(
    data_loc="./disk/scratch/datasets/cifar10/",
    batch_size=128,
    cutout=True,
    n_holes=1,
    length=16,
):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    if cutout:
        transform_train.transforms.append(Cutout(n_holes=n_holes, length=length))

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_loc, train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_loc, train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return trainloader, testloader


def load_model(model, sd):
    sd = torch.load("checkpoints/%s.t7" % sd, map_location="cpu")

    new_sd = model.state_dict()
    if "state_dict" in sd.keys():
        old_sd = sd["state_dict"]
    else:
        old_sd = sd["net"]

    new_names = [v for v in new_sd]
    old_names = [v for v in old_sd]
    for i, j in enumerate(new_names):
        if not "mask" in j:
            new_sd[j] = old_sd[old_names[i]]

    model.load_state_dict(new_sd)

    return model, sd


def get_error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res


# count only conv params for now
def get_no_params(net, verbose=False, mask=False):
    params = net
    tot = 0
    for p in params:
        no = torch.sum(params[p] != 0)
        if "conv" in p:
            tot += no
    return tot


def train(model, trainloader, criterion, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(trainloader):
        input, target = input.to(device), target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(model, epoch, valloader, criterion, checkpoint=None, seed=None):
    global error_history

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for _, (input, target) in enumerate(valloader):
        input, target = input.to(device), target.to(device)

        output = model(input)
        loss = criterion(output, target)
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

    error_history.append(top1.avg)
    if checkpoint:

        state = {
            "net": model.state_dict(),
            "masks": [w for name, w in model.named_parameters() if "mask" in name],
            "epoch": epoch,
            "error_history": error_history,
        }
        torch.save(state, f"checkpoints/{checkpoint}_{seed}.t7")


def finetune(model, trainloader, criterion, optimizer, steps=100):
    # switch to train mode
    model.train()
    dataiter = iter(trainloader)
    for i in range(steps):
        try:
            input, target = dataiter.next()
        except StopIteration:
            dataiter = iter(trainloader)
            input, target = dataiter.next()

        input, target = input.to(device), target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


# explicit by pass of the below
def select_devices(
    num_gpus_to_use=0, max_load=0.01, max_memory=0.01, exclude_gpu_ids=None
):

    if num_gpus_to_use == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        gpu_to_use = GPUtil.getAvailable(
            order="first",
            limit=num_gpus_to_use,
            maxLoad=max_load,
            maxMemory=max_memory,
            includeNan=False,
            excludeID=exclude_gpu_ids,
            excludeUUID=[],
        )
        if len(gpu_to_use) < num_gpus_to_use:
            raise OSError(
                "Couldnt find enough GPU(s) as required by the user, stopping program "
                "- consider reducing "
                "the requirements or using num_gpus_to_use=0 to use CPU"
            )

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu_idx) for gpu_idx in gpu_to_use
        )

        print("GPUs selected have IDs {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
