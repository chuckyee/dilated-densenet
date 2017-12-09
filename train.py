#!/usr/bin/env python

from __future__ import division, print_function


import argparse
import logging
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from dataset import SegmentationDataset, normalize
from densenet import DilatedDenseNet
from loss import CrossEntropyLoss2d


def train_epoch(epoch, args, model, loader, criterion, optimizer):
    model.train()

    losses = []
    sizes = []

    for step, (images,masks) in enumerate(loader):
        if args.ngpu > 0:
            images = images.cuda()
            masks = masks.cuda()

        x = Variable(images)
        y_true = Variable(masks)
        y_pred = model(x)

        optimizer.zero_grad()   # do I also need to zero the model gradients?
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])
        sizes.append(len(images))

        logging.info("Epoch: {}  Batch: {}  Loss: {}".format(
            epoch, step, loss.data[0]))

    total_loss = sum(n*x for n,x in zip(sizes, losses)) / sum(sizes)
    logging.info("Epoch: {}  Loss: {}".format(epoch, total_loss))

    if args.checkpoint:
        save_path = 'tensors-{:03d}.pt'.format(epoch)
        torch.save(model.state_dict(), save_path)


def evaluate(args, model, loader, criterion):
    model.eval()

    losses = []
    sizes = []

    for step, (images,masks) in enumerate(loader):
        if args.ngpu > 0:
            images = images.cuda()
            masks = masks.cuda()

        x = Variable(images)
        y_true = Variable(masks)
        y_pred = model(x)

        loss = criterion(y_pred, y_true).data[0]
        losses.append(loss)
        sizes.append(len(images))

        logging.info("Batch: {}  Loss: {}".format(step, loss))

    total_loss = sum(n*x for n,x in zip(sizes, losses)) / sum(sizes)
    logging.info("Loss: {}".format(loss))


def train_val_dataloaders(dataset, val_split=0.2, batch_size=32, seed=0):
    np.random.seed(seed)   # crucial: controls train-val split

    ntotal = len(dataset)
    ntrain = int((1 - val_split) * len(dataset))

    train_indices = np.random.choice(ntotal, ntrain, replace=False)
    train_sampler = SubsetRandomSampler(train_indices)

    val_indices = np.setdiff1d(np.arange(ntotal), train_indices, assume_unique=True)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader


def main(args):
    logging.info("Construct dataset...")

    input_transform = transforms.Compose([
        normalize,
        transforms.ToTensor(),
    ])
    output_transform = None
    dataset = SegmentationDataset(args.datadir, input_transform, output_transform)

    # extract number of channels in input images / number of classes
    image, mask = dataset[0]
    channels, _, _ = image.shape

    logging.info("Construct model...")

    model = DilatedDenseNet(
        image_channels=channels,
        num_init_features=args.features,
        growth_rate=args.features,
        layers=args.layers,
        dropout_rate=args.dropout,
        classes=args.classes,
        dilated=True)

    if args.ngpu > 0:
        model = nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        model = model.cuda()

    # setup cross entropy loss
    weight = torch.ones(2)
    if args.ngpu > 0:
        weight = weight.cuda()
    criterion = CrossEntropyLoss2d(weight)

    optimizer = Adam(model.parameters())

    if args.mode == 'train':
        train_loader, val_loader = train_val_dataloaders(
            dataset, args.val_split, args.batch_size, args.seed)

        logging.info("Begin training...")

        for epoch in range(args.start_epoch, args.num_epochs + 1):
            train_epoch(epoch, args, model, train_loader, criterion, optimizer)
            evaluate(args, model, val_loader, criterion)

    if args.mode == 'eval':
        logging.info("Begin model evaluation...")
        loader = DataLoader(dataset, batch_size=args.batch_size)
        evaluate(args, model, loader, criterion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=0)
    parser.add_argument('--datadir', default='.')
    parser.add_argument('--logging', default='INFO')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--features', type=int, default=12)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--classes', type=int, default=2)

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--start-epoch', type=int, default=1)
    parser_train.add_argument('--num-epochs', type=int, default=100)
    parser_train.add_argument('--seed', type=int, default=0)
    parser_train.add_argument('--val-split', type=float, default=0.2)
    parser_train.add_argument('--checkpoint', action='store_true')

    parser_eval = subparsers.add_parser('eval')

    args = parser.parse_args()

    # setup logging level
    numeric_level = getattr(logging, args.logging.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.log))
    logging.basicConfig(level=numeric_level)

    main(args)
