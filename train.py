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

from dataset import RVSC
from densenet import DilatedDenseNet
from loss import CrossEntropyLoss2d


def train(args, model):
    logging.info("Creating dataloaders...")

    input_transform = output_transform = None
    dataset = RVSC(args.datadir, input_transform, output_transform)

    # create train-val split
    # TODO: setup seed for reproducibility
    # TODO: setup structure for evaluation on validation set
    # total_images = len(dataset)
    # train_images = int((1 - args.val_split) * len(dataset))
    # train_sampler = SubsetRandomSampler(np.random.choice(total_images, train_images))
    # loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    logging.info("Construct model...")

    model.train()

    # setup cross entropy loss
    weight = torch.ones(2)
    if args.cuda:
        weight = weight.cuda()
    criterion = CrossEntropyLoss2d(weight)

    optimizer = Adam(model.parameters())

    logging.info("Begin training...")

    for epoch in range(1, args.num_epochs+1):
        epoch_loss = []

        for step, (images,labels) in enumerate(loader):
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            x = Variable(images)
            y_true = Variable(labels)
            y_pred = model(x)

            optimizer.zero_grad()
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])
            print(step, loss.data[0])

        save_path = f'tensors-{epoch:03d}.pt'
        torch.save(model.state_dict(), save_path)

def main(args):
    model = DilatedDenseNet(image_channels=1, num_init_features=12,
                            growth_rate=12, layers=8, dropout_rate=0,
                            classes=2, dilated=True)

    if args.cuda:
        model = model.cuda()

    if args.mode == 'train':
        train(args, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--datadir', default='.')
    parser.add_argument('--log', default='INFO')
    parser.add_argument('--batch-size', type=int, default=32)

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--num-epochs', type=int, default=100)
    parser_train.add_argument('--val-split', type=float, default=0.2)

    parser_eval = subparsers.add_parser('eval')

    args = parser.parse_args()

    # setup logging level
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log}')
    logging.basicConfig(level=numeric_level)

    main(args)
