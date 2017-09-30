#!/usr/bin/env python

from __future__ import division, print_function

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import RVSC
from densenet import DilatedDenseNet
from loss import CrossEntropyLoss2d

def train(args, model):
    model.train()

    input_transform = output_transform = None
    dataset = RVSC(args.datadir, input_transform, output_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    weight = torch.ones(2)
    if args.cuda:
        weight = weight.cuda()
    criterion = CrossEntropyLoss2d(weight)

    optimizer = Adam(model.parameters())

    for epoch in range(1, args.num_epochs+1):
        epoch_loss = []

        for step, (images,labels) in enumerate(loader):
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            x = torch.Variable(images)
            y_true = torch.Variable(labels)
            y_pred = model(x)

            optimizer.zero_grad()
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])
            print(loss.data[0])

def main(args):
    model = DilatedDenseNet(image_channels=1, num_init_features=12,
                            growth_rate=12, layers=8, dropout_rate=0,
                            classes=2, dilated=True)

    if args.cuda:
        model = model.cuda()

    if args.mode == 'train':
        train(args, model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--datadir', type=str)

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--num-epochs', type=int, default=100)
    parser_train.add_argument('--batch-size', type=int, default=1)

    parser_eval = subparsers.add_parser('eval')

    main(parser.parse_args())
