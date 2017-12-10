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
from torchvision import transforms, utils

from dataset import SegmentationDataset, normalize
from densenet import DilatedDenseNet
from loss import CrossEntropyLoss2d


def train_val_dataloaders(dataset, val_split=0.2, train_batch_size=32, val_batch_size=1, seed=0):
    np.random.seed(seed)   # crucial: controls train-val split

    ntotal = len(dataset)
    ntrain = int((1 - val_split) * len(dataset))

    train_indices = np.random.choice(ntotal, ntrain, replace=False)
    train_sampler = SubsetRandomSampler(train_indices)

    val_indices = np.setdiff1d(np.arange(ntotal), train_indices, assume_unique=True)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=train_batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=val_batch_size, sampler=val_sampler)

    return train_loader, val_loader


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

    if args.checkpoint > 0 and epoch % args.checkpoint == 0:
        save_path = 'tensors-{:03d}.pt'.format(epoch)
        torch.save(model.state_dict(), save_path)


def evaluate(args, model, loader, criterion, save=False):
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

        if save:
            utils.save_image(images.cpu(), "images-{:03d}.png".format(step),
                             normalize=True)
            utils.save_image(y_pred.data[:,1,:,:][:,None,:,:].cpu(),
                             "masks-{:03d}.png".format(step), normalize=True)

    total_loss = sum(n*x for n,x in zip(sizes, losses)) / sum(sizes)
    logging.info("Loss: {}".format(loss))


def main(args):
    logging.info("Arguments for execution:")
    for k,v in vars(args).items():
        logging.info("{} = {}".format(k, v))

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

    logging.info("Number of trainable parameters: {}".format(
        model.num_trainable_parameters()))
    logging.info(str(model))

    if args.ngpu > 0:
        model = nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        model = model.cuda()

    # setup cross entropy loss
    weight = torch.ones(2)
    if args.ngpu > 0:
        weight = weight.cuda()
    criterion = CrossEntropyLoss2d(weight)

    optimizer = Adam(model.parameters())

    if args.infile:
        model.load_state_dict(torch.load(args.infile))

    if args.mode == 'train':
        train_loader, val_loader = train_val_dataloaders(dataset,
            args.val_split, args.batch_size, args.val_batch_size, args.seed)

        logging.info("Begin training...")

        for epoch in range(args.start_epoch, args.num_epochs + 1):
            train_epoch(epoch, args, model, train_loader, criterion, optimizer)
            if args.val_split > 0:
                evaluate(args, model, val_loader, criterion)

        torch.save(model.cpu().state_dict(), args.outfile)

    if args.mode == 'eval':
        logging.info("Begin model evaluation...")
        loader = DataLoader(dataset, batch_size=args.batch_size)
        evaluate(args, model, loader, criterion, save=args.save_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=-1,
        help="Number of GPUs to use; 0 for CPU")
    parser.add_argument('--datadir', default='.',
        help="Data directory containing images/ and masks/ subdirectories.")
    parser.add_argument('--logging', default='INFO',
        help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument('--batch-size', type=int, default=32,
        help="Training batch size")
    parser.add_argument('--features', type=int, default=12,
        help="DenseNet growth rate")
    parser.add_argument('--layers', type=int, default=8,
        help="Depth of DenseNet")
    parser.add_argument('--dropout', type=float, default=0.0,
        help="Dropout probability, 0 to turn off")
    parser.add_argument('--classes', type=int, default=2,
        help="Number of classes in segmentation mask")
    parser.add_argument('--infile', default='',
        help="File for initial model weights")
    parser.add_argument('--outfile', default='tensors-final.pt',
        help="File to save final model weights")

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--seed', type=int, default=0,
        help="Random seed determining train-val split; keep SAME across runs")
    parser_train.add_argument('--num-epochs', type=int, default=100,
        help="Total epochs to train")
    parser_train.add_argument('--val-split', type=float, default=0.2,
        help="Train-validation split; 0 to train on all data")
    parser_train.add_argument('--checkpoint', type=int, default=0,
        help="Interval to write model weights (in epochs); 0 to disable")
    parser_train.add_argument('--start-epoch', type=int, default=1,
        help="Offset for epoch numbering; for restarting training")
    parser_train.add_argument('--val-batch-size', type=int, default=-1,
        help="Batch size for validation at end of each epoch")

    parser_eval = subparsers.add_parser('eval')
    parser_eval.add_argument('--save-images', action='store_true',
        help="Write predicted masks to file")

    args = parser.parse_args()

    # setup logging level
    numeric_level = getattr(logging, args.logging.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.log))
    logging.basicConfig(level=numeric_level)

    # automatically set GPU usage
    ngpus = torch.cuda.device_count()
    if args.ngpu < 0:
        args.ngpu = ngpus
    elif args.ngpu > ngpus:
        logging.warning("System only has {} GPUs; {} specified.".format(
            ngpus, args.ngpu))
        args.ngpu = ngpus
    logging.info("Using {} GPUs.".format(ngpus))

    # default validation batch size to training batch size
    if args.mode == 'train' and args.val_batch_size < 0:
        args.val_batch_size = args.batch_size

    main(args)
