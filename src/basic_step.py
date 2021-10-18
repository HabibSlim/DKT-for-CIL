"""
Training a model for the first (non-incremental) step.
"""

import argparse
import os
import time
import utils
import warnings

from copy import deepcopy
from datetime import timedelta

import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed

from torch.optim import lr_scheduler
from torchvision import models

from datasets.data_loader import get_loaders
from datasets.dataset_config import dataset_config


def parse_args(argv):
    """
    Parsing input arguments.
    """
    # Arguments
    parser = argparse.ArgumentParser(description='Training a model for the first (non-incremental) step.')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--models-dir', default='', type=str,
                        help='Output directory to save the basic step model.')

    # dataset args
    parser.add_argument('--datasets', default=['cif100'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--num-tasks', default=10, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')

    parser.add_argument('--batch-size', default=128, type=int, required=False,
                        help='Real batch size, before gradient accumulation (default=%(default)s)')
    parser.add_argument('--eff-batch-size', default=128, type=int, required=False,
                        help='Effective batch size, after gradient accumulation (default=%(default)s)')
    parser.add_argument('--test-batch-size', default=128, type=int, required=False,
                        help='Test batch size (default=%(default)s)')

    # training args
    parser.add_argument('--lr', default=0.1, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-decay', default=0.1, type=float, required=False,
                        help='Learning rate decay (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0005, type=float, required=False,
                        help='Weight decay (default=%(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float, required=False,
                        help='Momentum (default=%(default)s)')
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--patience', type=int, default=60, required=False,
                        help='Use patience while training (default=%(default)s)')

    args, extra_args = parser.parse_known_args(argv)

    args.iter_size = int(args.eff_batch_size / args.batch_size)

    # Printing input arguments
    print("Input arguments:")
    print(args)

    return args, extra_args


def main(argv=None):
    """
    Main training routine.
    """
    # Filtering EXIF warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    # Parsing input arguments
    args, _ = parse_args(argv)

    # Fixing random seed
    utils.seed_everything(args.seed)

    # Creating output model directories
    if not os.path.exists(args.models_dir):
        os.makedirs(args.models_dir)

    # Instantiating data loaders
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets,
                                                              args.num_tasks, None,
                                                              args.batch_size,
                                                              num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory,
                                                              force_order=True)
    trn_loader = trn_loader[0]
    val_loader = tst_loader[0]

    # - S: Total number of states
    # - P: Number of new classes in state 0
    S, P = taskcla[0]

    # Selecting device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: CUDA unavailable, using CPU instead!')
        device = 'cpu'

    # Creating model
    model = models.resnet18(pretrained=False, num_classes=P)
    model = model.to(device)

    # Defining loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    best_val_acc = -1
    best_epoch = 0
    best_model     = deepcopy(model)
    best_optimizer = deepcopy(optimizer)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               patience=args.patience,
                                               factor=args.lr_decay)

    # Training
    print("-" * 20)
    print("Training...")
    starting_time = time.time()

    for epoch in range(args.nepochs):
        # Training the model
        optimizer.zero_grad()
        model.train()

        loss = None
        running_loss = 0.0
        nb_batches = 0
        new_best = False

        for i, (images, targets) in enumerate(trn_loader):
            nb_batches += 1
            images, targets = images.to(device), targets.to(device)

            # Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.data /= args.iter_size
            loss.backward()
            running_loss += loss.data.item()

            if (i + 1) % args.iter_size == 0:
                optimizer.step()
                optimizer.zero_grad()
        scheduler.step(loss.cpu().data.numpy())

        # Evaluating the model
        model.eval()
        total_top5_hits, total_top1_hits, N = 0, 0, 0
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            top5_hits, top1_hits = utils.calculate_metrics(outputs, targets)

            total_top5_hits += top5_hits
            total_top1_hits += top1_hits
            N += len(targets)

        top1_avg = float(total_top1_hits) / N
        top5_avg = float(total_top5_hits) / N

        if top1_avg > best_val_acc:
            best_val_acc   = top1_avg
            best_model     = deepcopy(model)
            best_optimizer = deepcopy(optimizer)
            best_epoch     = epoch
            new_best       = True

        current_elapsed_time = time.time() - starting_time
        print('{:03}/{:03} | {} | Train : loss = {:.4f} | Val : acc@1 = {:.3f}% ; acc@5 = {:.3f}%  {:s}'.
              format(epoch + 1, args.nepochs,
                     timedelta(seconds=round(current_elapsed_time)),
                     running_loss / nb_batches,
                     top1_avg*100, top5_avg*100,
                     "*" if new_best else ""))

    print('Finished training, elapsed training time : {}'.format(
        timedelta(seconds=round(time.time() - starting_time))))

    # Training finished
    if best_model is not None:
        saved_model = os.path.join(args.models_dir, 'task0.ckpt')

        print('Saved best model at: [' + saved_model + ']')
        state = {
            'epoch':        best_epoch,
            'state_dict':   best_model.state_dict(),
            'optimizer':    best_optimizer.state_dict(),
            'best_val_acc': best_val_acc
        }
        print('best acc = ' + str(best_val_acc))
        torch.save(state, saved_model)


if __name__ == "__main__":
    main()
