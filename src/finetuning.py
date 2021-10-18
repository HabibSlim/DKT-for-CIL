"""
Finetuning a base model across multiple incremental steps.
"""

import argparse
import os
import time
from copy import deepcopy

import utils
import warnings

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
    parser = argparse.ArgumentParser(description='Finetuning a base model across multiple incremental steps.')

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

    # Checking directory folder
    if not os.path.exists(args.models_dir):
        print("Model directory [%s] not found. Exiting." % args.models_dir)
        exit(-1)

    # Selecting device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: CUDA unavailable, using CPU instead!')
        device = 'cpu'

    # Instantiating data loaders
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets,
                                                              args.num_tasks, None,
                                                              args.batch_size,
                                                              num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory,
                                                              force_order=True)

    # Storing evaluation metrics
    max_task = len(taskcla)

    # Main loop
    tstart = time.time()

    # - P: Number of new classes per state
    for t, (_, P) in enumerate(taskcla):
        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        batch_lr = args.lr / (t + 1)

        nb_old_cls = P if t == 0 else t * P
        nb_new_cls = P

        # Loading previous batch model
        prev_t = max(t-1, 0)
        model_load_path = os.path.join(args.models_dir, 'task%d.ckpt' % prev_t)
        model_ft = models.resnet18(pretrained=False, num_classes=nb_old_cls).to(device)

        print('Loading saved model from ' + model_load_path)
        state = torch.load(model_load_path, map_location=lambda storage, loc: storage)
        model_ft.load_state_dict(state['state_dict'])

        if t > 0:
            model_ft.fc = nn.Linear(512, nb_old_cls + nb_new_cls).to(device)

            # Defining Loss and Optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer_ft = optim.SGD(model_ft.parameters(),
                                     lr=batch_lr,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft,
                                                       patience=args.patience,
                                                       factor=args.lr_decay)

            best_val_acc = -1
            best_model = deepcopy(model_ft)

            # Training
            starting_time = time.time()

            for epoch in range(args.nepochs):
                # Training the model
                optimizer_ft.zero_grad()
                model_ft.train()

                loss = None
                running_loss = 0.0
                nb_batches = 0

                for i, (images, targets) in enumerate(trn_loader[t]):
                    nb_batches += 1
                    images, targets = images.to(device), targets.to(device)

                    # Forward + backward + optimize
                    outputs = model_ft(images)
                    loss = criterion(outputs, targets)
                    loss.data /= args.iter_size
                    loss.backward()
                    running_loss += loss.data.item()

                    if (i + 1) % args.iter_size == 0:
                        optimizer_ft.step()
                        optimizer_ft.zero_grad()

                scheduler.step(loss.cpu().data.numpy())

                current_elapsed_time = time.time() - starting_time
                print('{:03}/{:03} | {} | Train : loss = {:.4f} '.
                      format(epoch + 1, args.nepochs,
                             timedelta(seconds=round(current_elapsed_time)),
                             running_loss / nb_batches))

                # Evaluating the model on the test set
                model_ft.eval()
                total_top1_hits, N = 0, 0
                for u in range(t + 1):
                    for images, targets in tst_loader[u]:
                        images, targets = images.to(device), targets.to(device)
                        outputs = model_ft(images)
                        _, top1_hits = utils.calculate_metrics(outputs, targets)

                        total_top1_hits += top1_hits
                        N += len(targets)

                top1_avg = float(total_top1_hits)/N
                if top1_avg > best_val_acc:
                    best_val_acc   = top1_avg
                    best_model     = deepcopy(model_ft)

            # Saving the model for this state
            saved_model = os.path.join(args.models_dir, 'task%d.ckpt' % t)
            print('Saved best model at: [' + saved_model + ']')
            state = {
                'state_dict': best_model.state_dict()
            }
            torch.save(state, saved_model)

    # Final output
    print('[Elapsed time = {:.1f} mn]'.format((time.time() - tstart) / 60))
    print('Done!')

    print('-' * 108)
    print("All tasks optimized and evaluated.")


if __name__ == "__main__":
    main()
