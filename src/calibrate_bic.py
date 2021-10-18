"""
Calibrating BiC coefficients using pre-extracted logits from incremental models.
"""
import argparse
from collections import Counter
from copy import deepcopy

import numpy as np
import os
import time
import torch
import utils
import warnings

from adaptive_bic import BiCNet, CalibrationMethod


def parse_args(argv):
    """
    Parsing input arguments.
    """
    # Arguments
    parser = argparse.ArgumentParser(description='Calibrating BiC coefficients using pre-extracted'
                                                 'logits from incremental models')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')

    parser.add_argument('--logits-dir', type=str,
                        help='Directory containing serialized logits and associated labels')
    parser.add_argument('--bic-models-outdir', type=str,
                        help='Output directory to save BiC parameters')
    parser.add_argument('--bic-models-dir', default='', type=str,
                        help='Input directory containing BiC parameters.'
                             'If specified, will run in inference mode only.')

    parser.add_argument('--eval-only', action='store_true', required=False,
                        help='Only evaluate the model from the raw logits (default=%(default)s)')
    parser.add_argument('--recalibrate', action='store_true', required=False,
                        help='Recalibrate using the formula from SiW (default=%(default)s)')

    # dataset args
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=128, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=4, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')

    # training args
    parser.add_argument('--lr', default=0.001, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--patience', type=int, required=False,
                        help='Use patience while training (default=%(default)s)')

    parser.add_argument('--gridsearch', default=False, type=bool, required=False,
                        help='Use gridsearch to optimize the BiC layers (default=%(default)s)')
    parser.add_argument('--step', default=0.01, type=float, required=False,
                        help='Step for gridsearch (default=%(default)s)')

    # For a description of each method, check the CalibrationMethod class definition
    calib_methods = {'forward-all':  CalibrationMethod.FORWARD_ALL,
                     'forward-past': CalibrationMethod.FORWARD_PAST,
                     'forward-last': CalibrationMethod.FORWARD_LAST,
                     'adaptive':     CalibrationMethod.ADAPTIVE}
    parser.add_argument('--calib-method', default='adaptive', type=str, choices=list(calib_methods.keys()),
                        help='Calibration method to use (default=%(default)s)')

    args, extra_args = parser.parse_known_args(argv)

    # Calibration method
    args.calib_method = calib_methods[args.calib_method]

    # Updating paths
    args.logits_dir = os.path.expanduser(args.logits_dir)
    if args.bic_models_outdir:
        args.bic_models_outdir = os.path.expanduser(args.bic_models_outdir)
    if args.bic_models_dir:
        args.bic_models_dir = os.path.expanduser(args.bic_models_dir)

    # Printing input arguments
    print("Input arguments:")
    print(args)

    return args, extra_args


def read_logits(file_path):
    """
    Reading serialized output logits.

    Args:
        file_path:  File path for the serialized output logits
    """
    # Loading output logits
    if not os.path.isfile(file_path):
        print("File: %s" % file_path, " not found. Stopping at this task.")
        exit(0)

    return torch.load(file_path, map_location=lambda storage, loc: storage)


def read_labels(file_path):
    """
    Reading serialized output labels, and packing them in batches.

    Args:
        file_path:  File path for the serialized output logits
    """
    # Loading output logits
    if not os.path.isfile(file_path):
        print("File: %s" % file_path, " not found. Stopping at this task.")
        exit(0)

    return torch.load(file_path, map_location=lambda storage, loc: storage).type(torch.LongTensor)


def compute_topk_acc(pred, targets, topk):
    """
    Computing top-k accuracy given prediction and target vectors.

    Args:
        pred:    Network prediction
        targets: Ground truth labels
        topk:    k value
    """
    topk = min(topk, pred.shape[1])
    _, pred = pred.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    hits_tag = correct[:topk].reshape(-1).float().sum(0)

    return hits_tag


def calculate_metrics(outputs, targets):
    """
    Computing top-1 and top-5 task-agnostic accuracy metrics.

    Args:
        outputs: Network outputs list
        targets: Ground truth labels
    """
    if isinstance(outputs, list):
        outputs = torch.cat(outputs, dim=1)

    # Top-k prediction for TAg
    hits_tag_top5 = compute_topk_acc(outputs, targets, 5)
    hits_tag_top1 = compute_topk_acc(outputs, targets, 1)

    return hits_tag_top5.item(), hits_tag_top1.item()


def average_scores(t, val_logits, bic_net, device):
    """
    Pre-compute mean top-1 metrics for recalibration.

    Args:
        t:            Current task index
        val_logits:   Data loader to use
        bic_net:      Current recalibration model
        device:       Device to use
    """
    with torch.no_grad():
        all_max = torch.tensor([]).to(device)

        for u in range(t + 1):
            # Correcting logits with the current model
            logits = bic_net.forward(val_logits[u].to(device))
            logits = torch.cat(logits, dim=1)
            # Computing top-1 values alongside each axis
            max_vals, _ = logits.max(dim=1)

            all_max = torch.cat((max_vals, all_max))

        # Averaging values
        return float(all_max.mean())


def recalibrate(t, metrics, outputs):
    """
    Recalibrating the output prediction vectors, using mean top-1 statistics (following [1].)

        [1] Belouadah et al., "Initial classifier weights replay for memoryless class incremental learning"

    Args:
        t:        Current task index
        metrics:  Recalibration metrics to apply
        outputs:  Network outputs tensor
    """
    calib_out = []
    for i, x in enumerate(torch.chunk(outputs, t+1, dim=1)):
        calib_fact = metrics[t] / metrics[i]
        calib_out += [calib_fact * x]

    return torch.cat(calib_out, dim=1)


def bic_eval(bic_net, tst_logits, tst_labels, batch_size, use_bic, device, metrics=None):
    """
    Returns the top-5 and top-1 task-agnostic accuracy metrics for the BiC-corrected model,
    evaluated on test data.

    Args:
        bic_net:      BiC network to use for evaluation
        tst_logits:   Logits extracted on the test set (for this task)
        tst_labels:   Test loader (only used for the targets)
        batch_size:   Batch size to use
        use_bic:      Forward through the BiC layer to evaluate
        device:       Torch device to be used
        metrics:      Recalibration metrics to apply
    """
    with torch.no_grad():
        total_acc_tag_top5, total_acc_tag_top1, N = 0, 0, 0

        # Splitting logits/labels into batches
        tst_logits = torch.split(tst_logits, batch_size)
        tst_labels = torch.split(tst_labels, batch_size)

        for logits, targets in zip(tst_logits, tst_labels):
            logits = logits.to(device)
            targets = targets.to(device)

            # Applying the recalibration factor
            if metrics:
                logits = recalibrate(bic_net.t-1, metrics, logits)

            # Applying BiC correction
            if use_bic:
                outputs = bic_net.forward(logits)
            else:
                outputs = logits

            hits_tag_top5, hits_tag_top1 = calculate_metrics(outputs, targets)

            # Task Agnostic top-5 accuracies
            total_acc_tag_top5 += hits_tag_top5
            total_acc_tag_top1 += hits_tag_top1
            N += len(targets)

    return total_acc_tag_top5 / N, total_acc_tag_top1 / N


def bic_train(bic_net, bic_loader, n_epochs, lr, patience, device, metrics=None):
    """
    Train for a single task.

    Args:
        bic_net:     BiC network to use for evaluation
        bic_loader:  Validation loader for the current task
        n_epochs:    Number of epochs to train the BiC layer over
        lr:          Learning rate
        patience:    False to disable learning rate diminution after train accuracy stagnation
        device:      Device to use
        metrics:     Recalibration metrics to apply
    """
    print("Training BiC layer...")

    # Setting trainable parameters
    bic_net.train_init()

    # Using Adam as the main optimizer (better than SGD for this task)
    bic_optimizer = torch.optim.Adam(bic_net.trainable_p(), lr=lr, weight_decay=0.005)
    bic_net.print_parameters()

    # Stopping after max_patience epochs if accuracy does not improve
    if not patience:
        patience = n_epochs * 2
    current_patience = 0
    min_lr = 1e-5
    prev_acc = -1

    # Epochs loop
    class_count = False
    best_bic = deepcopy(bic_net)
    best_acc = 0

    # Accuracy metric to use for training evaluation
    topk = 5

    for e in range(n_epochs):
        # Train bias correction layers
        clock0 = time.time()
        total_loss, total_acc = 0, 0
        N = 0

        dic_c = Counter()
        for logits, targets in bic_loader:
            logits = logits.to(device)
            targets = targets.to(device)

            # (for the new class: alpha=1, beta=0.)
            all_outs = bic_net.forward(logits)
            pred_all_classes = torch.cat(all_outs, dim=1)

            if metrics:
                pred_all_classes = recalibrate(bic_net.t-1, metrics, pred_all_classes)

            # Outputs from all tasks are modified, except for the new task (alpha=1,beta=0 fixed for last task)
            loss = torch.nn.functional.cross_entropy(pred_all_classes, targets)
            loss = bic_net.beta_l2(loss, lambd=0.1)

            # Log
            total_loss += loss.item() * len(targets)
            total_acc += compute_topk_acc(pred_all_classes, targets, topk)

            # Backward
            bic_optimizer.zero_grad()
            loss.backward()
            bic_optimizer.step()

            N += logits.shape[0]
            if not class_count:
                for t in targets:
                    dic_c[int(t)] += 1

        # scheduler.step(e)
        clock1 = time.time()

        if not class_count:
            print("Seen classes count :: ", dic_c)
            class_count = True

        # Printing training status
        train_acc = 100 * (total_acc / N)
        if (e % 25) == 0 or train_acc > best_acc:
            bst = "*" if (train_acc > best_acc) else " "
            print(bst, '| Epoch {:3d}, time={:5.1f}s | Train: loss={:.3f}, Top-{:d} Train TAg acc={:5.2f}% |'
                  .format(e + 1, clock1 - clock0, total_loss / N, topk, train_acc))

        # Updating patience
        if train_acc <= prev_acc:
            current_patience += 1
        else:
            current_patience = 0
        prev_acc = train_acc

        if current_patience > patience:
            if lr / 10 > min_lr:
                lr /= 10
                old_lr = bic_optimizer.param_groups[0]['lr']
                bic_optimizer.param_groups[0]['lr'] = lr
                print('| Epoch {:3d}: train_acc not improving, lr changed to {:.8f} from {:.8f}'
                      .format(e + 1, lr, old_lr))
                current_patience = 0

        # Saving the best model
        if train_acc > best_acc:
            best_bic = deepcopy(bic_net)
            best_acc = train_acc

    print()
    best_bic.print_parameters()

    return best_bic


def bic_train_grid(bic_net, bic_loader, grid_p, method, device):
    """
    Train the model using a gridsearch approach.

    Args:
        bic_net:     BiC network to use for evaluation
        bic_loader:  Validation loader for the current task
        grid_p:      Grid precision
        method:      Forward method to use
        device:      Device to use
    """
    print("Training BiC layer...")

    if method == CalibrationMethod.ADAPTIVE:
        print("Gridsearch incompatible with adaptiveBiC.")
        exit(-1)

    # Epochs loop
    class_count = False
    best_bic = deepcopy(bic_net)
    best_acc = 0

    # Generating parameter spaces
    alpha_p = np.arange(0.5, 1., grid_p)
    beta_p = np.arange(-0.5, 0.5, grid_p)

    # Printing first parameters
    bic_net.print_parameters()

    it_tot = len(alpha_p) * len(beta_p)
    it_cur = 0

    with torch.no_grad():
        for alpha in alpha_p:
            it_per = (it_cur / it_tot) * 100
            print("Explored %d %% of the parameter space.\n" % it_per)
            for beta in beta_p:
                it_cur += 1

                # Train bias correction layers
                N = 0
                total_acc = 0

                dic_c = Counter()
                for logits, targets in bic_loader:
                    logits = logits.to(device)
                    targets = targets.to(device)

                    # Setting alpha and beta values on the last layer
                    bic_net.set_alpha(alpha)
                    bic_net.set_beta(beta)

                    # Forwarding through the current model
                    all_outs = bic_net.forward(logits)
                    pred_all_classes = torch.cat(all_outs, dim=1)

                    # Compute accuracy of current model
                    total_acc += compute_topk_acc(pred_all_classes, targets, 5)

                    N += logits.shape[0]
                    for t in targets:
                        dic_c[int(t)] += 1

                train_acc = 100 * (total_acc / N)

                if not class_count:
                    print("Seen classes count :: ", dic_c)
                    class_count = True

                # Saving the best model
                if train_acc > best_acc:
                    print("New best model found! Top-5 Acc: {:5.2f}%".format(train_acc))
                    best_bic = deepcopy(bic_net)
                    best_acc = train_acc

    print()
    best_bic.print_parameters()
    print("-----------")

    return best_bic


def make_bic_loader(t, val_logits, val_labels, batch_size):
    """
    Mixing all validation logits from previous and new classes.
        Previous and old classes logits must all have a number of dimensions
        equal to the total number of seen classes at time t.
        Thus, previous logits must be obtained by forwarding all validation sets
        through the model at time t.

    Args:
        t:            Current test task identifier
        val_logits:   List of validation logits for all tasks val. sets,
                      from the model at time t.
        val_labels:   Validation loader for all tasks.
        batch_size:   Current batch size
    """

    new_val_tuples = []
    new_cls_batches = 0

    # Fetching targets/logits from new classes
    logits = torch.split(val_logits[t], batch_size)
    labels = torch.split(val_labels[t], batch_size)
    for lg, tg in zip(logits, labels):
        new_val_tuples += [(lg, tg)]
        new_cls_batches += 1

    # Fetching targets/logits from old classes
    # -> Create logits/targets tensor, filter by class count and concatenate, and at the end split into batches
    old_val_tuples = []

    for k in range(t):
        # Splitting logits in batches
        logits = torch.split(val_logits[k], batch_size)
        labels = torch.split(val_labels[k], batch_size)

        old_cls_batches = 0
        for lg, tg in zip(logits, labels):
            old_val_tuples += [(lg, tg)]
            old_cls_batches += 1

    return old_val_tuples + new_val_tuples


def main(argv=None):
    """
    Calibrating BiC parameters for each 1..n-1 task.
    """
    # Filtering EXIF warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    # Parsing input arguments
    args, _ = parse_args(argv)
    if args.bic_models_dir:
        print("BiC parameters supplied: running in eval mode using [%s]." % args.bic_models_dir)

    # Fixing random seed
    utils.seed_everything(args.seed)

    # Selecting device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: CUDA unavailable, using CPU instead!')
        device = 'cpu'

    # Average metrics for recalibration
    # mapping t :-> u(M_t) mean of top-1 predictions in model t
    avg_metrics = {}

    # Storing evaluation metrics
    max_task = args.num_tasks
    acc_tag_top5 = (np.zeros((max_task, max_task)), np.zeros((max_task, max_task)))
    acc_tag_top1 = (np.zeros((max_task, max_task)), np.zeros((max_task, max_task)))

    # Initializing BicNet
    bic_net = BiCNet(device, args.calib_method).to(device)

    # Main loop
    tstart = time.time()

    for t in range(args.num_tasks):
        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Adding a head to the BiC layer
        bic_net.add_head()

        # Loading validation logits for this task
        if t > 0 or args.recalibrate:
            val_logits = []
            for u in range(t + 1):
                val_logits_file = "%s/logits/logits_val_%d_%d.ckpt" % (args.logits_dir, t, u)
                val_logits += [read_logits(val_logits_file)]
                print("Loaded val logits from: %s" % val_logits_file)

            # Computing mean top-1 scores
            if args.recalibrate:
                avg_metrics[t] = average_scores(t, val_logits, bic_net, device)

            # Training the layer, if not in evaluation mode
            if t > 0 and not (args.bic_models_dir or args.eval_only):
                # Loading validation labels for model t
                val_labels = []
                for u in range(t + 1):
                    val_labels_file = "%s/logits/labels_val_%d_%d.ckpt" % (args.logits_dir, t, u)
                    val_labels += [read_labels(val_labels_file)]
                    print("Loaded val logits from: %s" % val_labels_file)

                # Training the layer
                bic_loader = make_bic_loader(t, val_logits, val_labels, args.batch_size)

                if args.gridsearch:
                    # TODO: Remove gridsearch code
                    bic_net = bic_train_grid(bic_net, bic_loader,
                                             args.step, args.calib_method,
                                             device=device)
                else:
                    bic_net = bic_train(bic_net, bic_loader,
                                        args.nepochs, args.lr,
                                        args.patience,
                                        device=device, metrics=avg_metrics)

        # Evaluating the model on the test set
        for u in range(t + 1):
            # Loading output logits of model t on test set u
            tst_logits_file = "%s/logits/logits_tst_%d_%d.ckpt" % (args.logits_dir, t, u)
            tst_logits = read_logits(tst_logits_file)

            # Loading labels for model t on test set u
            tst_labels_file = "%s/logits/labels_tst_%d_%d.ckpt" % (args.logits_dir, t, u)
            tst_labels = read_labels(tst_labels_file)

            # Loading a trained BiC model
            if args.bic_models_dir and not args.eval_only:
                bic_model_file = "%s/bic_model_%d.ckpt" % (args.bic_models_dir, t)
                bic_state_dict = torch.load(bic_model_file, map_location=lambda storage, loc: storage)
                bic_net.load_state_dict(bic_state_dict)

            if not args.eval_only:
                # -> with BiC:
                acc_tag_top5[0][t, u], acc_tag_top1[0][t, u] = bic_eval(bic_net,
                                                                        tst_logits, tst_labels,
                                                                        args.batch_size,
                                                                        use_bic=True, device=device,
                                                                        metrics=avg_metrics)

                print('>>> With    BiC on task {:2d} : | Top-5 TAg acc={:5.2f}% | Top-1 TAg acc={:5.2f}% <<<'.
                      format(u, 100 * acc_tag_top5[0][t, u], 100 * acc_tag_top1[0][t, u]))

            # -> without BiC:
            acc_tag_top5[1][t, u], acc_tag_top1[1][t, u] = bic_eval(bic_net,
                                                                    tst_logits, tst_labels,
                                                                    args.batch_size,
                                                                    use_bic=False, device=device,
                                                                    metrics=avg_metrics)
            print('>>> Without BiC on task {:2d} : | Top-5 TAg acc={:5.2f}% | Top-1 TAg acc={:5.2f}% <<<'.
                  format(u, 100 * acc_tag_top5[1][t, u], 100 * acc_tag_top1[1][t, u]))

        # Saving learned parameters
        if args.bic_models_outdir and not args.bic_models_dir:
            bic_model_file = "%s/bic_model_%d.ckpt" % (args.bic_models_outdir, t)
            torch.save(bic_net.state_dict(), bic_model_file)

    # Print Summary
    if not args.eval_only:
        utils.print_summary([acc_tag_top5[0], acc_tag_top1[0]], ['Top-5 TAg Acc (BiC)', 'Top-1 TAg Acc (BiC)'])
    utils.print_summary([acc_tag_top5[1], acc_tag_top1[1]], ['Top-5 TAg Acc (raw)', 'Top-1 TAg Acc (raw)'])
    print('[Elapsed time = {:.1f} mn]'.format((time.time() - tstart) / 60))
    print('Done!')

    print('-' * 108)
    print("All tasks optimized and evaluated.")


if __name__ == '__main__':
    main()
