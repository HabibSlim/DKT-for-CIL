"""
Utility functions.
"""
import os
import torch
import random
import numpy as np

cudnn_deterministic = True


def seed_everything(seed=0):
    """
    Fixing all random seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def print_summary(metrics_list, labels_list):
    """
    Print metrics summary across incremental states

    Args:
        metrics_list: Network prediction
        labels_list:  List of labels for the given metrics
    """
    for metric, name in zip(metrics_list, labels_list):
        print('*' * 108)
        print(name)
        mean_inc_acc = []
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.2f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0:
                    avg = 100 * metric[i, :i].mean()
                    mean_inc_acc += [avg]
                    print('\tAvg.:{:5.2f}% '.format(avg), end='')
            else:
                avg = 100 * metric[i, :i + 1].mean()
                mean_inc_acc += [avg]
                print('\tAvg.:{:5.2f}% '.format(avg), end='')
            print()
        print()

        # Computing AIA across all incremental states (thus excluding the first non-incremental state)
        print('\tMean Incremental Acc.: {:5.2f}%'.format(np.mean(mean_inc_acc[1:])))
    print('*' * 108)


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
    pred = outputs

    # Top-k prediction for TAg
    hits_tag_top5 = compute_topk_acc(pred, targets, 5)
    hits_tag_top1 = compute_topk_acc(pred, targets, 1)

    return hits_tag_top5.item(), hits_tag_top1.item()
