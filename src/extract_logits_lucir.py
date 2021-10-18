"""
Extracting validation and test logits for each incremental state, for LUCIR models.
"""
import argparse
import os
import sys
import torch
import utils
import warnings

from datasets.data_loader import get_loaders
from datasets.dataset_config import dataset_config

sys.path.append(os.path.join(sys.path[0], "./lucir/"))
import modified_linear
import modified_resnet


def load_model(model_load_path, device):
    """
    Loading a fine-tuning model and normalizing it following SiW,
    while keeping only the first learned head for each task.

    Args:
        model_load_path: Serialized model full path
        device:          Torch device to be used
    """
    with torch.no_grad():
        # Loading the state dictionary
        if not os.path.isfile(model_load_path):
            print("File: %s" % model_load_path, " not found. Stopping at this task.")
            exit(0)

        # Loading the model
        base_net = torch.load(model_load_path, map_location=lambda storage, loc: storage)

    base_net.avgpool = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
    return base_net.to(device)


def parse_args(argv):
    """
    Parsing input arguments.
    """
    # Arguments
    parser = argparse.ArgumentParser(description='Extracting validation and test logits for each incremental state.')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')

    parser.add_argument('--models-dir', type=str,
                        help='Directory containing serialized models')
    parser.add_argument('--logits-outdir', type=str,
                        help='Output directory to save task logits')

    # dataset args
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=128, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=4, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')

    args, extra_args = parser.parse_known_args(argv)

    # Setting base model name
    dset_name = args.datasets[0]
    dset_name = 'cifar100_half' if dset_name == 'cif100_half' else dset_name
    args.models_base_name = "%s_s%d_k0_model_" % (dset_name, args.num_tasks)

    # Updating paths
    args.models_dir = os.path.expanduser(args.models_dir)
    args.logits_outdir = os.path.expanduser(args.logits_outdir)
    args.models_base_path = os.path.join(args.models_dir, args.models_base_name)

    # Printing input arguments
    print("Input arguments:")
    print(args)

    return args, extra_args


def extract_logits(net, loader, device):
    """
    Extracting output logits using the supplied network.

    Args:
        net:    Instantiated model
        loader: Loader for the current task
        device: Torch device to be used
    """
    all_logits = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)

    with torch.no_grad():
        net.eval()
        for images, targets in loader:
            # Forwarding through current model
            targets = targets.to(device)
            outputs = net(images.to(device))

            # Concatenating to output matrix
            all_logits = torch.cat((all_logits, outputs))
            all_labels = torch.cat((all_labels, targets))

    return all_logits, all_labels


def main(argv=None):
    """
    Extracting and saving output logits.
    """
    # Filtering EXIF warnings
    warnings.filterwarnings("ignore")

    # Parsing input arguments
    args, _ = parse_args(argv)

    # Fixing random seed
    utils.seed_everything(args.seed)

    # Selecting device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: CUDA unavailable, using CPU instead!')
        device = 'cpu'

    # Instantiating data loaders
    val_loader, _, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks, None,
                                                     args.batch_size, num_workers=8,
                                                     pin_memory=False,
                                                     force_order=True, val_only=True)

    # Creating output logits dir, if necessary
    if not os.path.exists(args.logits_outdir):
        os.makedirs(args.logits_outdir)

    # - P: Number of new classes per state
    for t, (_, P) in enumerate(taskcla):
        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Loading the network for this task
        task_model = "%s%d.pth" % (args.models_base_path, t)
        net = load_model(task_model, device)
        print("Loaded task model from: %s" % task_model)

        # Extracting and serializing validation and test logits
        for u in range(t + 1):
            # val:
            val_logits, val_labels = extract_logits(net, val_loader[u], device)
            val_logits_file = "%slogits_val_%d_%d.ckpt" % (args.logits_outdir, t, u)
            torch.save(val_logits, val_logits_file)
            val_labels_file = "%slabels_val_%d_%d.ckpt" % (args.logits_outdir, t, u)
            torch.save(val_labels, val_labels_file)

            print('Val logits shape: ', val_logits.shape)
            print('Val labels shape: ', val_labels.shape)

            # test:
            tst_logits, tst_labels = extract_logits(net, tst_loader[u], device)
            tst_logits_file = "%slogits_tst_%d_%d.ckpt" % (args.logits_outdir, t, u)
            torch.save(tst_logits, tst_logits_file)
            tst_labels_file = "%slabels_tst_%d_%d.ckpt" % (args.logits_outdir, t, u)
            torch.save(tst_labels, tst_labels_file)

            print('Test logits shape: ', tst_logits.shape)
            print('Test labels shape: ', tst_labels.shape)

        print('-' * 108)

    print("All output logits succesfully saved!")


if __name__ == '__main__':
    main()
