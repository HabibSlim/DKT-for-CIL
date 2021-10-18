"""
Extracting validation and test logits for each incremental state.
"""
import argparse
import os
import torch
import torchvision
import utils
import warnings

from enum import Enum

from datasets.data_loader import get_loaders
from datasets.dataset_config import dataset_config


class ModelType(Enum):
    """
    Defining method-specific models.

        [1] Hou et al., "Learning a unified classifier incrementally via rebalancing"
        [2] Belouadah et al., "Initial classifier weights replay for memoryless class incremental learning"
    """
    BASE = 0,
    CUSTOM_INIT_REPLAY = 1,  # [2]: keeping only first-learned classifiers
    CUSTOM_SIW = 2,          # [2]: normalizing classifier weights and keeping only first-learned classifiers
    CUSTOM_FT = 3            # simple finetuning with no regularization


# Grouping methods
CUSTOM_MODELS  = [ModelType.CUSTOM_INIT_REPLAY, ModelType.CUSTOM_SIW, ModelType.CUSTOM_FT]
SIW_METHODS    = [ModelType.CUSTOM_INIT_REPLAY, ModelType.CUSTOM_SIW]


def load_model(state, base_net, model_load_path, model_type, device):
    """
    Loading a serialized model (FACIL/LwF-compatible/LUCIR-compatible/torchvision-compatible)

    Args:
        state:
            t:  Incremental state index
            S:  Total number of states
            P:  Number of new classes in each state (should be constant)
        base_net:        Previously loaded model
        model_load_path: Serialized model full path
        model_type:      Accomodate for loading method-specific models
        device:          Torch device to be used
    """
    def normalize_mat(W_):
        """
        Normalizing a weight matrix W, class-wise.
        """
        for i in range(W_.shape[0]):
            mu = torch.mean(W_[i])
            std = torch.std(W_[i])
            W_[i] = (W_[i] - mu) / std

    def get_heads(net, P_):
        """
        Fetching the heads of a network.

        Args:
            net: Network
            P_:  Number of classes in each head
        """
        W_, b_ = net.fc.parameters()

        W_split = torch.split(W_, P_)
        b_split = torch.split(b_, P_)

        return list(zip(W_split, b_split))

    t, S, P = state

    with torch.no_grad():
        # Instantiating the base network
        first_init = False
        if base_net is None:
            first_init = True

            if model_type in CUSTOM_MODELS:
                base_net = torchvision.models.resnet18(pretrained=False, num_classes=0)

            if model_type in SIW_METHODS:
                base_net.init_heads = []

        # Adding new heads
        if model_type in CUSTOM_MODELS:
            base_net.fc = torch.nn.Linear(512, (t+1)*P)

        # Loading the state dictionary
        if not os.path.isfile(model_load_path):
            print("File: %s" % model_load_path, " not found. Stopping at this task.")
            exit(0)

        # Loading model weights
        state_dict = torch.load(model_load_path, map_location=lambda storage, loc: storage)
        if model_type in CUSTOM_MODELS:
            base_net.load_state_dict(state_dict['state_dict'])
        else:
            base_net.set_state_dict(state_dict)

        # Applying SiW - specific methods
        if model_type in SIW_METHODS:
            # Normalizing heads weights, class-wise
            if model_type == ModelType.CUSTOM_SIW:
                for head in get_heads(base_net, P):
                    W, b = head
                    normalize_mat(W)

                # DEBUG
                for j, head in enumerate(get_heads(base_net, P)):
                    W, b = head
                    print("W[%d]::shape = " % j, W.shape)
                    print("W[%d]::mean = " % j, round(float(torch.mean(W)), 3))
                    print("W[%d]::std = " % j, round(float(torch.std(W)), 3))
                    print()
                # ----------------------------

            # Saving the initial head learned for the last task
            W, b = get_heads(base_net, P)[-1]
            base_net.init_heads += [(torch.clone(W), torch.clone(b))]

            # Restoring all of the initial heads
            if not first_init:
                W, b = base_net.fc.parameters()
                for j, init_head in enumerate(base_net.init_heads):
                    W_0, b_0 = init_head
                    W.data[j * P:(j + 1) * P, ] = W_0
                    b.data[j * P:(j + 1) * P]   = b_0

            # DEBUG
            W, b = base_net.fc.parameters()
            print("W::shape", W.data.shape)

            print("Number of heads: %d " % len(get_heads(base_net, P)))
            for j, (head, init_head) in enumerate(zip(get_heads(base_net, P), base_net.init_heads)):
                W, _ = head
                print(" W[%d] :: %d x %d" % (j, W.shape[0], W.shape[1]))
                print(" W[%d] :: " % j, [round(float(v), 3) for v in W[0][30:45]] + ["..."])

                W, _ = init_head
                print("*W[%d] :: " % j, [round(float(v), 3) for v in W[0][30:45]] + ["..."])
                print()
            ###########

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
    parser.add_argument('--models-base-name', type=str,
                        help='Base serialized PyTorch task model name')

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

    # model args
    parser.add_argument('--network', default='resnet18', type=str,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")

    model_types = {'base':     ModelType.BASE,
                   'ft':       ModelType.CUSTOM_FT,
                   'siw-init': ModelType.CUSTOM_INIT_REPLAY,
                   'siw':      ModelType.CUSTOM_SIW}
    parser.add_argument('--model-type', type=str, default='base',
                        help='Use a specific model type when loading checkpoints')

    args, extra_args = parser.parse_known_args(argv)

    # Base model to use
    args.model_type = model_types[args.model_type]

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
        # Defining a feature extractor
        net.eval()

        for images, targets in loader:
            # Forwarding through current model
            targets = targets.to(device)
            outputs = net(images.to(device))

            if isinstance(outputs, list):
                outputs = torch.cat(outputs, dim=1)

            # Concatenating to output matrix
            all_logits = torch.cat((all_logits, outputs))
            all_labels = torch.cat((all_labels, targets))

    return all_logits, all_labels


def main(argv=None):
    """
    Extracting and saving output logits.
    """
    # Filtering EXIF warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

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
    _, val_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks, None,
                                                     args.batch_size, num_workers=args.num_workers,
                                                     pin_memory=args.pin_memory,
                                                     force_order=True)

    # Creating output logits dir, if necessary
    if not os.path.exists(args.logits_outdir):
        os.makedirs(args.logits_outdir)

    # Main loop
    net = None

    # - P: Number of new classes per state
    for t, (_, P) in enumerate(taskcla):
        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Loading the network for this task
        task_model = "%s%d.ckpt" % (args.models_base_path, t)
        net = load_model((t, args.num_tasks, P),
                         net, task_model, args.model_type, device)
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
