# coding=utf-8
"""
Extracting validation and test logits for each incremental state, for LwF TF models.
    Notes: This code is written in Python 2.7, and Tensorflow 1.8.0 for compatibility with the code from iCaRL.
    The initial TF code seems to occasionally shuffle labels in each batch,
    which is why labels are also serialized to guarantee that the same label order is kept.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append(os.path.join(sys.path[0], "./datasets/"))
sys.path.append(os.path.join(sys.path[0], "./lwf/"))

from dataset_config import dataset_config
from lwf_utils import init_network, load_files_list


def parse_args(argv):
    """
    Parsing input arguments.
    """
    # Arguments
    parser = argparse.ArgumentParser(description='Extracting validation and test logits for each incremental state.')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')

    parser.add_argument('--models-dir', type=str,
                        help='Directory containing serialized models')
    parser.add_argument('--logits-outdir', type=str,
                        help='Output directory to save task logits')
    parser.add_argument('--models-base-name', type=str,
                        help='Base serialized Tensorflow task model name')

    # dataset args
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--batch-size', default=128, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=4, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')

    args, extra_args = parser.parse_known_args(argv)

    # Updating paths
    args.models_dir = os.path.expanduser(args.models_dir)
    args.logits_outdir = os.path.expanduser(args.logits_outdir)
    args.models_base_path = args.models_dir

    args.datasets = args.datasets[0]

    # Printing input arguments
    print("Input arguments:")
    print(args)

    return args, extra_args


def extract_logits(state, data_files, models_path, dset_mean, device, batch_size):
    """
    Extracting logits from all files provided, using the model learned in state t.

    Args:
        state:
            t:  Incremental state index
            S:  Total number of states
            P:  Number of new classes in each state (should be constant)
        data_files:  Test files for the current state
        models_path: Base directory containing the trained models
        dset_mean:   Dataset mean
        device:      Device to use
        batch_size:  Batch size to use
    """
    t, S, P = state

    # Initializing network
    inits, scores, label_batch, loss_class, file_string_batch, op_feature_map = init_network(
        dset_mean, data_files, device, t, batch_size, S, P, models_path)

    # Total number of batches
    nb_batches = int(np.ceil(len(data_files)/float(batch_size)))

    # Number of extra entries in the last batch
    extra_entries = (nb_batches*batch_size) - len(data_files)

    # Storing all logits
    all_logits = None
    all_labels = None

    with tf.Session(config=config) as sess:
        # Launch the prefetch system
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(inits)

        for i in range(nb_batches):
            sc, l, loss, files_tmp, feat_map_tmp = sess.run(
                [scores, label_batch, loss_class, file_string_batch, op_feature_map])

            # Removing repeated entries, and truncating scores
            if i == nb_batches-1 and extra_entries > 0:
                labels = l[:-extra_entries]
                sc_trunc = sc[:-extra_entries, :(t + 1) * P]
            else:
                labels = l
                sc_trunc = sc[:, :(t + 1) * P]

            if all_logits is None:
                all_labels = labels
                all_logits = sc_trunc
            else:
                all_labels = np.concatenate((all_labels, labels))
                all_logits = np.concatenate((all_logits, sc_trunc))

        coord.request_stop()
        coord.join(threads)

    # Reset the graph to compute the numbers ater the next increment
    tf.reset_default_graph()

    # Removing repeated outputs
    return all_logits, all_labels


def main(argv=None):
    """
    Extracting and saving output logits.
    """
    # Filtering EXIF warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    # Parsing input arguments
    args, _ = parse_args(argv)

    # Selecting device
    device = "/gpu:%d" % args.gpu

    # Incremental variables
    S, P = args.num_tasks, dataset_config[args.datasets]['num_classes'] / args.num_tasks

    # Loading dataset statistics
    dset_mean, _ = dataset_config[args.datasets]['normalize']
    dset_mean = [e * 255 for e in dset_mean]

    # Loading test/val files
    test_files = load_files_list(dataset_config[args.datasets]['path'], "test", S, P)
    val_files  = load_files_list(dataset_config[args.datasets]['path'], "val", S, P)

    for t in range(S):
        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Test logits: extracting and saving
        test_iter = []
        for u in range(t + 1):
            test_iter.extend(test_files[u])

        all_logits, all_labels = extract_logits((t, S, P), test_iter,
                                                args.models_base_path, dset_mean,
                                                device, args.batch_size)
        print 'Test logits shape: ', all_logits.shape
        print 'Test labels shape: ', all_labels.shape

        for u, (logits, labels) in enumerate(zip(np.split(all_logits, t+1), np.split(all_labels, t+1))):
            tst_logits_file = "%slogits_tst_%d_%d" % (args.logits_outdir, t, u)
            np.save(tst_logits_file, logits)

            tst_labels_file = "%slabels_tst_%d_%d" % (args.logits_outdir, t, u)
            np.save(tst_labels_file, labels)

        # Validation logits: extracting and saving
        val_iter = []
        for u in range(t + 1):
            val_iter.extend(val_files[u])

        all_logits, all_labels = extract_logits((t, S, P), val_iter,
                                                args.models_base_path, dset_mean,
                                                device, args.batch_size)
        print 'Val logits shape: ', all_logits.shape
        print 'Val labels shape: ', all_labels.shape

        for u, (logits, labels) in enumerate(zip(np.split(all_logits, t+1), np.split(all_labels, t+1))):
            val_logits_file = "%slogits_val_%d_%d" % (args.logits_outdir, t, u)
            np.save(val_logits_file, logits)

            val_labels_file = "%slabels_val_%d_%d" % (args.logits_outdir, t, u)
            np.save(val_labels_file, labels)

    print("All output logits succesfully saved!")


if __name__ == '__main__':
    main()
