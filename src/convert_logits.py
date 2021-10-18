"""
Converting saved logits from numpy to torch format.
"""
import argparse
import os
import torch
import numpy as np
import warnings


def parse_args(argv):
    """
    Parsing input arguments.
    """
    # Arguments
    parser = argparse.ArgumentParser(description='Converting saved logits from numpy to torch format.')

    # miscellaneous args
    parser.add_argument('--logits-dir', type=str,
                        help='Output directory to save task logits')
    parser.add_argument('--num-tasks', default=4, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')

    args, extra_args = parser.parse_known_args(argv)

    # Updating paths
    args.logits_dir = os.path.expanduser(args.logits_dir)

    # Printing input arguments
    print("Input arguments:")
    print(args)

    return args, extra_args


def to_torch(file_name):
    """
    Convert an input numpy matrix to torch format.

    Args:
        file_name:  File name of the .npy file to be converted
    """
    full_file = "%s.npy" % file_name
    np_mat = np.load(full_file)

    full_file = "%s.ckpt" % file_name
    torch.save(torch.from_numpy(np_mat), full_file)


def main(argv=None):
    """
    Extracting output logits and saving them in the "results_path" folder.
    """
    # Filtering EXIF warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    # Parsing input arguments
    args, _ = parse_args(argv)

    # Iterating over
    for t in range(args.num_tasks):
        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Extracting and serializing validation and test logits
        for u in range(t + 1):
            # Validation logits/labels
            to_torch("%slogits_val_%d_%d" % (args.logits_dir, t, u))
            to_torch("%slabels_val_%d_%d" % (args.logits_dir, t, u))

            # Test logits/labels
            to_torch("%slogits_tst_%d_%d" % (args.logits_dir, t, u))
            to_torch("%slabels_tst_%d_%d" % (args.logits_dir, t, u))

        print('-' * 108)

    print("All output logits succesfully converted!")
    print("Deleting numpy logits...", end=" ")

    # Deleting numpy logits
    for item in os.listdir(args.logits_dir):
        if item.endswith(".npy"):
            os.remove(os.path.join(args.logits_dir, item))

    print("Done.")


if __name__ == '__main__':
    main()
