# coding=utf-8
"""
Learning without Forgetting implementation based on Rebuffi's iCaRL.
"""
from __future__ import division

import argparse
import numpy as np
import os
import pickle
import sys

import tensorflow as tf

import utils_data
import utils_icarl
import utils_resnet

# Configuring Tensorflow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Local imports
sys.path.append(os.path.join(sys.path[0], "../datasets/"))
from dataset_config import dataset_config


def parse_args(argv):
    """
    Parsing input arguments.
    """
    # Arguments
    parser = argparse.ArgumentParser(description='Training the LwF baseline.')

    # miscellaneous args
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--models-dir', default='', type=str,
                        help='Output directory to save the basic step model.')

    # dataset args
    parser.add_argument('--datasets', type=str,
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--num-tasks', type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--batch-size', default=128, type=int, required=False,
                        help='Real batch size, before gradient accumulation (default=%(default)s)')

    # training args
    parser.add_argument('--lr', default=1.0, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=5., type=float, required=False,
                        help='Learning rate decay factor (default=%(default)s)')
    parser.add_argument('--lr-strat', type=int, nargs="+",
                        help='Learning rate scheduler strategy (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.00001, type=float, required=False,
                        help='Weight decay (default=%(default)s)')
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')

    args, extra_args = parser.parse_known_args(argv)
    args.datasets = args.datasets[0]

    # Total number of classes
    args.num_classes_total = dataset_config[args.datasets]['num_classes']

    # Number of classes per task
    args.num_classes = int(args.num_classes_total/args.num_tasks)

    # Defining training/validation sets lists
    lists_path = dataset_config[args.datasets]['path']
    args.train_list = os.path.join(lists_path, 'train_no_val.lst')
    args.val_list   = os.path.join(lists_path, 'test.lst')

    # Loading dataset statistics
    images_mean, _ = dataset_config[args.datasets]['normalize']
    args.images_mean = [e * 255 for e in images_mean]

    # Printing input arguments
    print("Input arguments:")
    print(args)

    return args, extra_args


def main(argv=None):
    """
    Main training routine.
    """
    # Parsing input arguments
    args, _ = parse_args(argv)

    device = '/gpu:0'
    if not os.path.exists(args.models_dir):
        os.makedirs(args.models_dir)
    np.random.seed(args.seed)

    # Initializing variables
    class_means = np.zeros((512, args.num_tasks * args.num_classes, 2, args.num_tasks))
    loss_batch = []
    files_protoset = []
    for _ in range(args.num_tasks * args.num_classes):
        files_protoset.append([])

    # Random mixing
    print("Preparing data...")
    order = np.arange(args.num_tasks * args.num_classes)

    # Preparing the files per group of classes
    files_train, files_valid = utils_data.prepare_files(args.train_list, args.val_list,
                                                        args.num_tasks,  args.num_classes)

    with open(args.models_dir + str(args.num_classes) + 'settings_resnet.pickle', 'wb') as fp:
        pickle.dump(order, fp)
        pickle.dump(files_valid, fp)
        pickle.dump(files_train, fp)

    # Main algorithm
    for itera in range(args.num_tasks):
        # Files to load : training samples + protoset
        print('Batch of classes number {0} arrives ...'.format(itera + 1))

        # Adding the stored exemplars to the training set
        files_from_cl = files_train[itera]

        # Importing the data reader
        image_train, label_train = utils_data.read_data(files_from_cl=files_from_cl)
        image_batch, label_batch_0 = tf.train.batch([image_train, label_train],
                                                    batch_size=args.batch_size, num_threads=8)
        label_batch = tf.one_hot(label_batch_0, args.num_tasks * args.num_classes)

        # Defining the objective for the neural network
        if itera == 0:
            # No distillation
            variables_graph, variables_graph2, scores, scores_stored = utils_icarl.prepare_networks(args.images_mean,
                                                                                                    device, image_batch,
                                                                                                    args.num_classes,
                                                                                                    args.num_tasks)

            # Define the objective for the neural network: 1 vs all cross_entropy
            with tf.device(device):
                scores = tf.concat(scores, 0)
                l2_reg = args.weight_decay * tf.reduce_sum(
                    tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='ResNet18'))
                loss_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch, logits=scores))
                loss = loss_class + l2_reg
                learning_rate = tf.placeholder(tf.float32, shape=[])
                opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
                train_step = opt.minimize(loss, var_list=variables_graph)

        if itera > 0:
            # Distillation
            variables_graph, variables_graph2, scores, scores_stored = utils_icarl.prepare_networks(args.images_mean,
                                                                                                    device, image_batch,
                                                                                                    args.num_classes,
                                                                                                    args.num_tasks)

            # Copying the network to use its predictions as ground truth labels
            op_assign = [(variables_graph2[i]).assign(variables_graph[i]) for i in range(len(variables_graph))]

            # Define the objective for the neural network : 1 vs all cross_entropy + distillation
            with tf.device(device):
                scores = tf.concat(scores, 0)
                scores_stored = tf.concat(scores_stored, 0)
                old_cl = (order[range(itera * args.num_classes)]).astype(np.int32)
                new_cl = (order[range(itera * args.num_classes, args.num_tasks * args.num_classes)]).astype(np.int32)
                label_old_classes = tf.sigmoid(tf.stack([scores_stored[:, i] for i in old_cl], axis=1))
                label_new_classes = tf.stack([label_batch[:, i] for i in new_cl], axis=1)
                pred_old_classes = tf.stack([scores[:, i] for i in old_cl], axis=1)
                pred_new_classes = tf.stack([scores[:, i] for i in new_cl], axis=1)
                l2_reg = args.weight_decay * tf.reduce_sum(
                    tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='ResNet18'))
                loss_class = tf.reduce_mean(tf.concat(
                    [tf.nn.sigmoid_cross_entropy_with_logits(labels=label_old_classes, logits=pred_old_classes),
                     tf.nn.sigmoid_cross_entropy_with_logits(labels=label_new_classes, logits=pred_new_classes)], 1))
                loss = loss_class + l2_reg
                learning_rate = tf.placeholder(tf.float32, shape=[])
                opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
                train_step = opt.minimize(loss, var_list=variables_graph)

        # Run the learning phase
        with tf.Session(config=config) as sess:
            # Launch the data reader
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(tf.global_variables_initializer())
            lr = args.lr

            # Run the loading of the weights for the learning network and the copy network
            if itera > 0:
                void0 = sess.run([(variables_graph[i]).assign(save_weights[i]) for i in range(len(variables_graph))])
                void1 = sess.run(op_assign)

            for epoch in range(args.nepochs):
                print("Batch of classes {} out of {} batches".format(itera + 1, args.num_tasks))
                print('Epoch %i' % epoch)

                for i in range(int(np.ceil(len(files_from_cl) / args.batch_size))):
                    loss_class_val, _, sc, lab = sess.run([loss_class, train_step, scores, label_batch_0],
                                                          feed_dict={learning_rate: lr})
                    loss_batch.append(loss_class_val)

                    # Plot the training error every 10 batches
                    if len(loss_batch) == 10:
                        # print(np.mean(loss_batch))
                        loss_batch = []

                    # Plot the training top 1 accuracy every 80 batches
                    if (i + 1) % 80 == 0:
                        stat = []
                        stat += ([ll in best for ll, best in zip(lab, np.argsort(sc, axis=1)[:, -1:])])
                        stat = np.average(stat)
                        print('Training accuracy %f' % stat)

                # Decrease the learning by 5 every 10 epoch after 20 epochs at the first learning rate
                if epoch in args.lr_strat:
                    lr /= args.lr_factor

            coord.request_stop()
            coord.join(threads)

            # copy weights to store network
            save_weights = sess.run([variables_graph[i] for i in range(len(variables_graph))])
            utils_resnet.save_model(args.models_dir + 'model-iteration' + str(args.num_classes) + '-%i.pickle' % itera,
                                    scope='ResNet18', sess=sess)

        # Reset the graph
        tf.reset_default_graph()

        # Pickle class means and protoset
        with open(args.models_dir + str(args.num_classes) + 'class_means.pickle', 'wb') as fp:
            pickle.dump(class_means, fp)
        with open(args.models_dir + str(args.num_classes) + 'files_protoset.pickle', 'wb') as fp:
            pickle.dump(files_protoset, fp)


if __name__ == "__main__":
    main()
