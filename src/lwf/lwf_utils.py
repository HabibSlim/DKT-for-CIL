"""
iCaRL utility functions.
"""
import os
import pickle
from collections import defaultdict

import numpy as np
import tensorflow as tf

import tf_resnet


def read_test_data(files_from_cl):
    """
    Reading test data.
    """
    image_list = np.array([e.split()[0] for e in files_from_cl])
    labels_list = np.array([e.split()[1] for e in files_from_cl])
    files_list = files_from_cl

    assert (len(image_list) == len(labels_list))
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    files = tf.convert_to_tensor(files_list, dtype=tf.string)
    labels = tf.convert_to_tensor(labels_list, dtype=tf.int32)
    input_queue = tf.train.slice_input_producer([images, labels, files], shuffle=False, capacity=2000)
    image_file_content = tf.read_file(input_queue[0])
    label = input_queue[1]
    file_string = input_queue[2]
    image = tf.image.resize_images(tf.image.decode_jpeg(image_file_content, channels=3), [224, 224])

    return image, label, file_string


def init_network(images_mean, files_from_cl, device, itera, batch_size, nb_groups, nb_cl, save_path):
    """
    Reading the input data and initializing a ResNet-18 network.
    """
    image_train, label_train, file_string = read_test_data(files_from_cl=files_from_cl)
    image_batch, label_batch, file_string_batch = tf.train.batch([image_train, label_train, file_string],
                                                                 batch_size=batch_size, num_threads=8)
    label_batch_one_hot = tf.one_hot(label_batch, nb_groups * nb_cl)

    # Network and loss function
    mean_img = tf.constant(images_mean, dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    with tf.variable_scope('ResNet18'):
        with tf.device(device):
            scores = tf_resnet.ResNet18(image_batch - mean_img, phase='test', num_outputs=nb_cl * nb_groups)
            graph = tf.get_default_graph()
            op_feature_map = graph.get_operation_by_name('ResNet18/pool_last/avg').outputs[0]

    loss_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch_one_hot, logits=scores))

    # Initialization
    net_path = os.path.join(save_path, 'model-iteration' + str(nb_cl) + '-%i.pickle' % itera)
    net_file = open(net_path, 'rb')
    params = dict(pickle.load(net_file))
    inits = tf_resnet.get_weight_initializer(params)

    return inits, scores, label_batch, loss_class, file_string_batch, op_feature_map


def load_files_list(list_path, split, S, P):
    """
    Loading validation files and splitting them by tasks.
    """
    val_path = os.path.join(list_path, "%s.lst" % split)

    with open(val_path, 'r') as f:
        val_files = f.read().splitlines()

    # Grouping entries by class
    val_dset = defaultdict(list)
    for file_entry in val_files:
        file_, y = file_entry.split(' ')
        y = int(y)
        val_dset[y] += [file_]

    # Separating by incremental state
    val_batched = [[] for _ in range(S)]
    for t in range(S):
        val_batched += []
        for y in range(t * P, (t + 1) * P):
            val_batched[t] += [s_ + " " + str(y) for s_ in val_dset[y]]

    return val_batched


def print_summary(metrics_list, labels_list):
    """
    Print metrics summary across incremental states.
    Python 2.7 compatible version.

    Args:
        metrics_list: Network prediction
        labels_list:  List of labels for the given metrics
    """
    for metric, name in zip(metrics_list, labels_list):
        print('*' * 108)
        print(name)
        mean_inc_acc = []
        for i in range(metric.shape[0]):
            print '\t',
            for j in range(metric.shape[1]):
                print '{:5.2f}% '.format(100 * metric[i, j]),
            if np.trace(metric) == 0.0:
                if i > 0:
                    avg = 100 * metric[i, :i].mean()
                    mean_inc_acc += [avg]
                    print '\tAvg.:{:5.2f}% '.format(avg),
            else:
                avg = 100 * metric[i, :i + 1].mean()
                mean_inc_acc += [avg]
                print '\tAvg.:{:5.2f}% '.format(avg),
            print ''
        print ''

        # Computing AIA across all incremental states (thus excluding the first non-incremental state)
        print('\tMean Incremental Acc.: {:5.2f}%'.format(np.mean(mean_inc_acc[1:])))
    print('*' * 108)
