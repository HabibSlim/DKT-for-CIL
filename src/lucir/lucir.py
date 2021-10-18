"""
Learning a Unified Classifier Incrementally via Rebalancing (Hou et al.).
"""

import argparse
import copy
import math
import modified_resnet
import modified_linear
import numpy as np
import os
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim import lr_scheduler
from torchvision import transforms

from utils_dataset import split_images_labels, save_protosets, merge_images_labels
from utils_incremental.compute_features import compute_features
from utils_incremental.compute_accuracy import compute_accuracy
from utils_incremental.train_eval_LF import train_eval_LF
from utils_loader import ImagesFolder

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
    parser.add_argument('--adapt-lambda', default=True, type=bool, required=False,
                        help='Adjust lambda after each task (default=%(default)s)')
    parser.add_argument('--lambd', type=float, required=False,
                        help='Lambda base value (default=%(default)s)')
    parser.add_argument('--less-forget', default=True, type=bool, required=False,
                        help='Use the less-forget constraint (default=%(default)s)')
    parser.add_argument('--lr', default=1.0, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=5., type=float, required=False,
                        help='Learning rate decay factor (default=%(default)s)')
    parser.add_argument('--lr-strat', type=int, nargs="+",
                        help='Learning rate scheduler strategy (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.00001, type=float, required=False,
                        help='Weight decay (default=%(default)s)')
    parser.add_argument('--momentum', type=float, required=False,
                        help='SGD momentum (default=%(default)s)')
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
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
    args.test_list  = os.path.join(lists_path, 'test.lst')

    # Loading dataset statistics
    args.images_stats = dataset_config[args.datasets]['normalize']

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

    # Filtering EXIF warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    # Defining output models prefixes
    ckp_prefix = '{}_s{}_k{}'.format(args.datasets, args.num_tasks, 0)
    np.random.seed(args.seed)

    # Defining device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Instantiating dataloaders
    dataset_mean, dataset_std = args.images_stats
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

    trainset = ImagesFolder(
        args.train_list,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    testset = ImagesFolder(
        args.test_list,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize, ]))

    evalset = ImagesFolder(
        args.test_list,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize, ]))
    ################################

    # Initialization
    X_train_total, Y_train_total = split_images_labels(trainset.imgs)
    X_valid_total, Y_valid_total = split_images_labels(testset.imgs)

    top1_cnn_cumul_acc = []
    top5_cnn_cumul_acc = []
    top1_icarl_cumul_acc = []
    top5_icarl_cumul_acc = []
    top1_ncm_cumul_acc = []
    top5_ncm_cumul_acc = []

    # Defining the class order
    order = np.arange(args.num_classes_total)
    order_list = list(order)

    # Initialization of the variables for this run
    X_valid_cumuls = []
    X_protoset_cumuls = []
    X_train_cumuls = []
    Y_valid_cumuls = []
    Y_protoset_cumuls = []
    Y_train_cumuls = []

    # The following contains all the training samples of the different classes
    # because we want to compare our method with the theoretical case where all the training samples are stored
    prototypes = [[] for _ in range(args.num_classes_total)]
    for orde in range(args.num_classes_total):
        prototypes[orde] = X_train_total[np.where(Y_train_total == order[orde])]

    prototypes = np.array(prototypes)
    max_class_len = max(len(e) for e in prototypes)

    alpha_dr_herding = np.zeros((int(args.num_classes_total / args.num_classes),
                                 max_class_len, args.num_classes), np.float32)

    for b in range(0, args.num_tasks):
        if b == 0:
            ############################################################
            last_iter = 0
            ############################################################
            # initializing the model
            tg_model = modified_resnet.resnet18(num_classes=args.num_classes)
            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            print("in_features:", in_features, "out_features:", out_features)
            ref_model = None
        elif b == 1:
            ############################################################
            last_iter = b
            ############################################################
            # increment classes
            ref_model = copy.deepcopy(tg_model)
            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            print("in_features:", in_features, "out_features:", out_features)
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features, args.num_classes)
            new_fc.fc1.weight.data = tg_model.fc.weight.data
            new_fc.sigma.data = tg_model.fc.sigma.data
            tg_model.fc = new_fc
            lamda_mult = out_features * 1.0 / args.num_classes
        else:
            ############################################################
            last_iter = b
            ############################################################
            ref_model = copy.deepcopy(tg_model)
            in_features = tg_model.fc.in_features
            out_features1 = tg_model.fc.fc1.out_features
            out_features2 = tg_model.fc.fc2.out_features
            print("in_features:", in_features,
                  "out_features1:", out_features1,
                  "out_features2:", out_features2)
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features1 + out_features2, args.num_classes)
            new_fc.fc1.weight.data[:out_features1] = tg_model.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = tg_model.fc.fc2.weight.data
            new_fc.sigma.data = tg_model.fc.sigma.data
            tg_model.fc = new_fc
            lamda_mult = (out_features1 + out_features2) * 1.0 / args.num_classes

        if b > 0 and args.less_forget and args.adapt_lambda:
            cur_lamda = args.lambd * math.sqrt(lamda_mult)
        else:
            cur_lamda = args.lambd
        if b > 0 and args.less_forget:
            print("###############################")
            print("Lamda for less forget is set to ", cur_lamda)
            print("###############################")

        # Prepare the training data for the current batch of classes
        indices_train_10 = np.array([i in order[range(last_iter * args.num_classes, (b + 1) * args.num_classes)]
                                     for i in Y_train_total])
        indices_test_10 = np.array([i in order[range(last_iter * args.num_classes, (b + 1) * args.num_classes)]
                                    for i in Y_valid_total])

        X_train = X_train_total[indices_train_10]
        X_valid = X_valid_total[indices_test_10]
        X_valid_cumuls.append(X_valid)
        X_train_cumuls.append(X_train)
        X_valid_cumul = np.concatenate(X_valid_cumuls)

        Y_train = Y_train_total[indices_train_10]
        Y_valid = Y_valid_total[indices_test_10]
        Y_valid_cumuls.append(Y_valid)
        Y_train_cumuls.append(Y_train)
        Y_valid_cumul = np.concatenate(Y_valid_cumuls)

        if b != 0:
            X_protoset = np.concatenate(X_protoset_cumuls)
            Y_protoset = np.concatenate(Y_protoset_cumuls)
            X_train = np.concatenate((X_train, X_protoset), axis=0)
            Y_train = np.concatenate((Y_train, Y_protoset))

        # Launch the training loop
        print('Batch of classes number {0} arrives ...'.format(b + 1))
        map_Y_train = np.array([order_list.index(i) for i in Y_train])
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])

        # Imprint weights
        if b > 0:
            print("Imprint weights")
            #########################################
            # compute the average norm of old embdding
            old_embedding_norm = tg_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
            average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)
            #########################################
            tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
            num_features = tg_model.fc.in_features
            novel_embedding = torch.zeros((args.num_classes, num_features))

            for cls_idx in range(b * args.num_classes, (b + 1) * args.num_classes):
                cls_indices = np.array([i == cls_idx for i in map_Y_train])
                assert (len(np.where(cls_indices == 1)[0]) <= max_class_len)
                current_eval_set = merge_images_labels(X_train[cls_indices], np.zeros(len(X_train[cls_indices])))
                evalset.imgs = evalset.samples = current_eval_set
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.batch_size,
                                                         shuffle=False, num_workers=args.num_workers)
                num_samples = len(X_train[cls_indices])
                cls_features = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
                cls_embedding = torch.mean(norm_features, dim=0)
                novel_embedding[cls_idx - b * args.num_classes] = F.normalize(cls_embedding, p=2,
                                                                              dim=0) * average_old_embedding_norm
            tg_model.to(device)
            tg_model.fc.fc2.weight.data = novel_embedding.to(device)

        ############################################################
        current_train_imgs = merge_images_labels(X_train, map_Y_train)
        trainset.imgs = trainset.samples = current_train_imgs
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)

        print('Training-set size = ' + str(len(trainset)))

        current_test_imgs = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
        testset.imgs = testset.samples = current_test_imgs
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=args.num_workers)
        print('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
        print('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid_cumul), max(map_Y_valid_cumul)))
        ##############################################################
        ckp_name = os.path.join(args.models_dir, ckp_prefix + '_model_{}.pth'.format(b))
        print('ckp_name', ckp_name)

        ###############################
        if b > 0 and args.less_forget:
            # fix the embedding of old classes
            ignored_params = list(map(id, tg_model.fc.fc1.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params,
                                 tg_model.parameters())
            tg_params = [{'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
                         {'params': tg_model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
        else:
            tg_params = tg_model.parameters()
        ###############################
        tg_model = tg_model.to(device)
        if b > 0:
            ref_model = ref_model.to(device)
        tg_optimizer = optim.SGD(tg_params, lr=args.lr, momentum=args.momentum,
                                 weight_decay=args.weight_decay)
        tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=args.lr_strat, gamma=args.lr_factor)
        ###############################
        print("train_eval_LF")
        tg_model = train_eval_LF(args.nepochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler,
                                 trainloader, testloader,
                                 b, 0, cur_lamda)

        if not os.path.isdir(args.models_dir):
            os.makedirs(args.models_dir)
        torch.save(tg_model, ckp_name)

        # Exemplars
        nb_protos_cl = 0
        print('nb_protos_cl = ' + str(nb_protos_cl))

        tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
        num_features = tg_model.fc.in_features

        # Prepare the protoset
        X_protoset_cumuls = []
        Y_protoset_cumuls = []

        # Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
        print('Computing mean-of_exemplars and theoretical mean...')
        class_means = np.zeros((num_features, args.num_classes_total, 2))
        for b2 in range(b + 1):
            for iter_dico in range(args.num_classes):
                current_cl = order[range(b2 * args.num_classes, (b2 + 1) * args.num_classes)]
                current_eval_set = merge_images_labels(prototypes[b2 * args.num_classes + iter_dico],
                                                       np.zeros(len(prototypes[b2 * args.num_classes + iter_dico])))
                evalset.imgs = evalset.samples = current_eval_set
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.batch_size,
                                                         shuffle=False, num_workers=args.num_workers, pin_memory=True)
                num_samples = len(prototypes[b2 * args.num_classes + iter_dico])
                mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)

                D = mapped_prototypes.T
                D = D / np.linalg.norm(D, axis=0)
                D2 = D

                # iCaRL
                alph = alpha_dr_herding[b2, :, iter_dico]
                assert ((alph[num_samples:] == 0).all())
                alph = alph[:num_samples]
                alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.
                X_protoset_cumuls.append(prototypes[b2 * args.num_classes + iter_dico][np.where(alph == 1)[0]])
                Y_protoset_cumuls.append(
                    order[b2 * args.num_classes + iter_dico] * np.ones(len(np.where(alph == 1)[0])))
                alph = alph / np.sum(alph)
                class_means[:, current_cl[iter_dico], 0] = (np.dot(D, alph) + np.dot(D2, alph)) / 2
                class_means[:, current_cl[iter_dico], 0] /= np.linalg.norm(class_means[:, current_cl[iter_dico], 0])

                # Normal NCM
                alph = np.ones(num_samples) / num_samples
                class_means[:, current_cl[iter_dico], 1] = (np.dot(D, alph) + np.dot(D2, alph)) / 2
                class_means[:, current_cl[iter_dico], 1] /= np.linalg.norm(class_means[:, current_cl[iter_dico], 1])

        class_means_name = os.path.join(args.models_dir, ckp_prefix + '_class_means_{}.pth'.format(b))

        torch.save(class_means, class_means_name)

        current_means = class_means[:, order[range(0, (b + 1) * args.num_classes)]]
        ##############################################################
        # Calculate validation error of model on the cumul of classes:
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
        current_eval_set = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
        evalset.imgs = evalset.samples = current_eval_set
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
        cumul_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader)

        ###############################
        print('Saving protoset...')
        map_Y_protoset_cumuls = np.array([order_list.index(i) for i in np.concatenate(Y_protoset_cumuls)])
        current_eval_set = merge_images_labels(np.concatenate(X_protoset_cumuls), map_Y_protoset_cumuls)
        save_protosets(current_eval_set, ckp_prefix, b, args.models_dir)
        ##############################################################

        top1_cnn_cumul_acc.append(float(str(cumul_acc[0])[:6]))
        top5_cnn_cumul_acc.append(float(str(cumul_acc[1])[:6]))
        top1_icarl_cumul_acc.append(float(str(cumul_acc[2])[:6]))
        top5_icarl_cumul_acc.append(float(str(cumul_acc[3])[:6]))
        top1_ncm_cumul_acc.append(float(str(cumul_acc[4])[:6]))
        top5_ncm_cumul_acc.append(float(str(cumul_acc[5])[:6]))

        print("###########################################################")
        print('TOP-1 detailed Results')
        print('LUCIR - CNN = ' + str(top1_cnn_cumul_acc))
        print('LUCIR - NCM = ' + str(top1_ncm_cumul_acc))
        print('iCaRL       = ' + str(top1_icarl_cumul_acc))
        print("###########################################################")
        print('TOP-5 detailed Results')
        print('LUCIR - CNN = ' + str(top5_cnn_cumul_acc))
        print('LUCIR - NCM = ' + str(top5_ncm_cumul_acc))
        print('iCaRL       = ' + str(top5_icarl_cumul_acc))
        print("###########################################################")
        print('mean inc accuracy')
        mean_top1_cnn_cumul_acc = np.mean(np.array(top1_cnn_cumul_acc)[1:])
        mean_top5_cnn_cumul_acc = np.mean(np.array(top5_cnn_cumul_acc)[1:])
        mean_top1_icarl_cumul_acc = np.mean(np.array(top1_icarl_cumul_acc)[1:])
        mean_top5_icarl_cumul_acc = np.mean(np.array(top5_icarl_cumul_acc)[1:])
        mean_top1_ncm_cumul_acc = np.mean(np.array(top1_ncm_cumul_acc)[1:])
        mean_top5_ncm_cumul_acc = np.mean(np.array(top5_ncm_cumul_acc)[1:])
        print(
            'LUCIR - CNN | acc@1 = {:.2f} \t acc@5 = {:.2f} '.format(mean_top1_cnn_cumul_acc, mean_top5_cnn_cumul_acc))
        print(
            'LUCIR - NCM | acc@1 = {:.2f} \t acc@5 = {:.2f} '.format(mean_top1_ncm_cumul_acc, mean_top5_ncm_cumul_acc))
        print('iCaRL       | acc@1 = {:.2f} \t acc@5 = {:.2f} '.format(mean_top1_icarl_cumul_acc,
                                                                       mean_top5_icarl_cumul_acc))


if __name__ == "__main__":
    main()
