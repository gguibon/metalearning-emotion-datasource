import os
import copy
import itertools
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from termcolor import colored

import code.train.utils as utils
from code.classifier.mlp import MLP
from code.dataset.parallel_sampler import ParallelSampler


def test(test_data, model, args, verbose=True):
    '''
        Finetune model based on bag of sampled target tasks.
    '''

    sampled_tasks = ParallelSampler(
            test_data, args, args.test_episodes).get_epoch()

    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=args.test_episodes, ncols=80,
                leave=False, desc=colored('Finetuning on test', 'yellow'))

    acc,f1, mcc = [], [], []

    for task in sampled_tasks:
        res_acc, res_f1, res_mcc = finetune_one(task, model, args)
        acc.append(res_acc)
        f1.append(res_f1)
        mcc.append(res_mcc)

    acc, f1, mcc = np.array(acc), np.array(f1), np.array(mcc)

    if verbose:
        print("{}, {:s} {:>7.4f} ({:s} {:>7.4f}), {:s} {:>7.4f} ({:s} {:>7.4f}), {:s} {:>7.4f} ({:s} {:>7.4f}),".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                colored("acc mean", "blue"),
                np.mean(acc),
                colored("std", "blue"),
                np.std(acc),
                colored("f1 mean", "blue"),
                np.mean(f1),
                colored("std", "blue"),
                np.std(f1),
                colored("mcc mean", "blue"),
                np.mean(mcc),
                colored("std", "blue"),
                np.std(mcc),
                ), flush=True)

        # latex table
        print("{:s} & {:s} & {:>7.4f} \\tiny $\\pm {:>7.4f}$ & {:>7.4f} \\tiny $\\pm {:>7.4f}$ & {:>7.4f} \\tiny $\\pm {:>7.4f}$ ".format(
                args.embedding,
                args.classifier,
                np.mean(acc),
                np.std(acc),
                np.mean(f1),
                np.std(f1),
                np.mean(mcc),
                np.std(mcc),
                ), flush=True)

    return np.mean(acc), np.std(acc), np.mean(f1), np.std(f1), np.mean(mcc), np.std(mcc)


def finetune_one(task, model, args):
    '''
        Finetune model on single target task.
    '''
    # copy model so we don't overwrite saved weights
    ebd = copy.deepcopy(model['ebd'])

    # this is a newly initialized layer
    top = MLP.get_top_layer(args, args.way).cuda(args.cuda)

    # this contains the new top layer
    clf = MLP(ebd.ebd_dim, args, top).cuda(args.cuda)

    # this copies the old mlp weights to the current one
    old_clf_dict = model['clf'].state_dict()
    cur_clf_dict = clf.state_dict()
    for key in cur_clf_dict.keys():
        if key[:4] == 'mlp.':
            cur_clf_dict[key] = old_clf_dict[key]

    support, query = task

    # new optimizer and scheduler per task
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad,
                itertools.chain(ebd.parameters(),
                                clf.parameters())), lr=args.lr)

    _, YS = torch.unique(support['label'], sorted=True, return_inverse=True)

    for _ in range(args.finetune_maxepochs):
        ebd.train()  # change this .train() to .eval() if we don't change ebd
        clf.train()
        opt.zero_grad()

        # Embedding the document
        XS = ebd(support)

        # Apply the classifier)
        acc, loss, _, _ = clf(XS, YS)

        loss.backward()

        ebd_norm = utils.get_norm(ebd)
        clf_norm = utils.get_norm(clf)

        if (ebd_norm ** 2 + clf_norm ** 2) ** 0.5 < 1e-3:
            break

        opt.step()

    # evaluate on query
    ebd.eval()
    clf.eval()

    XQ = ebd(query)
    _, YQ = torch.unique(query['label'], sorted=True, return_inverse=True)

    # Apply the classifier
    acc, _, f1, mcc = clf(XQ, YQ)

    return acc, f1, mcc
