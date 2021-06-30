import os
import collections
import json
from collections import defaultdict, Counter
from termcolor import colored

import numpy as np
import torch
from torchtext.vocab import Vocab, Vectors

from code.embedding.avg import AVG
from code.embedding.cxtebd import CXTEBD
from code.embedding.wordebd import WORDEBD
import code.dataset.stats as stats
from code.dataset.utils import tprint

from transformers import BertTokenizer

from sklearn.model_selection import train_test_split


def _get_goemotion_meta_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'admiration': 0,
        'amusement': 1,
        'anger': 2, #ekman
        'annoyance': 3,
        'approval': 4,
        'caring': 5,
        'confusion': 6,
        'curiosity': 7,
        'desire': 8,
        'disappointment': 9,
        'disapproval': 10,
        'disgust': 11, #ekman
        'embarrassment': 12,
        'excitement': 13,
        'fear': 14, #ekman
        'gratitude': 15,
        'grief': 16,
        'joy': 17, #ekman
        'love': 18,
        'nervousness': 19,
        'optimism': 20,
        'pride': 21,
        'realization': 22,
        'relief': 23,
        'remorse': 24,
        'sadness': 25, #ekman
        'surprise': 26, #ekman
        'neutral': 27
    }

    train_classes = [ 0, 4, 3, 1, 18, 6, 22, 13, 24, 19, 21] #11
    val_classes = [ 15, 7, 10, 20, 9, 5, 8, 12, 23, 16 ] #10
    # Ekman 6 : Joy, Sadness, Anger, Fear, Disgust, Surprise
    test_classes = [ 17, 25, 2, 14, 11, 26 ] #6

    return train_classes, val_classes, test_classes


def _get_goemotion_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'admiration': 0,
        'amusement': 1,
        'anger': 2, #ekman
        'annoyance': 3,
        'approval': 4,
        'caring': 5,
        'confusion': 6,
        'curiosity': 7,
        'desire': 8,
        'disappointment': 9,
        'disapproval': 10,
        'disgust': 11, #ekman
        'embarrassment': 12,
        'excitement': 13,
        'fear': 14, #ekman
        'gratitude': 15,
        'grief': 16,
        'joy': 17, #ekman
        'love': 18,
        'nervousness': 19,
        'optimism': 20,
        'pride': 21,
        'realization': 22,
        'relief': 23,
        'remorse': 24,
        'sadness': 25, #ekman
        'surprise': 26, #ekman
        'neutral': 27
    }
    all = list(label_dict.values()) 
    all.remove(27) # 27 tags (remove the neutral one)
    # train_classes = all
    # val_classes =all
    train_classes = [ 17, 25, 2, 14, 11, 26 ]
    val_classes = [ 17, 25, 2, 14, 11, 26 ]
    test_classes = [ 17, 25, 2, 14, 11, 26 ] #6

    return train_classes, val_classes, test_classes


def _get_dailydialog_utterances_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = { 
        'no emotion': 0, 
        'anger': 1, 
        'disgust': 2,
        'fear': 3,
        'happiness': 4,
        'sadness': 5,
        'surprise': 6
    }

    train_classes = [ 1, 2, 3, 4, 5, 6]
    val_classes = [ 1, 2, 3, 4, 5, 6]
    test_classes = [ 1, 2, 3, 4, 5, 6]

    return train_classes, val_classes, test_classes


def _get_dailydialog_utterances_ekman(args, strategy='emotweet28'):
    '''
        dedicated to final test using model trained on emotTweet and then finetuned on Emotweet 5 ekman equivalent classes beforehand.
        Args:
            strategy (list): ['emotweet28', 'goemotions']
        @return list of classes associated with each split
    '''
    label_dict = { 
        'no emotion': 0, # not targeted
        'anger': 1, 
        'disgust': 2, # no equivalent
        'fear': 3,
        'happiness': 4,
        'sadness': 5,
        'surprise': 6
    }

    train_classes = [ 1, 2, 3, 4, 5, 6 ] #+ [ i for i in range(5, args.n_train_class) ]
    val_classes = [ 1, 2, 3, 4, 5, 6 ] #+ [ i for i in range(5, args.n_val_class) ]
    if strategy in ['emotweet28']: test_classes = [ 1, 3, 4, 5, 6 ]
    else: test_classes = [ 1, 2, 3, 4, 5, 6 ]

    return train_classes, val_classes, test_classes


def _get_dailydialog_utterances_meta_classes(args):
    '''
        dedicated to the meta version
        @return list of classes associated with each split
    '''
    label_dict = { 
        'no emotion': 0, 
        'anger': 1, 
        'disgust': 2,
        'fear': 3,
        'happiness': 4,
        'sadness': 5,
        'surprise': 6
    }

    train_classes = [ 4, 6 ]
    val_classes = [ 1, 3 ]
    test_classes = [ 2, 5]

    return train_classes, val_classes, test_classes


def _load_json(path):
    '''
        load data file
        @param path: str, path to the data file
        @return data: list of examples
    '''
    label = {}
    text_len = []
    with open(path, 'r', errors='ignore') as f:
        data = []
        for line in f:
            row = json.loads(line)

            # count the number of examples per label
            if int(row['label']) not in label:
                label[int(row['label'])] = 1
            else:
                label[int(row['label'])] += 1

            item = {
                'label': int(row['label']),
                'text': row['text'][:500]  # truncate the text to 500 tokens
            }

            if 'split' in row: item['split'] = row['split']

            text_len.append(len(row['text']))

            keys = ['head', 'tail', 'ebd_id']
            for k in keys:
                if k in row:
                    item[k] = row[k]

            data.append(item)

        tprint('Class balance:')

        print(label)

        tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))
        tprint('Max len: {}'.format(max(text_len)))

        return data


def _read_words(data):
    '''
        Count the occurrences of all words
        @param data: list of examples
        @return words: list of words (with duplicates)
    '''
    words = []
    for example in data:
        words += example['text']
    return words


def _meta_split(all_data, train_classes, val_classes, test_classes):
    '''
        Split the dataset according to the specified train_classes, val_classes
        and test_classes

        @param all_data: list of examples (dictionaries)
        @param train_classes: list of int
        @param val_classes: list of int
        @param test_classes: list of int

        @return train_data: list of examples
        @return val_data: list of examples
        @return test_data: list of examples
    '''
    train_data, val_data, test_data = [], [], []

    for example in all_data:
        if example['label'] in train_classes:
            train_data.append(example)
        if example['label'] in val_classes:
            val_data.append(example)
        if example['label'] in test_classes:
            test_data.append(example)

    return train_data, val_data, test_data


def _meta_split_by_ratio(all_data, train_classes, val_classes, test_classes):
    '''
        Split the dataset by only taking into account random ratio (0.75 for train, 0.125 val, 0.125 test)

        @param all_data: list of examples (dictionaries)
        @param train_classes: list of int
        @param val_classes: list of int
        @param test_classes: list of int

        @return train_data: list of examples
        @return val_data: list of examples
        @return test_data: list of examples
    '''
    train_data, val_data, test_data = [], [], []
    train_data, valtest_data = train_test_split(all_data)
    val_data, test_data = train_test_split(valtest_data, test_size=0.5)
    del valtest_data
    return train_data, val_data, test_data


def _meta_split_by_field(all_data, train_classes, val_classes, test_classes):
    '''
        Split the dataset according to the specified train_classes, val_classes
        and test_classes
        Consider a 'split' field for the different train test val sets

        @param all_data: list of examples (dictionaries)
        @param train_classes: list of int
        @param val_classes: list of int
        @param test_classes: list of int

        @return train_data: list of examples
        @return val_data: list of examples
        @return test_data: list of examples
    '''
    train_data, val_data, test_data = [], [], []

    for example in all_data:
        if example['split'] == 'train' and example['label'] in train_classes:
            train_data.append(example)
        if example['split'] == 'val' and example['label'] in val_classes:
            val_data.append(example)
        if example['split'] == 'test' and example['label'] in test_classes:
            test_data.append(example)

    return train_data, val_data, test_data


def _split_by_field(all_data, train_classes, val_classes, test_classes):
    '''
        Split the dataset according to the specified train_classes, val_classes
        and test_classes
        Consider a 'split' field for the different train test val sets

        @param all_data: list of examples (dictionaries)
        @param train_classes: list of int
        @param val_classes: list of int
        @param test_classes: list of int

        @return train_data: list of examples
        @return val_data: list of examples
        @return test_data: list of examples
    '''
    train_data, val_data, test_data = [], [], []

    for example in all_data:
        if example['split'] == 'train' :
            train_data.append(example)
        if example['split'] == 'val' :
            val_data.append(example)
        if example['split'] == 'test' :
            test_data.append(example)

    return train_data, val_data, test_data


def _del_by_idx(array_list, idx, axis):
    '''
        Delete the specified index for each array in the array_lists

        @params: array_list: list of np arrays
        @params: idx: list of int
        @params: axis: int

        @return: res: tuple of pruned np arrays
    '''
    if type(array_list) is not list:
        array_list = [array_list]

    # modified to perform operations in place
    for i, array in enumerate(array_list):
        array_list[i] = np.delete(array, idx, axis)

    if len(array_list) == 1:
        return array_list[0]
    else:
        return array_list


def _data_to_nparray(data, vocab, args):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''
    doc_label = np.array([x['label'] for x in data], dtype=np.int64)

    raw = np.array([e['text'] for e in data], dtype=object)

    if args.bert:
        tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased', do_lower_case=True)

        # convert to wpe
        vocab_size = 0  # record the maximum token id for computing idf
        for e in data:
            e['bert_id'] = tokenizer.encode(" ".join(e['text']),
                                            add_special_tokens=True)
                                            # max_length=80)
            vocab_size = max(max(e['bert_id'])+1, vocab_size)

        text_len = np.array([len(e['bert_id']) for e in data])
        max_text_len = max(text_len)

        text = np.zeros([len(data), max_text_len], dtype=np.int64)

        del_idx = []
        # convert each token to its corresponding id
        for i in range(len(data)):
            text[i, :len(data[i]['bert_id'])] = data[i]['bert_id']

            # filter out document with only special tokens
            # unk (100), cls (101), sep (102), pad (0)
            if np.max(text[i]) < 103:
                del_idx.append(i)

        text_len = text_len

    else:
        # compute the max text length
        text_len = np.array([len(e['text']) for e in data])
        max_text_len = max(text_len)

        # initialize the big numpy array by <pad>
        text = vocab.stoi['<pad>'] * np.ones([len(data), max_text_len],
                                         dtype=np.int64)

        del_idx = []
        # convert each token to its corresponding id
        for i in range(len(data)):
            text[i, :len(data[i]['text'])] = [
                    vocab.stoi[x] if x in vocab.stoi else vocab.stoi['<unk>']
                    for x in data[i]['text']]

            # filter out document with only unk and pad
            if np.max(text[i]) < 2:
                del_idx.append(i)

        vocab_size = vocab.vectors.size()[0]

    text_len, text, doc_label, raw = _del_by_idx(
            [text_len, text, doc_label, raw], del_idx, 0)

    new_data = {
        'text': text,
        'text_len': text_len,
        'label': doc_label,
        'raw': raw,
        'vocab_size': vocab_size,
    }

    if 'pos' in args.auxiliary:
        # use positional information in fewrel
        head = np.vstack([e['head'] for e in data])
        tail = np.vstack([e['tail'] for e in data])

        new_data['head'], new_data['tail'] = _del_by_idx(
            [head, tail], del_idx, 0)

    return new_data


def _split_dataset(data, finetune_split):
    """
        split the data into train and val (maintain the balance between classes)
        @return data_train, data_val
    """

    # separate train and val data
    # used for fine tune
    data_train, data_val = defaultdict(list), defaultdict(list)

    # sort each matrix by ascending label order for each searching
    idx = np.argsort(data['label'], kind="stable")

    non_idx_keys = ['vocab_size', 'classes2id', 'is_train']
    for k, v in data.items():
        if k not in non_idx_keys:
            data[k] = v[idx]

    # loop through classes in ascending order
    classes, counts = np.unique(data['label'], return_counts=True)
    start = 0
    for label, n in zip(classes, counts):
        mid = start + int(finetune_split * n)  # split between train/val
        end = start + n  # split between this/next class

        for k, v in data.items():
            if k not in non_idx_keys:
                data_train[k].append(v[start:mid])
                data_val[k].append(v[mid:end])

        start = end  # advance to next class

    # convert back to np arrays
    for k, v in data.items():
        if k not in non_idx_keys:
            data_train[k] = np.concatenate(data_train[k], axis=0)
            data_val[k] = np.concatenate(data_val[k], axis=0)

    return data_train, data_val


def load_dataset(args):
    if args.dataset in ['dailydialog_u', 'dailydialog_u_test']:
        train_classes, val_classes, test_classes = _get_dailydialog_utterances_classes(args)
    elif args.dataset == 'dailydialog_u_meta':
        train_classes, val_classes, test_classes = _get_dailydialog_utterances_meta_classes(args)
    elif args.dataset == 'dailydialog_u_ekman':
        train_classes, val_classes, test_classes = _get_dailydialog_utterances_ekman(args, strategy=args.finetuned_dataset)
    elif args.dataset in ['goemotions', 'dailydialog_u_test_from_goemotions']: 
        train_classes, val_classes, test_classes = _get_goemotion_classes(args)
    elif args.dataset == 'goemotions_meta':
        train_classes, val_classes, test_classes = _get_goemotion_meta_classes(args)
    else:
        raise ValueError(
            'args.dataset should be one of'
            '[dailydialog_u, dailydialog_u_meta, dailydialog_u_ekman, emotweet28, goemotions_meta, goemotweet_meta]')

    assert(len(train_classes) == args.n_train_class)
    assert(len(val_classes) == args.n_val_class)
    assert(len(test_classes) == args.n_test_class)

    if args.mode == 'finetune':
        # in finetune, we combine train and val for training the base classifier
        train_classes = train_classes + val_classes
        args.n_train_class = args.n_train_class + args.n_val_class
        args.n_val_class = args.n_train_class
    elif args.mode == 'finetune2':
        # here we ony test data
        train_classes = test_classes
        val_classes = test_classes
        args.n_train_class, args.n_val_class = args.n_test_class, args.n_test_class
        

    tprint('Loading data')
    all_data = _load_json(args.data_path)

    if args.mode in ['test_from_other']:
        if args.dataset == 'dailydialog_u_ekman' and args.finetuned_dataset == 'emotweet28':
            print( colored('swap target label correspondance', 'yellow'), len(all_data) )
            dailydialog2emotweet_labels= { 1:3, 3:12, 4:14, 5:25, 6:27}
            print('test_classes', test_classes)
            print('all_data', set([ d['label'] for d in all_data]) )
            all_data = [ { 'text':d['text'], 'label':dailydialog2emotweet_labels[d['label']] } for d in all_data if d['label'] in test_classes ]
            print('all_data', set([ d['label'] for d in all_data]) )
            print( colored('res', 'yellow'), len(all_data))
        elif args.dataset == 'dailydialog_u_ekman' and args.finetuned_dataset == 'goemotions_meta':
            print( colored(args.dataset, 'yellow'), colored(args.finetuned_dataset, 'yellow') )
            # 'anger': 1,'disgust': 2,'fear': 3, 'happiness': 4,'sadness': 5,'surprise': 6
            dailydialog2goemotions_labels= { 1:2, 2:11, 3:14, 4:17, 5:25, 6:26}
            all_data = [ { 'text':d['text'], 'label':dailydialog2goemotions_labels[d['label']] } for d in all_data if d['label'] in test_classes ]
        elif args.dataset == 'dailydialog_u_ekman' and args.finetuned_dataset == 'goemotweet_meta':
            print( colored(args.dataset, 'yellow'), colored(args.finetuned_dataset, 'yellow') )
            # 'anger': 1,'disgust': 2,'fear': 3, 'happiness': 4,'sadness': 5,'surprise': 6
            # 'disgust': 14, 'fear': 20, 'sadness': 42, 'anger': 2, 'surprise': 44, 'sadness': 42, 'happiness,joy': 23,
            dailydialog2goemotions_labels= { 1:2, 2:14, 3:20, 4:23, 5:42, 6:44}
            all_data = [ { 'text':d['text'], 'label':dailydialog2goemotions_labels[d['label']] } for d in all_data if d['label'] in test_classes ]
        elif args.dataset == 'dailydialog_u_test_from_goemotions':
            dailydialog2goemotions_labels= { 1:2, 2:11, 3:14, 4:17, 5:25, 6:26}
            all_data = [ { 'text':d['text'], 'label':dailydialog2goemotions_labels[d['label']] } for d in all_data if d['split'] == 'test' ]


    tprint('Loading word vectors')
    path = os.path.join(args.wv_path, args.word_vector)
    if not os.path.exists(path):
        # Download the word vector and save it locally:
        tprint('Downloading word vectors')
        import urllib.request
        urllib.request.urlretrieve(
            'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec',
            path)

    vectors = Vectors(args.word_vector, cache=args.wv_path)
    vocab = Vocab(collections.Counter(_read_words(all_data)), vectors=vectors,
                  specials=['<pad>', '<unk>'], min_freq=5)

    # print word embedding statistics
    wv_size = vocab.vectors.size()
    tprint('Total num. of words: {}, word vector dimension: {}'.format(
        wv_size[0],
        wv_size[1]))

    num_oov = wv_size[0] - torch.nonzero(
            torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
    tprint(('Num. of out-of-vocabulary words'
           '(they are initialized to zeros): {}').format( num_oov))

    # Split into meta-train, meta-val, meta-test data (or just splits)
    if args.dataset in ['sstweets','dailydialog_u', 'goemotions']:
        train_data, val_data, test_data = _meta_split_by_ratio(
            all_data, train_classes, val_classes, test_classes)
    elif args.dataset in ['dailydialog_u_ekman']:
        train_data, val_data, test_data = all_data, all_data, all_data
    elif args.dataset in ['dailydialog_u_test']:
        train_data, val_data, test_data = _meta_split_by_field(all_data, train_classes ,val_classes, test_classes)
        trainset = Counter([d['label'] for d in train_data])
        valset = Counter([d['label'] for d in val_data])
        testset = Counter([d['label'] for d in test_data])
        print(colored('check sets splits', 'yellow'), trainset, valset, testset)
    elif args.dataset in ['dailydialog_u_test_from_goemotions']:
        train_data, val_data, test_data = _split_by_field(all_data, train_classes ,val_classes, test_classes)
        trainset = Counter([d['label'] for d in train_data])
        valset = Counter([d['label'] for d in val_data])
        testset = Counter([d['label'] for d in test_data])
        print(colored('check sets splits (non meta)', 'yellow'), trainset, valset, testset)
    else:
        train_data, val_data, test_data = _meta_split( all_data, train_classes, val_classes, test_classes)
    tprint('#train {}, #val {}, #test {}'.format( len(train_data), len(val_data), len(test_data)))

    # Convert everything into np array for fast data loading
    train_data = _data_to_nparray(train_data, vocab, args)
    val_data = _data_to_nparray(val_data, vocab, args)
    test_data = _data_to_nparray(test_data, vocab, args)

    train_data['is_train'] = True
    # this tag is used for distinguishing train/val/test when creating source pool

    stats.precompute_stats(train_data, val_data, test_data, args)

    if args.meta_w_target:
        # augment meta model by the support features
        if args.bert:
            ebd = CXTEBD(args.pretrained_bert,
                         cache_dir=args.bert_cache_dir,
                         finetune_ebd=False,
                         return_seq=True)
        else:
            ebd = WORDEBD(vocab, finetune_ebd=False)

        train_data['avg_ebd'] = AVG(ebd, args)
        if args.cuda != -1:
            train_data['avg_ebd'] = train_data['avg_ebd'].cuda(args.cuda)

        val_data['avg_ebd'] = train_data['avg_ebd']
        test_data['avg_ebd'] = train_data['avg_ebd']

    # if finetune, train_classes = val_classes and we sample train and val data
    # from train_data
    if args.mode in ['finetune']:
        train_data, val_data = _split_dataset(train_data, args.finetune_split)
    elif args.mode in ['finetune2']:
        train_data, val_data = _split_dataset(test_data, args.finetune_split)

    return train_data, val_data, test_data, vocab
