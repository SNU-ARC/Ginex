import os
import json
import numpy as np
import torch
from lib.utils import *


def get_mmap_dataset(path='../dataset/ogbn-papers100M-ginex', split_idx_path=None):
    indptr_path = os.path.join(path, 'indptr.dat')
    indices_path = os.path.join(path, 'indices.dat')
    features_path = os.path.join(path, 'features.dat')
    labels_path = os.path.join(path, 'labels.dat')
    conf_path = os.path.join(path, 'conf.json')

    conf = json.load(open(conf_path, 'r'))

    indptr = np.fromfile(indptr_path, dtype=conf['indptr_dtype']).reshape(tuple(conf['indptr_shape']))
    indices = np.memmap(indices_path, mode='r', shape=tuple(conf['indices_shape']), dtype=conf['indices_dtype'])
    features_shape = conf['features_shape']
    features = np.memmap(features_path, mode='r', shape=tuple(features_shape), dtype=conf['features_dtype'])
    labels = np.fromfile(labels_path, dtype=conf['labels_dtype'], count=conf['num_nodes']).reshape(tuple([conf['labels_shape'][0]]))

    indptr = torch.from_numpy(indptr)
    indices = torch.from_numpy(indices)
    features = torch.from_numpy(features)
    labels = torch.from_numpy(labels)

    num_nodes = conf['num_nodes']
    num_features = conf['features_shape'][1]
    num_classes = conf['num_classes']

    split_idx = torch.load(split_idx_path)
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']

    return indptr, indices, features, labels, num_features, num_classes, num_nodes, train_idx, val_idx, test_idx


class GinexDataset():
    def __init__(self, path='../dataset/ogbn-papers100M-ginex', split_idx_path=None, score_path=None):
        self.indptr_path = os.path.join(path, 'indptr.dat')
        self.indices_path = os.path.join(path, 'indices.dat')
        self.features_path = os.path.join(path, 'features.dat')
        self.labels_path = os.path.join(path, 'labels.dat')
        conf_path = os.path.join(path, 'conf.json')
        self.conf = json.load(open(conf_path, 'r'))

        split_idx = torch.load(split_idx_path)
        self.train_idx = split_idx['train']
        self.val_idx = split_idx['valid']
        self.test_idx = split_idx['test']

        self.score_path = score_path

        self.num_nodes = self.conf['num_nodes']
        self.num_features = self.conf['features_shape'][1]
        self.num_classes = self.conf['num_classes']


    # Return indptr & indices
    def get_adj_mat(self):
        indptr = np.fromfile(self.indptr_path, dtype=self.conf['indptr_dtype']).reshape(tuple(self.conf['indptr_shape']))
        indices = np.memmap(self.indices_path, mode='r', shape=tuple(self.conf['indices_shape']), dtype=self.conf['indices_dtype'])
        indptr = torch.from_numpy(indptr)
        indices = torch.from_numpy(indices)
        return indptr, indices


    def get_col(self):
        indices = np.memmap(self.indices_path, mode='r', shape=tuple(self.conf['indices_shape']), dtype=self.conf['indices_dtype'])
        indices = torch.from_numpy(indices)
        return indices


    def get_rowptr_mt(self):
        indptr_size = self.conf['indptr_shape'][0]
        indptr = mt_load(self.indptr_path, indptr_size)
        return indptr


    def get_labels(self):
        labels = torch.from_numpy(np.fromfile(self.labels_path, dtype=self.conf['labels_dtype'], count=self.num_nodes).reshape(tuple([self.conf['labels_shape'][0]])))
        return labels


    def get_labels_mt(self):
        labels_size = self.conf['labels_shape'][0]
        labels = mt_load_float(self.labels_path, labels_size)
        return labels


    def make_new_shuffled_train_idx(self):
        self.shuffled_train_idx = self.train_idx[torch.randperm(self.train_idx.numel())]


    def get_mmapped_features(self):
        features_shape = self.conf['features_shape']
        features_shape[1] = self.num_features
        features = np.memmap(self.features_path, mode='r', shape=tuple(features_shape), dtype=self.conf['features_dtype'])
        features = torch.from_numpy(features)
        return features


    def get_score(self):
        return torch.load(self.score_path)
