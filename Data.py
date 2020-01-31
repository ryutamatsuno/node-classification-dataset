import os
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import torch
import codecs

TOPDIR = 'data/'


def add_self_loops(edge_list, size):
    i = torch.arange(size, dtype=torch.int64).view(1, -1)
    self_loops = torch.cat((i, i), dim=0)
    edge_list = torch.cat((edge_list, self_loops), dim=1)
    return edge_list


def get_degree(edge_list):
    row, col = edge_list
    deg = torch.bincount(row)
    return deg


def normalize_adj(edge_list):
    deg = get_degree(edge_list)
    row, col = edge_list
    deg_inv_sqrt = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    weight = torch.ones(edge_list.size(1))
    v = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj


def edgelist2normalized_adj(edge_list, size) -> (torch.Tensor, torch.sparse.FloatTensor):
    edge_list = add_self_loops(edge_list, size)
    norm_adj = normalize_adj(edge_list)
    return edge_list, norm_adj


def edgelist2adj(edge_list) -> torch.sparse.FloatTensor:
    v = torch.ones(edge_list.size(1))
    adj = torch.sparse.FloatTensor(edge_list, v)
    return adj


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def split_data(labels: torch.Tensor, n_train_per_class: int, n_val: int, seed) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    np.random.seed(seed)
    n_class = int(torch.max(labels)) + 1
    train_idx = np.array([], dtype=np.int64)
    remains = np.array([], dtype=np.int64)
    for c in range(n_class):
        candidate = torch.nonzero(labels == c).T.numpy()[0]
        np.random.shuffle(candidate)
        train_idx = np.concatenate([train_idx, candidate[:n_train_per_class]])
        remains = np.concatenate([remains, candidate[n_train_per_class:]])
    np.random.shuffle(remains)
    val_idx = remains[:n_val]
    test_idx = remains[n_val:]

    assert test_idx.shape[0] > val_idx.shape[0], ('No Test data', val_idx.shape[0], test_idx.shape[0])
    train_mask = index_to_mask(train_idx, labels.size(0))
    val_mask = index_to_mask(val_idx, labels.size(0))
    test_mask = index_to_mask(test_idx, labels.size(0))
    return train_mask, val_mask, test_mask

def preprocess_features(features:torch.Tensor):
    rowsum = features.sum(dim=1, keepdim=True)
    rowsum[rowsum == 0] = 1
    features = features / rowsum
    return features


class Data(object):

    @staticmethod
    def load(dataname):

        top_dir = TOPDIR + dataname

        labels = np.loadtxt(top_dir + '/labels.csv', dtype=np.int, delimiter=",")
        features = np.loadtxt(top_dir + '/features.csv', dtype=np.float, delimiter=",")
        edge_list = np.loadtxt(top_dir + '/edge_list.edg', dtype=np.int, delimiter=",").transpose()
        split_setting = np.loadtxt(top_dir + '/split.txt', dtype=np.int, delimiter=",").tolist()

        labels = torch.tensor(labels, dtype=torch.long)
        features = torch.tensor(features, dtype=torch.float)
        edge_list = torch.tensor(edge_list, dtype=torch.long)

        # train_mask = torch.tensor(train_mask, dtype=torch.bool)
        # valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        # tests_mask = torch.tensor(tests_mask, dtype=torch.bool)
        # print(train_mask.shape)

        # use random mask
        # train_mask, valid_mask, tests_mask = split_data(labels, 5, 50, seed)

        data = Data(edge_list, features, labels, split_setting)
        return data

    def save(self, dataname):

        top_dir = TOPDIR + dataname
        if not os.path.exists(top_dir):
            os.mkdir(top_dir)
        np.savetxt(top_dir + '/labels.csv', self.labels.numpy(), '%d', delimiter=",")
        np.savetxt(top_dir + '/features.csv', self.features.numpy(), '%f', delimiter=",")
        np.savetxt(top_dir + '/edge_list.edg', self.raw_edge_list.numpy().transpose(), '%d', delimiter=",")
        np.savetxt(top_dir + '/split.txt', self.split_setting, '%d', delimiter=",")

    def __init__(self, edge_list: torch.Tensor, features: torch.Tensor, labels: torch.Tensor, split_setting: list):

        self.raw_edge_list = edge_list
        self.raw_adj = edgelist2adj(edge_list)

        # normalized edge_list, normalized adj
        self.norm_edge_list, self.norm_adj = edgelist2normalized_adj(edge_list, features.size(0))

        self.features = features
        self.labels = labels
        self.split_setting = split_setting

        self.num_features = features.size(1)
        self.num_classes = int(torch.max(labels)) + 1

        self.train_mask = None
        self.valid_mask = None
        self.tests_mask = None
        self.update_mask()

        self.num_train = torch.sum(self.train_mask.int(), dim=0).item()
        self.num_valid = torch.sum(self.valid_mask.int(), dim=0).item()
        self.num_tests = torch.sum(self.tests_mask.int(), dim=0).item()

    # def to(self, device):
    #     self.adj = self.adj.to(device)
    #     self.edge_list = self.edge_list.to(device)
    #     self.features = self.features.to(device)
    #     self.labels = self.labels.to(device)
    #     self.train_mask = self.train_mask.to(device)
    #     self.valid_mask = self.valid_mask.to(device)
    #     self.tests_mask = self.tests_mask.to(device)

    @property
    def A(self):
        return self.raw_adj

    def update_mask(self, seed=None):
        self.train_mask, self.valid_mask, self.tests_mask = split_data(self.labels, self.split_setting[0], self.split_setting[1], seed)

        self.num_train = torch.sum(self.train_mask.int(), dim=0).item()
        self.num_valid = torch.sum(self.valid_mask.int(), dim=0).item()
        self.num_tests = torch.sum(self.tests_mask.int(), dim=0).item()

    def print_statisitcs(self):

        edge_list = self.raw_edge_list.numpy().transpose()

        G = nx.Graph()
        G.add_edges_from(edge_list)

        print(" - statistics - ")
        print("N:", len(nx.nodes(G)))
        print("M:", len(nx.edges(G)))
        assert len(nx.edges(G)) *2 == edge_list.shape[0]
        print("M:", edge_list.shape[0], ' OK!')
        print("number of components:", nx.number_connected_components(G))
        print('features:', self.num_features)
        print('classes:', self.num_classes)

        # counts for each label

        def label_counts(data: Data, mask=None):
            counts = np.zeros(data.num_classes, dtype=np.int)
            labels = data.labels.numpy()
            for i in range(labels.shape[0]):
                if mask is not None:
                    if not mask[i]:
                        continue
                counts[int(labels[i])] += 1
            return counts

        print('data label balance:', label_counts(self))

        print('num_train:', self.num_train, label_counts(self, self.train_mask))
        print('num_valid:', self.num_valid, label_counts(self, self.valid_mask))
        print('num_tests:', self.num_tests, label_counts(self, self.tests_mask))

        print(" - ")
