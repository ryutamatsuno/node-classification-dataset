import os
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import torch
import codecs


RAWDIR = './data_raw'

from Data import Data

def is_sparse(x: torch.Tensor) -> bool:
    """
    :param x:
    :return: True if x is sparse tensor else False
    """
    try:
        x._indices()
    except RuntimeError:
        return False
    return True

class SparseTensor(torch.Tensor):
    """
    NeverUse
    """
    def __init__(self):
        #super().__init__()
        raise NotImplementedError

def adj_list_from_dict(graph):
    G = nx.from_dict_of_lists(graph)
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    # converting 1 undirected edges -> 2 directed edges
    indices = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    return indices

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index



def get_largest_component(G):
    if nx.number_connected_components(G) == 1:
        return G
    Gs = sorted(nx.connected_components(G), key=len, reverse=True)
    H = G.subgraph(Gs[0])
    return H

def save_giant(data:Data, name):

    labels = data.labels
    features = data.features
    edge_list = data.raw_edge_list
    # train_mask = data.train_mask
    # valid_mask = data.valid_mask
    # tests_mask = data.tests_mask

    # print(labels.shape)
    # print(features.shape)
    # print(edge_list.shape)

    # print(train_mask.shape)
    # print(valid_mask.shape)
    # print(tests_mask.shape)

    edge_list = edge_list.numpy()
    es = []
    for i in range(edge_list.shape[1]):
        es.append((edge_list[0,i],edge_list[1,i]))

    G = nx.Graph()
    G.add_edges_from(es)
    G.remove_edges_from(nx.selfloop_edges(G))

    # print("N:",len(nx.nodes(G)))
    # print("M:",len(nx.edges(G)))
    # print("M:",edge_list.shape[1])
    #
    # assert len(nx.edges(G)) *2 == edge_list.shape[1]

    #print("number of componetns:", nx.number_connected_components(G))


    if nx.number_connected_components(G) > 1:


        G = get_largest_component(G)


        # # sampling
        # sample_size = 1000
        # chosen_nodes = np.random.permutation(np.array(G.nodes))[:sample_size]
        # G = nx.subgraph(G, chosen_nodes)
        # G = get_largest_component(G)
        # print("sampled size:",len(nx.nodes(G)))

        # fix
        sampled_nodes = np.array(sorted(nx.nodes(G)), dtype=np.int)
        num_nodes = len(sampled_nodes)

        labels = labels.numpy()[sampled_nodes]
        features = features.numpy()[sampled_nodes]

        # use old mask
        # train_mask = train_mask[sampled_nodes]
        # valid_mask = valid_mask[sampled_nodes]
        # tests_mask = tests_mask[sampled_nodes]

        # # gen new mask
        # idx = np.arange(num_nodes)
        # idx = np.random.permutation(idx)
        # train_num = int(0.3 * num_nodes)
        # valid_num = int(0.3 * num_nodes)
        # tests_num = num_nodes - train_num - valid_num
        #
        # train_mask = np.zeros(num_nodes, dtype=np.int)
        # train_mask[idx[:train_num]] = 1
        # train_mask = train_mask.astype(bool)
        #
        # valid_mask = np.zeros(num_nodes, dtype=np.int)
        # valid_mask[idx[train_num:train_num + valid_num]] = 1
        # valid_mask = valid_mask.astype(bool)
        #
        # tests_mask = np.zeros(num_nodes, dtype=np.int)
        # tests_mask[idx[train_num + valid_num:]] = 1
        # tests_mask = tests_mask.astype(bool)

        # mapping
        remap = {}
        for i in range(num_nodes):
            remap[sampled_nodes[i]] = i
        G = nx.relabel_nodes(G,mapping=remap)

        # oubling edge_list
        edge_list = np.array(nx.edges(G), dtype=np.int).transpose() # 2,M
        directed = np.stack((edge_list[1], edge_list[0]), axis=0)
        edge_list = np.concatenate((edge_list, directed), axis=1)

        # print("N:", len(G.nodes()))
        # print("M:", len(G.edges()))

        data = Data(torch.tensor(edge_list, dtype=torch.long), torch.tensor(features, dtype=torch.float), torch.tensor(labels, dtype=torch.long), data.split_setting)

    data.print_statisitcs()
    data.save(name)
    return data

def convert_raw_data(dataset_str: str, seed=None):
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        data = load_planetoid_data(dataset_str)
    elif dataset_str in ['chameleon','cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        data = load_geom(dataset_str, seed)
    else:
        data = load_npz_data(dataset_str, seed)
    save_giant(data, dataset_str)
    exit()


##########################################################
#
#  Loading raw data
#
##########################################################

def load_geom(dataset_str, seed):

    dir = RAWDIR + '/from_geom/' + dataset_str

    # edge
    with open(dir +'/out1_graph_edges.txt', mode='r') as f:
        lines = f.readlines()[1:]
    es = []

    for l in lines:
        l = l.replace('\n','')
        splited = l.split('\t')
        splited = [int(x) for x in splited]
        es.append(splited)
        #es.append([splited[1], splited[0]])

    # make G to remove multi edge and selfloops
    G = nx.Graph()
    G.add_edges_from(es)
    G.remove_edges_from(nx.selfloop_edges(G))
    es = G.edges()

    edge_list = np.array(es, dtype=np.int).transpose(1,0)
    directed = np.stack((edge_list[1], edge_list[0]), axis=0)
    edge_list = np.concatenate((edge_list, directed), axis=1)

    edge_list = torch.LongTensor(edge_list)


    # data
    with open(dir +'/out1_node_feature_label.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]

    labels =[]
    features = []
    for l in lines:
        l = l.replace('\n','')
        splited = l.split('\t')
        labels.append(int(splited[2]))

        features.append([int(x) for x in splited[1].split(',')])

    if dataset_str == 'film':
        feat_dim = max([max(x) for x in features])
        N = len(features)
        npf = np.zeros((N, feat_dim), dtype=np.float)
        for x in features:
            npf[np.array(x, dtype=np.int)] = 1
        features = npf

    labels = torch.LongTensor(np.array(labels,dtype=np.int))
    features = torch.tensor(np.array(features,dtype=np.int), dtype=torch.float)
    #train_mask, val_mask, test_mask = split_data(labels, 10, 50, seed)

    data = Data(edge_list, features, labels, [2,50])

    return data


def load_npz_data(dataset_str, seed):
    with np.load(RAWDIR + '/npz/' + dataset_str + '.npz', allow_pickle=True) as loader:
        loader = dict(loader)
        adj_mat = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                shape=loader['adj_shape']).tocoo()
        if dataset_str[:2] == 'ms':
            edge_list = torch.cat((torch.tensor(adj_mat.row).type(torch.int64).view(1, -1),
                                   torch.tensor(adj_mat.col).type(torch.int64).view(1, -1)), dim=0)
        else:
            edge_list1 = torch.cat((torch.tensor(adj_mat.row).type(torch.int64).view(1, -1),
                                    torch.tensor(adj_mat.col).type(torch.int64).view(1, -1)), dim=0)
            edge_list2 = torch.cat((torch.tensor(adj_mat.col).type(torch.int64).view(1, -1),
                                    torch.tensor(adj_mat.row).type(torch.int64).view(1, -1)), dim=0)
            edge_list = torch.cat([edge_list1, edge_list2], dim=1)


        if 'attr_data' in loader:
            feature_mat = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape']).todense()
        elif 'attr_matrix' in loader:
            feature_mat = loader['attr_matrix']
        else:
            feature_mat = None
        features = torch.tensor(feature_mat)

        if 'labels_data' in loader:
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape']).todense()
        elif 'labels' in loader:
            labels = loader['labels']
        else:
            labels = None
        labels = torch.tensor(labels).long()

        #train_mask, val_mask, test_mask = split_data(labels, 20, 500, seed)

    data = Data(edge_list, features, labels, [20,500])

    return data

def load_planetoid_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        with open(RAWDIR + "/planetoid/ind.{}.{}".format(dataset_str, name), 'rb') as f:
            if sys.version_info > (3, 0):
                out = pkl.load(f, encoding='latin1')
            else:
                out = objects.append(pkl.load(f))

            if name == 'graph':
                objects.append(out)
            else:
                out = out.todense() if hasattr(out, 'todense') else out
                objects.append(torch.Tensor(out))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx = parse_index_file(RAWDIR + "/planetoid/ind.{}.test.index".format(dataset_str))
    train_idx = torch.arange(y.size(0), dtype=torch.long)
    val_idx = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    sorted_test_idx = np.sort(test_idx)

    if dataset_str == 'citeseer':
        len_test_idx = max(test_idx) - min(test_idx) + 1
        tx_ext = torch.zeros(len_test_idx, tx.size(1))
        tx_ext[sorted_test_idx - min(test_idx), :] = tx
        ty_ext = torch.zeros(len_test_idx, ty.size(1))
        ty_ext[sorted_test_idx - min(test_idx), :] = ty

        tx, ty = tx_ext, ty_ext

    features = torch.cat([allx, tx], dim=0)
    features[test_idx] = features[sorted_test_idx]

    labels = torch.cat([ally, ty], dim=0).max(dim=1)[1]
    labels[test_idx] = labels[sorted_test_idx]

    edge_list = adj_list_from_dict(graph)
    # train_mask = index_to_mask(train_idx, labels.shape[0])
    # val_mask = index_to_mask(val_idx, labels.shape[0])
    # test_mask = index_to_mask(test_idx, labels.shape[0])

    data = Data(edge_list, features, labels, [20,500])
    return data
