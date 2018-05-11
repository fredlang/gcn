import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import json
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import matplotlib.pyplot as plt
import xml.dom.minidom
from xml.dom.minidom import parse

# 从json文件中读取度量矩阵、模型路径、组件配对
def load_corrInformation(corrpath_str):
    with open(corrpath_str, 'r') as load_f:
        load_dict = json.load(load_f)

    # 模型文件list_modepath
    list_modepath = []

    # 构造距离度量矩阵distance_matrix
    row_index = []
    col_index = []
    cost = []

    # 组件对list_partpair
    list_partpair = []

    for corr in load_dict:
        row_index.append(corr['i'])
        col_index.append(corr['j'])
        cost.append(corr['cost'])
        list_modepath.append(corr['source'])
        correspondence = corr['correspondence']
        part1 = []
        part2 = []
        for partpair in correspondence:
            part1.append(partpair[0])
            part2.append(partpair[1])
        parts = np.array([np.array(part1), np.array(part2)])
        list_partpair.append(parts)

    count  = max(max(col_index)+1,max(col_index)+1)
    distance_matrix = sp.csr_matrix((cost,(row_index,col_index)), shape=(count,count)).transpose(copy=True).todense()
    distance_matrix = distance_matrix + distance_matrix.transpose()

    return distance_matrix,list_modepath,list_partpair

# 从xml文件读取图结构、对称信息、feature
def load_modelxml(model_xml_str):
    DOMTree = xml.dom.minidom.parse(model_xml_str)
    collection = DOMTree.documentElement
    nodes = collection.getElementsByTagName("node")
    edges = collection.getElementsByTagName("edge")
    groups = collection.getElementsByTagName("group")

    name_list = []
    path_list = []
    label_list = []
    group_list = []

    for node in nodes:
        # 读取name和path
        name_list.append(node.getElementsByTagName('id')[0].childNodes[0].data)
        path_list.append(node.getElementsByTagName('mesh')[0].childNodes[0].data)
        label_list.append(-1)

    # 图结构邻接矩阵
    rownum_list = []
    colnum_list = []
    data = []
    for edge in edges:
        rownum_list.append(name_list.index(edge.getElementsByTagName('n')[0].childNodes[0].data))
        colnum_list.append(name_list.index(edge.getElementsByTagName('n')[1].childNodes[0].data))
        data.append(1)

    count = len(name_list)
    adj_matrix = sp.csr_matrix((data,(rownum_list,colnum_list)), shape=(count,count)).transpose(copy=True).todense()
    adj_matrix = adj_matrix + adj_matrix.transpose()

    for group in groups:
        groupteam = []
        for element in group.getElementsByTagName('n'):
            groupteam.append(element.childNodes[0].data)
        group_list.append(groupteam)

    return name_list,path_list,label_list,group_list,adj_matrix


def drawPlotAndLabel(X,labels,core_samples_mask):
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            print(f.name)
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

#    print("test_idx_range:",test_idx_range)


    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
#    print("features ALL:", features)
    features[test_idx_reorder, :] = features[test_idx_range, :]
#    print("features Test:", features)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#    print("adjacency_matrixt:", adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    '''
    print(type(adj))
    print(adj)
    print(type(features))
    print(features)
    '''
    print(type(y_train))
    print(y_train)
    '''
    print(y_val)
    print(y_test)
    print(train_mask)
    print(val_mask)
    print(test_mask)
    '''

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
