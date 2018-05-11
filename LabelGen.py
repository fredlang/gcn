import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

import matplotlib.pyplot as plt
import networkx as nx

from  utils import load_corrInformation
from  utils import load_modelxml


#################################
# 各个对象作用
# X 模型之间相似度矩阵
# modepath_list 模型路径
# partpair_list 对应到的组件对

X, modepath_list, partpair_list = load_corrInformation('D:\\langxf\\0416\\chair_corr.json')
allpartname_list = []
allpartpath_list = []
allpartlabel_list = []
allgroup_list = []
alladjmatrix_list = []
################################



# 根据list_partpair初始化标签信息
for model in modepath_list:
    name_list, path_list, label_list, group_list, adj_matrix = load_modelxml(model)
    allpartname_list.append(name_list)
    allpartpath_list.append(path_list)
    allpartlabel_list.append(label_list)
    allgroup_list.append(group_list)
    alladjmatrix_list.append(adj_matrix)


# 先拿最近一个模型进行配对，然后深度优先进行标签传播
count = X.shape[0]  # 行数
for k in range(count):
    # 根据距离矩阵创建距离字典 对象在modepath_list中的index:相应距离
    distance_dic = {}
    i = 0
    for dis in X[k]:
        distance_dic[i] = dis
        i = i + 1

    # 排序：找到分数最接近的模型
    sorted(distance_dic.items(),key=lambda item:item[1])

    #


# 查询是否所有组件均已赋标签


def setlable(X,adjList):
    X.shape()







# #############################################################################
# Cluster 1
# <editor-fold desc="Compute DBSCAN">
db = DBSCAN(eps=0.055, min_samples=2, metric='precomputed')
db.fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# print(labels)
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print(n_clusters_)
# </editor-fold>

# #############################################################################


#  ############################################################################
# Cluster 2
# <editor-fold desc="Compute SpectralClustering">
# db2 = SpectralClustering(n_clusters=4, affinity='precomputed', n_neighbors=4)
# db2.fit(X)
# labels2 = db2.labels_
# print(labels2)
# </editor-fold>
#  ############################################################################


unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k in unique_labels:
    print(k,":", end=" ")

    class_member_mask = (labels == k)
    for i in range(len(class_member_mask)):
        if class_member_mask[i]:
            print(i, end=" ")

    print("")

