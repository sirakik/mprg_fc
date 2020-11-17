import numpy as np


def soccer_graph_edge_list():
    # roop edge
    roop_edge_list = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10],
                      [11, 11],
                      [12, 12], [13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20],
                      [21, 21],
                      [22, 22]]

    # ball edge
    ball_edge_list_left = [[22, 0], [22, 1], [22, 2], [22, 3], [22, 4], [22, 5], [22, 6], [22, 7], [22, 8], [22, 9],
                           [22, 10]]
    ball_edge_list_right = [[22, 11], [22, 12], [22, 13], [22, 14], [22, 15], [22, 16], [22, 17], [22, 18], [22, 19],
                            [22, 20], [22, 21]]
    ball_edge_list = np.concatenate((ball_edge_list_left, ball_edge_list_right), axis=0)

    # FW edge
    fw_edge_list_left = [[2, 1], [2, 10], [2, 7], [2, 8], [2, 12], [2, 13], [2, 21], [2, 18], [2, 19], [2, 14], [2, 15],
                         [2, 16], [2, 11],
                         [1, 7], [1, 8], [1, 9], [1, 12], [1, 13], [1, 18], [1, 19], [1, 20], [1, 15], [1, 16], [1, 17],
                         [1, 11]]
    fw_edge_list_right = [[13, 12], [13, 18], [13, 19], [13, 20], [13, 2], [13, 1], [13, 7], [13, 8], [13, 9], [13, 4],
                          [13, 5], [13, 6], [13, 0],
                          [12, 21], [12, 18], [12, 19], [12, 2], [12, 1], [12, 10], [12, 7], [12, 8], [12, 3], [12, 4],
                          [12, 5], [12, 0]]
    fw_edge_list = np.concatenate((fw_edge_list_left, fw_edge_list_right), axis=0)

    # MF edge
    mf_edge_list_left = [[10, 3], [10, 4], [10, 7], [10, 2], [10, 12], [10, 21], [10, 18], [10, 14], [10, 15],
                         [7, 3], [7, 4], [7, 5], [7, 8], [7, 1], [7, 2], [7, 12], [7, 21], [7, 18],
                         [8, 4], [8, 5], [8, 6], [8, 9], [8, 1], [8, 2], [8, 12], [8, 13], [8, 19], [8, 20],
                         [9, 5], [9, 6], [9, 1], [9, 13], [9, 19], [9, 20], [9, 16], [9, 17]]
    mf_edge_list_right = [[21, 14], [21, 15], [21, 18], [21, 12], [21, 10], [21, 3], [21, 4], [21, 7], [21, 2],
                          [18, 14], [18, 15], [18, 16], [18, 19], [18, 13], [18, 12], [18, 10], [18, 7], [18, 2],
                          [19, 12], [19, 15], [19, 16], [19, 17], [19, 20], [19, 13], [19, 1], [19, 8], [19, 9],
                          [20, 13], [20, 16], [20, 17], [20, 1], [20, 8], [20, 9], [20, 5], [20, 6]]
    mf_edge_list = np.concatenate((mf_edge_list_left, mf_edge_list_right), axis=0)

    # DF edge
    df_edge_list_left = [[3, 0], [3, 4], [3, 7], [3, 10], [3, 21], [3, 12],
                         [4, 0], [4, 5], [4, 8], [4, 7], [4, 10], [4, 12], [4, 13], [4, 21],
                         [5, 0], [5, 6], [5, 9], [5, 8], [5, 7], [5, 12], [5, 13], [5, 20],
                         [6, 0], [6, 9], [6, 8], [6, 13], [6, 20]]
    df_edge_list_right = [[14, 21], [14, 18], [14, 15], [14, 11], [14, 10], [14, 2],
                          [15, 21], [15, 18], [15, 19], [15, 16], [15, 11], [15, 10], [15, 2], [15, 1],
                          [16, 18], [16, 19], [16, 20], [16, 17], [16, 11], [16, 2], [16, 1], [16, 9],
                          [17, 19], [17, 20], [17, 11], [17, 1], [17, 9]]
    df_edge_list = np.concatenate((df_edge_list_left, df_edge_list_right), axis=0)

    # GK edge
    gk_edge_list_left = [[0, 3], [0, 4], [0, 5], [0, 6], [0, 21], [0, 12], [0, 13], [0, 20]]
    gk_edge_list_right = [[11, 14], [11, 15], [11, 16], [11, 17], [11, 10], [11, 2], [11, 1], [11, 9]]
    gk_edge_list = np.concatenate((gk_edge_list_left, gk_edge_list_right), axis=0)

    return [roop_edge_list, ball_edge_list, fw_edge_list, mf_edge_list, df_edge_list, gk_edge_list]


def get_multi_A(edge_lists):
    A = np.zeros((len(edge_lists), 23, 23))
    for i, edge_list in enumerate(edge_lists):
        adj = np.zeros((23, 23))
        for edge in edge_list:
            if adj[edge[0]][edge[1]] != 1:
                adj[edge[0]][edge[1]] = adj[edge[1]][edge[0]] = 1
        A[i] = adj

    return A


def get_single_A(edge_lists):
    A = np.zeros((1, 23, 23))
    adj = np.zeros((23, 23))
    for edge_list in edge_lists:
        for edge in edge_list:
            if adj[edge[0]][edge[1]] != 1:
                adj[edge[0]][edge[1]] = adj[edge[1]][edge[0]] = 1
    A[0] = adj

    return A


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)

    return DAD


def get_A(multi_A):
    edge_lists = soccer_graph_edge_list()

    if multi_A:
        A = get_multi_A(edge_lists)
    else:
        A = get_single_A(edge_lists)

    for i in range(len(A)):
        A[i] = normalize_undigraph(A[i])

    return A


if __name__ == '__main__':
    A = get_A(multi_A=False)

    # for a in A:
    #    print(a)

    print(A)
