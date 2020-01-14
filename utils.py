def preprocess(handle):
    features, supports = agent_obs[handle]

    feature = torch.sparse.FloatTensor(torch.from_numpy(features[0]).long().t(),
                                       torch.from_numpy(features[1]),
                                       features[2]).float().to(device)

    support = torch.sparse.FloatTensor(torch.from_numpy(supports[0]).long().t(),
                                       torch.from_numpy(supports[1]),
                                       supports[2]).float().to(device)

    return feature, support


def preprocess_features(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = features.sum(1)  # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten()  # 1/rowsum, [2708]
    r_inv[np.isinf(r_inv)] = 0.  # zero inf data
    r_mat_inv = sp.diags(r_inv)  # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features)  # D^-1:[2708, 2708]@X:[2708, 2708]
    return sparse_to_tuple(features)  # [coordinates, data, shape], []


def preprocess_features(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = features.sum(1)  # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten()  # 1/rowsum, [2708]
    r_inv[np.isinf(r_inv)] = 0.  # zero inf data
    r_mat_inv = sp.diags(r_inv)  # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features)  # D^-1:[2708, 2708]@X:[2708, 2708]
    return sparse_to_tuple(features)  # [coordinates, data, shape], []


def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = coo_matrix(mx)
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


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj((adj + sp.eye(adj.shape[0])).astype('bool'))
    return sparse_to_tuple(adj_normalized)

