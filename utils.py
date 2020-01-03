#%%
import numpy as np
import scipy.sparse as sp
from metattack import utils
import sys, os
import pickle
from metattack import meta_gradient_attack as mtk
from gcn.utils import load_data
import tensorflow as tf
from tensorflow.contrib import slim
from sklearn.metrics import f1_score

# from normalization import fetch_normalization, row_normalize
from time import perf_counter
from ipdb import set_trace

from copy import copy



#%%

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

#%%



def mtl_f1(predicted, actual, type = 'micro'):
    # pred: N x D
    predicted = tf.round(tf.nn.sigmoid(predicted)), 
    actual = tf.round(actual)
    predicted = tf.cast(predicted, tf.bool)
    predicted = tf.cast(predicted, tf.bool)
    TP = tf.count_nonzero(predicted * actual, 0)
    # TN = tf.count_nonzero((predicted - 1) * (actual - 1), 0)
    FP = tf.count_nonzero(predicted * (actual - 1), 0)
    FN = tf.count_nonzero((predicted - 1) * actual, 0)
    
    if type == 'micro':
        TP = tf.reduce_sum(TP)
        FP = tf.reduce_sum(FP)
        FN = tf.reduce_sum(FN)


    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return tf.reshape(tf.reduce_mean(f1), [-1])

#%% test gat

#%%



def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

#%%
# sample reddit graphs for our problem
def sample_reddit():
    reddit_path = 'path_to_reddit'
    # please download the reddit data from here: http://snap.stanford.edu/graphsage/reddit.zip 
    results = loadRedditFromNPZ(reddit_path)
    A = results[0]
    X = results[1]
    Y = np.zeros((X.shape[0],))
    Y[results[5]] = results[2]
    Y[results[6]] = results[3]
    Y[results[7]] = results[4]

    return A, X, Y
    
    # groups = [[] for _ in range(int(Y.max() - Y.min() + 1))]
    # for y in range(Y.shape[0]):
    #     groups[int(Y[y])].append(y)
    # nb_nodes_per_group = [len(g) for g in groups]


def get_perturbated_graph(origin_graph, split, rate = 0.10, variant = "A-Meta-Self", gpu_id = '0', isMTL = False):
    share_perturbation = rate
    hidden_sizes = [16]
    _A_obs, _X_obs, _z_obs = origin_graph
    _A_obs.setdiag(0)
    _A_obs = _A_obs.astype("float32")
    _A_obs.eliminate_zeros()
    _X_obs = _X_obs.astype("float32")

    # assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
    # assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
    # assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    _N = _A_obs.shape[0]
    _K = _z_obs.shape[1]
    _Z_obs = _z_obs
    _An = utils.preprocess_graph(_A_obs)
    sizes = [16, _K]
    degrees = _A_obs.sum(0).A1

    unlabeled_share = 0.8
    val_share = 0.1
    train_share = 1 - unlabeled_share - val_share

    split_train, split_val, split_unlabeled = split
    split_unlabeled = np.union1d(split_val, split_unlabeled)
    
    perturbations = int(share_perturbation * (_A_obs.sum()//2))
    train_iters = 100
    dtype = tf.float32 # change this to tf.float16 if you run out of GPU memory. Might affect the performance and lead to numerical instability


    #%%
    surrogate = mtk.GCNSparse(_A_obs, _X_obs, _Z_obs, hidden_sizes, isMTL=isMTL, gpu_id=gpu_id)
    surrogate.build(with_relu=False)
    surrogate.train(split_train)


    #%%
    # Predict the labels of the unlabeled nodes to use them for self-training.
    if not isMTL:
        labels_self_training = np.eye(_K)[surrogate.logits.eval(session=surrogate.session).argmax(1)]
    else:
        labels_self_training = np.round(sigmoid(surrogate.logits.eval(session=surrogate.session)))
    labels_self_training[split_train] = _Z_obs[split_train]




    enforce_ll_constrant = False
    approximate_meta_gradient = False
    if variant.startswith("A-"): # approximate meta gradient
        approximate_meta_gradient = True
        if "Train" in variant:
            lambda_ = 1
        elif "Self" in variant:
            lambda_ = 0
        else:
            lambda_ = 0.5
            
    if "Train" in variant:
        idx_attack = split_train
    elif "Self" in variant:
        idx_attack = split_unlabeled
    else:  # Both
        idx_attack = np.union1d(split_train, split_unlabeled)


    #%%
    if approximate_meta_gradient:
        gcn_attack = mtk.GNNMetaApprox(_A_obs, _X_obs, labels_self_training, hidden_sizes, 
                                    gpu_id=gpu_id, _lambda=lambda_, train_iters=train_iters, dtype=dtype, isMTL = isMTL)
    else:
        if sp.issparse(_X_obs):
            _X_obs = _X_obs.toarray().astype("float32")
        gcn_attack = mtk.GNNMeta(_A_obs, _X_obs, labels_self_training, hidden_sizes, 
                                gpu_id=gpu_id, attack_features=False, train_iters=train_iters, dtype=dtype, isMTL = isMTL)


    #%%
    gcn_attack.build()
    gcn_attack.make_loss(ll_constraint=enforce_ll_constrant)


    #%%
    if approximate_meta_gradient:
        gcn_attack.attack(perturbations, split_train, split_unlabeled, idx_attack)
    else:
        gcn_attack.attack(perturbations, split_train, idx_attack)


    #%%
    # adjacency_changes = gcn_attack.adjacency_changes.eval(session=gcn_attack.session).reshape(_A_obs.shape)
    modified_adjacency = gcn_attack.modified_adjacency.eval(session=gcn_attack.session)
    return sp.csr_matrix(modified_adjacency)


#%%



def compute_diff(clean_A, perturb_A):
    # return a mask between perturb_A and clean_A, the shape is their larger one
    added = perturb_A - clean_A
    added[added<0] = 0
    removed = clean_A - perturb_A
    removed[removed<0] = 0
    return added.astype(np.float32), removed.astype(np.float32)

class GraphData(object):
    def __init__(self, cache_dir = 'data/dest', acceptale_ptb_rate = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]):
        self.root = cache_dir
        self.acceptale_ptb_rate = acceptale_ptb_rate


    ####################### sampling method ########################### 
    # Use sample_{setting} to define sampling methods


    def sample_reddit(self):
        A, X, Y = sample_reddit()
        Y = Y.astype(np.int)
        node_gp = [[] for _ in range(np.unique(Y).shape[0])]
        for y in range(Y.shape[0]):
            node_gp[Y[y]].append(y)

        cnt = [len(gp) for gp in node_gp]
        cnt = list(zip(cnt, list(range(len(cnt)))))
        cnt = sorted(cnt, key = lambda x:-x[0])
        sel_gp = list(zip(*cnt[1:8]))[1]
        sel_node = []
        for gp in sel_gp:
            sel_node.extend(node_gp[gp])

        sel_node = sorted(sel_node)

        A = A + A.T
        A[A > 1] = 1
        # A = A.astype(np.float32)

        A = A[sel_node][:,sel_node]
        X = X[sel_node]
        Y = Y[sel_node]

        # map label to new label
        label_mapper = dict(zip(sel_gp, list(range(len(sel_gp)))))
        Y_new = []
        for y in Y:
            Y_new.append(label_mapper[y])
        Y = np.asarray(Y_new)

        nb_node = 4000
        nb_graph = 5
        graphs = []
        nodes = np.random.permutation(A.shape[0])
        g = 0
        while len(graphs) < nb_graph and nb_node * (g +1) < A.shape[0]:
            sel_nodes = nodes[nb_node * g: nb_node * (g +1)]
            g += 1
            _Y = Y[sel_nodes]
            _A = A[sel_nodes][:,sel_nodes]
            _X = X[sel_nodes]
            _Y = np.eye(len(sel_gp))[_Y]
            sel = utils.largest_connected_components(_A)
            _A = _A[sel][:,sel]
            _X = _X[sel]
            _Y = _Y[sel]
            Y_sum = _Y.sum(0)
            if 2 * Y_sum.min() < Y_sum.max():
                print(f'Imbalance sample {list(Y_sum)}, retry')
                continue
            print('Reddit: num node {}, num edge {}'.format(len(list(sel)), _A.sum()))
            graphs.append([_A.astype(np.float32),_X.astype(np.float32),_Y])
        return graphs

    def sample_pubmed(self):
        A, X, Y = load_data('pubmed')

        A = A + A.T
        A[A > 1] = 1
        

        nb_node = 4000
        nb_graph = 5
        graphs = []
        nodes = np.random.permutation(A.shape[0])
        g = 0
        for g in range(nb_graph):
            sel_nodes = nodes[nb_node * g: nb_node * (g +1)]
            _Y = Y[sel_nodes]
            _A = A[sel_nodes][:,sel_nodes]
            _X = X[sel_nodes]
            sel = utils.largest_connected_components(_A)
            _A = _A[sel][:,sel]
            _X = _X[sel]
            _Y = _Y[sel]
            print('Pubmed: num node {}, num edge {}'.format(len(list(sel)), _A.sum()))
            graphs.append([_A.astype(np.float32), sp.csr_matrix(_X), _Y])
        return graphs

    def sample_yelp(self):
        graphs = pickle.load(open('data/yelp_graphs.pkl', 'rb'))
        for g in graphs:
            g[0] = g[0] + g[0].T
            g[0][g[0] > 1] = 1
            g[0].eliminate_zeros()
        return graphs

    def sample_yelp_large(self):
        graphs = pickle.load(open('data/yelp_region_graphs.pkl', 'rb'))
        for g in graphs:
            g[0] = g[0] + g[0].T
            g[0][g[0] > 1] = 1
            g[0].eliminate_zeros()
        return graphs

    
    ####################### sampling method end #######################

    ####################### perturbation method #######################

    def ptb_metattack(self, graph, ptb_rate, gpu, isMTL, sub_size = 4000):
        nb_node = graph[0].shape[0]
        nb_sub = nb_node // sub_size + 1
        ptb = copy(graph[0].astype(np.float32))
        if nb_sub > 1:
            print(f'graph is too large, split to {nb_sub} sub graphs')
        for i_sub in range(nb_sub):
            nb_nd_st = i_sub * sub_size
            nb_nd_ed = (i_sub + 1) * sub_size
            if nb_nd_ed > nb_node:
                nb_nd_ed = nb_node
            sub_graph = [graph[0][nb_nd_st:nb_nd_ed, nb_nd_st:nb_nd_ed], graph[1][nb_nd_st:nb_nd_ed], graph[2][nb_nd_st:nb_nd_ed]]
            sub_nb_node = sub_graph[0].shape[0]
            target_train, target_val, target_test = np.split(np.random.permutation(sub_nb_node), [int(sub_nb_node * 0.5), int(sub_nb_node * 0.8)])
            print(f'start ptb sub graph {i_sub}')
            sub_ptb = get_perturbated_graph(sub_graph, (target_train, target_val, target_test), rate=ptb_rate, gpu_id=gpu, isMTL=isMTL)
            sub_ptb = sub_ptb.astype(np.float32)
            ptb[nb_nd_st:nb_nd_ed, nb_nd_st:nb_nd_ed] = sub_ptb

        return ptb.astype(np.float32)

    def ptb_random(self, graph, ptb_rate, gpu, isMTL):
        adjacency_matrix = graph[0]
        nb_node = graph[0].shape[0]
        exist_edge = list(np.array(adjacency_matrix.nonzero()).T)
        exist_edge = [(int(e[0]), int(e[1])) for e in exist_edge]
        exist_edge = set(exist_edge)
        new_edge = []
        neg_edge = []
        while len(new_edge) + len(neg_edge) < len(exist_edge) * ptb_rate:
            left, right = np.random.random_integers(0, nb_node-1), np.random.random_integers(0, nb_node-1)
            if not (left, right) in exist_edge:
                new_edge.append([left, right])
            if not (right, left) in exist_edge:
                new_edge.append([right, left])
            if (left, right) in exist_edge:
                neg_edge.append([left, right])
            if (right, left) in exist_edge:
                neg_edge.append([right, left])

        row, column = zip(*(new_edge + neg_edge))
        perturbated_edges = sp.csr_matrix((np.asarray([1.0] * len(new_edge) + [-1.0] * len(neg_edge), dtype=np.float32), (row, column)), shape = (nb_node, nb_node))
        perturbed_A = adjacency_matrix + perturbated_edges
        perturbed_A[perturbed_A>1.0] = 1.0
        perturbed_A.eliminate_zeros()
        return perturbed_A.astype(np.float32)

    def ptb_target(self, graph, ptb_rate, gpu):
        from nettack import utils as ne_utils
        from nettack import GCN as ne_GCN
        from nettack import nettack as ntk
        gpu_id = gpu
        nb_node = ptb_rate
        _A_obs, _X_obs, _z_obs = copy(graph)
        

        _X_obs = sp.csr_matrix(_X_obs).astype('float32')

        _N = _A_obs.shape[0]
        _K = _z_obs.shape[1]
        _Z_obs = _z_obs
        _z_obs = np.argmax(_Z_obs, 1)
        _An = ne_utils.preprocess_graph(_A_obs)
        sizes = [16, _K]
        degrees = _A_obs.sum(0).A1

        seed = 0
        unlabeled_share = 0.8
        val_share = 0.1
        train_share = 1 - unlabeled_share - val_share
        np.random.seed(seed)

        split_train, split_val, split_unlabeled = ne_utils.train_val_test_split_tabular(np.arange(_N),
                                                                            train_size=train_share,
                                                                            val_size=val_share,
                                                                            test_size=unlabeled_share,
                                                                            stratify=_z_obs)
        
        attacked = set()
        blacklist = set()
        while len(attacked) < nb_node:
            u = np.random.choice(split_unlabeled)
            while u in attacked or u in blacklist:
                u = np.random.choice(split_unlabeled)
            try:
                surrogate_model = ne_GCN.GCN(sizes, _An, _X_obs, with_relu=False, name="surrogate", gpu_id=gpu_id)
                surrogate_model.train(split_train, split_val, _Z_obs)
                W1 =surrogate_model.W1.eval(session=surrogate_model.session)
                W2 =surrogate_model.W2.eval(session=surrogate_model.session)
                nettack = ntk.Nettack(_A_obs, _X_obs, _z_obs, W1, W2, u, verbose=False)
                direct_attack = True
                n_influencers = 1 if direct_attack else 5
                n_perturbations = int(degrees[u]) # How many perturbations to perform. Default: Degree of the node
                perturb_features = False
                perturb_structure = True
                nettack.attack_surrogate(n_perturbations, perturb_structure=perturb_structure, perturb_features=perturb_features, direct=direct_attack, n_influencers=n_influencers)
                surrogate_model.session.close()
                tf.reset_default_graph()
            except:
                blacklist.add(u)
                continue
            attacked.add(u)
            _A_obs = nettack.adj.tocsr()
            _An = ne_utils.preprocess_graph(_A_obs)
        
        return _An, list(attacked)








    #####################perturbation method end ######################
    
    def load_graph(self, setting, perturbation_method, perturbation_rate, gpu = '0'):
        setting_root = os.path.join(self.root, setting)
        if not os.path.isdir(setting_root):
            os.mkdir(setting_root)

        # load cln:
        cln_graphs = []
        cln_graph_file = os.path.join(setting_root, 'cln.pkl')
        if os.path.isfile(cln_graph_file):
            cln_graphs = pickle.load(open(cln_graph_file, 'rb'))
            if setting == 'ppi':
                cln_graphs = [[g[0], g[1], g[2][:,:10]] for g in cln_graphs]
        else:
            try:
                cln_graphs = getattr(self, 'sample_' + setting)()
            except:
                print(f'Please implenment the sampling method for \"{setting}\" first!')
                return None, None
            pickle.dump(cln_graphs, open(cln_graph_file, 'wb'))
        if not perturbation_rate in self.acceptale_ptb_rate:
            return cln_graphs, None

        ptb_graphs = []
        ptb_setting_root = os.path.join(setting_root, perturbation_method)
        if not os.path.isdir(ptb_setting_root):
            os.mkdir(ptb_setting_root)
        ptb_graph_file = os.path.join(ptb_setting_root, f'ptb_{perturbation_rate}.pkl')
        if os.path.isfile(ptb_graph_file):
            ptb_graphs = pickle.load(open(ptb_graph_file, 'rb'))
        else:
            for i in range(len(cln_graphs)):
                # always use the first 10% nodes as atk
                try:
                    perturbed_A = getattr(self, 'ptb_' + perturbation_method)(cln_graphs[i], perturbation_rate, gpu, True if setting == 'ppi' else False)
                except:
                    print(f'Please implenment the perturbation method for \"{perturbation_method}\" first!')
                    return cln_graphs, None
                if not perturbation_method == 'target':
                    ptb_graphs.append(perturbed_A)
                else:
                    ptb_graphs.append(perturbed_A[0])
                    pickle.dump(perturbed_A[1], open(os.path.join(ptb_setting_root, 'graph0_sel_nodes.pkl'), 'wb'))
                # for remained graphs, only ptb the first one. For remained clean graphs, there is no need for ptb
                if perturbation_method != 'metattack':
                    mtk_graphs = [cn[0] for cn in cln_graphs[1:]]
                    ptb_graphs.extend(mtk_graphs)
                    break
            pickle.dump(ptb_graphs, open(ptb_graph_file, 'wb'))

        print('Graph statistic:')
        for i in range(len(cln_graphs)):
            print(f"""Graph {i}: {cln_graphs[i][0].shape[0]} nodes, 
            {cln_graphs[i][0].nonzero()[0].shape[0]} edges,
            feature dim {cln_graphs[i][1].shape[1]}, 
            label num {cln_graphs[i][2].shape[1]}""")
            # {ptb_graphs[i].nonzero()[0].shape[0] - cln_graphs[i][0].nonzero()[0].shape[0]} ptb edges,
        return cln_graphs, ptb_graphs


class PreprocessGCN(mtk.GCNSparse):
    """
        Class for preprocessing based GCN
    """
    def __init__(self, adjacency_matrix, attribute_matrix, labels_onehot, hidden_sizes, preprocessed_path, setting, rate, isMTL = False, gpu_id=None):
        """
        Parameters
        ----------
        adjacency_matrix: sp.spmatrix [N,N]
                Unweighted, symmetric adjacency matrix where N is the number of nodes. Should be a scipy.sparse matrix.

        attribute_matrix: sp.spmatrix or np.array [N,D]
            Attribute matrix where D is the number of attributes per node. Can be sparse or dense.

        labels_onehot: np.array [N,K]
            One-hot matrix of class labels, where N is the number of nodes. Labels of the unlabeled nodes should come
            from self-training using only the labels of the labeled nodes.

        hidden_sizes: list of ints
            List that defines the number of hidden units per hidden layer. Input and output layers not included.

        gpu_id: int or None
            GPU to use. None means CPU-only

        """
        self.isMTL = isMTL
                
        if not sp.issparse(adjacency_matrix):
            raise ValueError("Adjacency matrix should be a sparse matrix.")

        self.N, self.D = attribute_matrix.shape
        self.K = labels_onehot.shape[1]
        self.hidden_sizes = hidden_sizes
        self.graph = tf.Graph()

        self.pp_path = os.path.join(preprocessed_path, setting, 'preprocess')
        if not os.path.isdir(self.pp_path):
            os.mkdir(self.pp_path)
        self.adj_path = os.path.join(self.pp_path, f'{rate}.pkl')
        if os.path.isfile(self.adj_path):
            adjacency_matrix = pickle.load(open(self.adj_path, 'rb'))
        else:
            # preprocess based on X
            isSparse = False
            if sp.issparse(attribute_matrix):
                isSparse = True
            edges = np.array(adjacency_matrix.nonzero()).T
            for edge in edges:
                if edge[0] < edge[1]:
                    if isSparse:
                        # Jaccard similarity
                        nb_shared_ftr = attribute_matrix[edge[0]].multiply(attribute_matrix[edge[1]]).count_nonzero()
                        J = nb_shared_ftr * 1.0 / (attribute_matrix[edge[0]].count_nonzero() + attribute_matrix[edge[1]].count_nonzero() - nb_shared_ftr)
                        if J < 0.8:
                            adjacency_matrix[edge[0],edge[1]] = 0
                            adjacency_matrix[edge[1],edge[0]] = 0
                    else:
                        # Cosine similarity
                        J = (attribute_matrix[edge[0]] * attribute_matrix[edge[1]]).sum() / np.sqrt(np.square(attribute_matrix[edge[0]]).sum() + np.square(attribute_matrix[edge[1]]).sum())
                        if J < 0:
                            adjacency_matrix[edge[0],edge[1]] = 0
                            adjacency_matrix[edge[1],edge[0]] = 0
            # adjacency_matrix
            pickle.dump(adjacency_matrix, open(self.adj_path, 'wb'))


        with self.graph.as_default():
            self.idx = tf.placeholder(tf.int32, shape=[None])
            self.labels_onehot = labels_onehot

            adj_norm = utils.preprocess_graph(adjacency_matrix).astype("float32")
            self.adj_norm = tf.SparseTensor(np.array(adj_norm.nonzero()).T,
                                            adj_norm[adj_norm.nonzero()].A1, [self.N, self.N])

            self.sparse_attributes = sp.issparse(attribute_matrix)

            if self.sparse_attributes:
                self.attributes = tf.SparseTensor(np.array(attribute_matrix.nonzero()).T,
                                                    attribute_matrix[attribute_matrix.nonzero()].A1, [self.N, self.D])
            else:
                self.attributes = tf.constant(attribute_matrix, dtype=tf.float32)

            w_init = slim.xavier_initializer
            self.weights = []
            self.biases = []

            previous_size = self.D
            for ix, layer_size in enumerate(self.hidden_sizes):
                weight = tf.get_variable(f"W_{ix + 1}", shape=[previous_size, layer_size], dtype=tf.float32,
                                            initializer=w_init())
                bias = tf.get_variable(f"b_{ix + 1}", shape=[layer_size], dtype=tf.float32,
                                        initializer=w_init())
                self.weights.append(weight)
                self.biases.append(bias)
                previous_size = layer_size
            weight_final = tf.get_variable(f"W_{len(hidden_sizes) + 1}", shape=[previous_size, self.K],
                                            dtype=tf.float32,
                                            initializer=w_init())
            bias_final = tf.get_variable(f"b_{len(hidden_sizes) + 1}", shape=[self.K], dtype=tf.float32,
                                            initializer=w_init())

            self.weights.append(weight_final)
            self.biases.append(bias_final)

            if gpu_id is None:
                config = tf.ConfigProto(
                    device_count={'GPU': 0}
                )
            else:
                gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
                config = tf.ConfigProto(gpu_options=gpu_options)

            session = tf.Session(config=config)
            self.session = session

            self.logits = None
            self.logits_gather = None
            self.loss = None
            self.optimizer = None
            self.train_op = None
            self.initializer = None

class JointTrainGCN(mtk.GCNSparse):
    """
    GCN implementation with a sparse adjacency matrix and possibly sparse attribute matrices. Note that this becomes
    the surrogate model from the paper if we set the number of layers to 2 and leave out the ReLU activation function
    (see build()).
    """

    def __init__(self, extra_graphs, adjacency_matrix, attribute_matrix, labels_onehot, hidden_sizes, gpu_id=None, isMTL = False):
        """
        Parameters
        ----------
        extra_graphs: [adjacency_matrix, attribute_matrix, labels_onehot] * K
                K extra graphs

        adjacency_matrix: sp.spmatrix [N,N]
                Unweighted, symmetric adjacency matrix where N is the number of nodes. Should be a scipy.sparse matrix.

        attribute_matrix: sp.spmatrix or np.array [N,D]
            Attribute matrix where D is the number of attributes per node. Can be sparse or dense.

        labels_onehot: np.array [N,K]
            One-hot matrix of class labels, where N is the number of nodes. Labels of the unlabeled nodes should come
            from self-training using only the labels of the labeled nodes.

        hidden_sizes: list of ints
            List that defines the number of hidden units per hidden layer. Input and output layers not included.

        gpu_id: int or None
            GPU to use. None means CPU-only

        """
        self.isMTL = isMTL
        if not sp.issparse(adjacency_matrix):
            raise ValueError("Adjacency matrix should be a sparse matrix.")

        self.N, self.D = attribute_matrix.shape
        self.K = labels_onehot.shape[1]
        self.hidden_sizes = hidden_sizes
        self.graph = tf.Graph()
        # graph 0 is the target graph
        self.num_graph = len(extra_graphs) + 1
        self.graphs = [[adjacency_matrix, attribute_matrix, labels_onehot],] + extra_graphs

        with self.graph.as_default():
            self.idx = tf.placeholder(tf.int32, shape=[None])
            self.labels_onehot = [graph[2] for graph in self.graphs]
            
            self.adj_norm = []
            for i in range(self.num_graph):
                _adj_norm = utils.preprocess_graph(self.graphs[i][0]).astype("float32")
                self.adj_norm.append(tf.SparseTensor(np.array(_adj_norm.nonzero()).T,
                                                _adj_norm[_adj_norm.nonzero()].A1, [_adj_norm.shape[0], _adj_norm.shape[1]]))



            self.sparse_attributes = sp.issparse(attribute_matrix)

            if self.sparse_attributes:
                self.attributes = [tf.SparseTensor(np.array(graph[1].nonzero()).T,
                                                  graph[1][graph[1].nonzero()].A1, [graph[1].shape[0], graph[1].shape[1]]) for graph in self.graphs]
            else:
                self.attributes = [tf.constant(graph[1], dtype=tf.float32) for graph in self.graphs]

            w_init = slim.xavier_initializer
            self.weights = []
            self.biases = []

            previous_size = self.D
            for ix, layer_size in enumerate(self.hidden_sizes):
                weight = tf.get_variable(f"W_{ix + 1}", shape=[previous_size, layer_size], dtype=tf.float32,
                                         initializer=w_init())
                bias = tf.get_variable(f"b_{ix + 1}", shape=[layer_size], dtype=tf.float32,
                                       initializer=w_init())
                self.weights.append(weight)
                self.biases.append(bias)
                previous_size = layer_size
            weight_final = tf.get_variable(f"W_{len(hidden_sizes) + 1}", shape=[previous_size, self.K],
                                           dtype=tf.float32,
                                           initializer=w_init())
            bias_final = tf.get_variable(f"b_{len(hidden_sizes) + 1}", shape=[self.K], dtype=tf.float32,
                                         initializer=w_init())

            self.weights.append(weight_final)
            self.biases.append(bias_final)

            if gpu_id is None:
                config = tf.ConfigProto(
                    device_count={'GPU': 0}
                )
            else:
                gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
                config = tf.ConfigProto(gpu_options=gpu_options)

            session = tf.Session(config=config)
            self.session = session

            self.logits = None
            self.logits_gather = None
            self.loss = None
            self.optimizer = None
            self.train_op = None
            self.initializer = None

    def build(self, with_relu=True, learning_rate=1e-2):
        with self.graph.as_default():
            losses = []
            for i in range(self.num_graph):
                hidden = self.attributes[i]
                for ix in range(len(self.hidden_sizes)):
                    w = self.weights[ix]
                    b = self.biases[ix]
                    if ix == 0 and self.sparse_attributes:
                        hidden = tf.sparse_tensor_dense_matmul(self.adj_norm[i],
                                                            tf.sparse_tensor_dense_matmul(hidden, w)) + b
                    else:
                        hidden = tf.sparse_tensor_dense_matmul(self.adj_norm[i], hidden @ w) + b

                    if with_relu:
                        hidden = tf.nn.relu(hidden)
                logits = tf.sparse_tensor_dense_matmul(self.adj_norm[i], hidden @ self.weights[-1]) + self.biases[-1]
                if i == 0:
                    self.logits = logits
                    logits_gather = tf.gather(logits, self.idx)
                    labels_gather = tf.gather(self.labels_onehot[i], self.idx)
                else:
                    logits_gather = logits
                    labels_gather = self.labels_onehot[i]
                if not self.isMTL:
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_gather, logits=logits_gather)
                else:
                    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_gather, logits=logits_gather)
                losses.append(loss)

            self.losses = losses
            self.loss = tf.concat(self.losses, axis = 0)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, var_list=[*self.weights, *self.biases])
            self.initializer = tf.local_variables_initializer()


class PretrainGCN(mtk.GCNSparse):
    """
    GCN implementation with a sparse adjacency matrix and possibly sparse attribute matrices. Note that this becomes
    the surrogate model from the paper if we set the number of layers to 2 and leave out the ReLU activation function
    (see build()).
    """

    def __init__(self, extra_graphs, adjacency_matrix, attribute_matrix, labels_onehot, hidden_sizes, gpu_id=None, isMTL = False):
        """
        Parameters
        ----------
        extra_graphs: [adjacency_matrix, attribute_matrix, labels_onehot] * K
                K extra graphs

        adjacency_matrix: sp.spmatrix [N,N]
                Unweighted, symmetric adjacency matrix where N is the number of nodes. Should be a scipy.sparse matrix.

        attribute_matrix: sp.spmatrix or np.array [N,D]
            Attribute matrix where D is the number of attributes per node. Can be sparse or dense.

        labels_onehot: np.array [N,K]
            One-hot matrix of class labels, where N is the number of nodes. Labels of the unlabeled nodes should come
            from self-training using only the labels of the labeled nodes.

        hidden_sizes: list of ints
            List that defines the number of hidden units per hidden layer. Input and output layers not included.

        gpu_id: int or None
            GPU to use. None means CPU-only

        """
        self.isMTL = isMTL
        if not sp.issparse(adjacency_matrix):
            raise ValueError("Adjacency matrix should be a sparse matrix.")

        self.N, self.D = attribute_matrix.shape
        self.K = labels_onehot.shape[1]
        self.hidden_sizes = hidden_sizes
        self.graph = tf.Graph()
        # graph 0 is the target graph
        self.num_graph = len(extra_graphs) + 1
        self.graphs = [[adjacency_matrix, attribute_matrix, labels_onehot],] + extra_graphs

        with self.graph.as_default():
            self.idx = tf.placeholder(tf.int32, shape=[None])
            self.labels_onehot = [graph[2] for graph in self.graphs]
            
            self.adj_norm = []
            for i in range(self.num_graph):
                _adj_norm = utils.preprocess_graph(self.graphs[i][0]).astype("float32")
                self.adj_norm.append(tf.SparseTensor(np.array(_adj_norm.nonzero()).T,
                                                _adj_norm[_adj_norm.nonzero()].A1, [_adj_norm.shape[0], _adj_norm.shape[1]]))



            self.sparse_attributes = sp.issparse(attribute_matrix)

            if self.sparse_attributes:
                self.attributes = [tf.SparseTensor(np.array(graph[1].nonzero()).T,
                                                  graph[1][graph[1].nonzero()].A1, [graph[1].shape[0], graph[1].shape[1]]) for graph in self.graphs]
            else:
                self.attributes = [tf.constant(graph[1], dtype=tf.float32) for graph in self.graphs]

            w_init = slim.xavier_initializer
            self.weights = []
            self.biases = []

            previous_size = self.D
            for ix, layer_size in enumerate(self.hidden_sizes):
                weight = tf.get_variable(f"W_{ix + 1}", shape=[previous_size, layer_size], dtype=tf.float32,
                                         initializer=w_init())
                bias = tf.get_variable(f"b_{ix + 1}", shape=[layer_size], dtype=tf.float32,
                                       initializer=w_init())
                self.weights.append(weight)
                self.biases.append(bias)
                previous_size = layer_size
            weight_final = tf.get_variable(f"W_{len(hidden_sizes) + 1}", shape=[previous_size, self.K],
                                           dtype=tf.float32,
                                           initializer=w_init())
            bias_final = tf.get_variable(f"b_{len(hidden_sizes) + 1}", shape=[self.K], dtype=tf.float32,
                                         initializer=w_init())

            self.weights.append(weight_final)
            self.biases.append(bias_final)

            if gpu_id is None:
                config = tf.ConfigProto(
                    device_count={'GPU': 0}
                )
            else:
                gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
                config = tf.ConfigProto(gpu_options=gpu_options)

            session = tf.Session(config=config)
            self.session = session

            self.logits = None
            self.logits_gather = None
            self.loss = None
            self.optimizer = None
            self.train_op = None
            self.initializer = None

    def build(self, with_relu=True, learning_rate=1e-2):
        # this build will call self.pretune so that extra graphs are utilized
        with self.graph.as_default():
            losses = []
            for i in range(1, self.num_graph):
                hidden = self.attributes[i]
                for ix in range(len(self.hidden_sizes)):
                    w = self.weights[ix]
                    b = self.biases[ix]
                    if ix == 0 and self.sparse_attributes:
                        hidden = tf.sparse_tensor_dense_matmul(self.adj_norm[i],
                                                            tf.sparse_tensor_dense_matmul(hidden, w)) + b
                    else:
                        hidden = tf.sparse_tensor_dense_matmul(self.adj_norm[i], hidden @ w) + b

                    if with_relu:
                        hidden = tf.nn.relu(hidden)
                logits = tf.sparse_tensor_dense_matmul(self.adj_norm[i], hidden @ self.weights[-1]) + self.biases[-1]
                if i == 0:
                    self.logits = logits
                    self.logits_gather = tf.gather(logits, self.idx)
                    self.labels_gather = tf.gather(self.labels_onehot[i], self.idx)
                else:
                    logits_gather = logits
                    labels_gather = self.labels_onehot[i]
                if not self.isMTL:
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_gather, logits=logits_gather)
                else:
                    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_gather, logits=logits_gather)
                losses.append(loss)

            # pre train
            self.losses = losses
            self.loss = tf.concat(self.losses, axis = 0)

            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, var_list=[*self.weights, *self.biases])
            self.initializer = tf.local_variables_initializer()
            self.session.run(tf.global_variables_initializer())
            _iter = range(200)
            for _it in _iter:
                self.session.run(self.train_op)
            i = 0
            hidden = self.attributes[i]
            for ix in range(len(self.hidden_sizes)):
                w = self.weights[ix]
                b = self.biases[ix]
                if ix == 0 and self.sparse_attributes:
                    hidden = tf.sparse_tensor_dense_matmul(self.adj_norm[i],
                                                        tf.sparse_tensor_dense_matmul(hidden, w)) + b
                else:
                    hidden = tf.sparse_tensor_dense_matmul(self.adj_norm[i], hidden @ w) + b

                if with_relu:
                    hidden = tf.nn.relu(hidden)
            logits = tf.sparse_tensor_dense_matmul(self.adj_norm[i], hidden @ self.weights[-1]) + self.biases[-1]
            self.logits = logits
            logits_gather = tf.gather(logits, self.idx)
            labels_gather = tf.gather(self.labels_onehot[i], self.idx)
            if not self.isMTL:
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_gather, logits=logits_gather)
            else:
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_gather, logits=logits_gather)
            self.loss = loss
            self.train_op = self.optimizer.minimize(self.loss, var_list=[*self.weights, *self.biases])


class SecondPtbGAT(mtk.GCNSparse):
    """
    GAT implementation with a attn penalty on perturbed edges
    """

    def __init__(self, adjacency_matrix, perturbated_edges, attribute_matrix, labels_onehot, hidden_sizes, head, isMTL = False, gpu_id=None, enable_penalty = False):
        """
        Parameters
        ----------
        adjacency_matrix: sp.spmatrix [N,N]
                Unweighted, symmetric adjacency matrix where N is the number of nodes. Should be a scipy.sparse matrix.

        attribute_matrix: sp.spmatrix or np.array [N,D]
            Attribute matrix where D is the number of attributes per node. Can be sparse or dense.

        labels_onehot: np.array [N,K]
            One-hot matrix of class labels, where N is the number of nodes. Labels of the unlabeled nodes should come
            from self-training using only the labels of the labeled nodes.

        hidden_sizes: list of ints
            List that defines the number of hidden units per hidden layer. Input and output layers not included.

        gpu_id: int or None
            GPU to use. None means CPU-only

        """
        self.isMTL = isMTL
        if not sp.issparse(adjacency_matrix):
            raise ValueError("Adjacency matrix should be a sparse matrix.")

        self.N, self.D = attribute_matrix.shape
        self.K = labels_onehot.shape[1]
        self.hidden_sizes = hidden_sizes + [self.K,]
        self.head = head + [head[-1],]
        assert len(self.hidden_sizes) == len(self.head)
        self.graph = tf.Graph()
        self.enable_penalty = enable_penalty

        # generate some ptb edges
        if perturbated_edges == None:
            exist_edge = list(np.array(adjacency_matrix.nonzero()).T)
            exist_edge = [(int(e[0]), int(e[1])) for e in exist_edge]
            exist_edge = set(exist_edge)
            new_edge = []
            while len(new_edge) < len(exist_edge) * 0.2:
                left, right = np.random.random_integers(self.N) - 1, np.random.random_integers(self.N) -1
                if left == right:
                    continue
                if not (left, right) in exist_edge and not (right, left) in exist_edge:
                    new_edge.append([left, right])
                    new_edge.append([right, left])
            row, column = zip(*new_edge)
            perturbated_edges = sp.csr_matrix((np.asarray([1.0] * len(new_edge), dtype=np.float32), (row, column)), shape = (self.N, self.N))
            # print(perturbated_edges)
            # print(adjacency_matrix)

        with self.graph.as_default():
            self.idx = tf.placeholder(tf.int32, shape=[None])
            self.labels_onehot = labels_onehot

            adj_norm = adjacency_matrix
            self.adj_norm = tf.SparseTensor(np.array(adj_norm.nonzero()).T,
                                            adj_norm[adj_norm.nonzero()].A1, [self.N, self.N])

            # store purturbed edges
            self.ind = tf.SparseTensor(np.array(perturbated_edges.nonzero()).T,
                                            perturbated_edges[perturbated_edges.nonzero()].A1, [self.N, self.N])

                                

            self.sparse_attributes = sp.issparse(attribute_matrix)

            if self.sparse_attributes:
                self.attributes = tf.SparseTensor(np.array(attribute_matrix.nonzero()).T,
                                                  attribute_matrix[attribute_matrix.nonzero()].A1, [self.N, self.D])
            else:
                self.attributes = tf.constant(attribute_matrix, dtype=tf.float32)

            w_init = slim.xavier_initializer
            weights = {}

            previous_size = self.D
            for ix, layer_size in enumerate(self.hidden_sizes):
                for hd in range(self.head[ix]):
                    weight = tf.get_variable(f"W_{ix + 1}_{hd}", shape=[previous_size, layer_size], dtype=tf.float32,
                                                initializer=w_init())
                    bias = tf.get_variable(f"b_{ix + 1}_{hd}", shape=[layer_size], dtype=tf.float32,
                                            initializer=w_init())
                    a1 = tf.get_variable(f"a1_{ix + 1}_{hd}", shape=[layer_size, 1], dtype=tf.float32,
                                            initializer=w_init())
                    a2 = tf.get_variable(f"a2_{ix + 1}_{hd}", shape=[layer_size, 1], dtype=tf.float32,
                                            initializer=w_init())
                    weights[f'w{ix}_{hd}'] = weight
                    weights[f'b{ix}_{hd}'] = bias
                    weights[f'a1{ix}_{hd}'] = a1
                    weights[f'a2{ix}_{hd}'] = a2
                previous_size = layer_size * self.head[ix]
                
            self.weights = weights
            if gpu_id is None:
                config = tf.ConfigProto(
                    device_count={'GPU': 0}
                )
            else:
                gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
                config = tf.ConfigProto(gpu_options=gpu_options)

            session = tf.Session(config=config)
            self.session = session

            self.logits = None
            self.logits_gather = None
            self.loss = None
            self.optimizer = None
            self.train_op = None
            self.initializer = None

    def build(self, in_drop=0.0, coef_drop=0.0, with_relu=True, learning_rate=1e-2):
        with self.graph.as_default():
            weights = self.weights
            hidden = self.attributes
            head = self.head
            attn_scores = {}
            attn_common_edge = []
            attn_ptb_edge = []
            # attn_scores['name'] = ('origin_attn_avg','ptb_attn_avg')
            for ix in range(len(self.hidden_sizes)):
                attn_scores[ix] = []
                new_hidden = []
                for hd in range(head[ix]):
                    w = weights[f'w{ix}_{hd}']
                    b = weights[f'b{ix}_{hd}']
                    a1 = weights[f'a1{ix}_{hd}']
                    a2 = weights[f'a2{ix}_{hd}']
                    if ix == 0 and self.sparse_attributes:
                        _hidden = tf.sparse_tensor_dense_matmul(hidden, w)
                    else:
                        _hidden = hidden @ w

                    f_1 = _hidden @ a1
                    f_2 = _hidden @ a2

                    # transfer to sparse first
                    f1 = self.adj_norm * f_1
                    f2 = self.adj_norm * tf.transpose(f_2, [1,0])

                    raw_attn = tf.sparse_add(f1, f2)

                    origin_attn_avg = tf.reduce_sum(raw_attn.values) / tf.reduce_sum(self.adj_norm.values)
                    attn_common_edge.append(tf.reshape(origin_attn_avg, [-1]))

                    raw_attn = tf.SparseTensor(indices=raw_attn.indices, 
                            values=tf.nn.leaky_relu(raw_attn.values), 
                            dense_shape=raw_attn.dense_shape)


                    # attention scores
                    attn = tf.sparse_softmax(raw_attn)

                    # compute the attention scores for perturbed edges
                    _f1 = self.ind * f_1
                    _f2 = self.ind * tf.transpose(f_2, [1,0])

                    raw_attn_on_added_edge = tf.sparse_add(_f1, _f2)
                    raw_attn_on_added_edge = tf.SparseTensor(indices=raw_attn_on_added_edge.indices, 
                        values=tf.nn.leaky_relu(raw_attn_on_added_edge.values), 
                        dense_shape=raw_attn_on_added_edge.dense_shape)
                    ptb_attn_avg = tf.reduce_sum(raw_attn_on_added_edge.values) / tf.reduce_sum(self.ind.values)
                    attn_ptb_edge.append(tf.reshape(ptb_attn_avg, [-1]))

                    # dropout
                    if coef_drop != 0.0:
                        attn = tf.SparseTensor(indices=attn.indices,
                    values=tf.nn.dropout(attn.values, 1.0 - coef_drop),
                    dense_shape=attn.dense_shape)
                    if in_drop != 0.0:
                        _hidden = tf.nn.dropout(_hidden, 1.0 - in_drop)

                    _hidden = tf.sparse_tensor_dense_matmul(attn, _hidden) + b

                    if with_relu:
                        if ix != len(self.hidden_sizes) - 1:
                            _hidden = tf.nn.relu(_hidden)
                        else:
                            if not self.isMTL:
                                _hidden = tf.nn.relu(_hidden)
                    
                    new_hidden.append(_hidden)
                hidden = tf.concat(new_hidden, axis=-1)
            attn_common_edge_avg = tf.reduce_mean(tf.concat(attn_common_edge, axis = -1))
            attn_ptb_edge_avg = tf.reduce_mean(tf.concat(attn_ptb_edge, axis = -1))
            attn_score = tf.math.maximum(-100.0, attn_ptb_edge_avg - attn_common_edge_avg)
            


            self.attn_scores = attn_scores
            self.logits = tf.add_n(new_hidden) / head[-1]
            self.logits_gather = tf.gather(self.logits, self.idx)
            labels_gather = tf.gather(self.labels_onehot, self.idx)
            if not self.isMTL:
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_gather, logits=self.logits_gather)
            else:
                self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_gather, logits=self.logits_gather)
            self.loss = tf.reduce_mean(self.loss) + attn_score
            self.loss = tf.reshape(self.loss, [-1])
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, var_list=list(self.weights.values()))
            self.initializer = tf.local_variables_initializer()


#%% generate and save ptb results

if __name__ == '__main__':
    all_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    data_loader = GraphData(acceptale_ptb_rate=all_rates)
    for setting in ['reddit', 'pubmed', 'yelp', 'yelp_large']:
        for ptb_method in ['target']:
            for ptb_rate in all_rates:
                print(f'\n\nperturb {setting} by {ptb_method} under {ptb_rate}')
                cln_graphs, ptb_graphs = data_loader.load_graph(setting, ptb_method, ptb_rate, '1')

