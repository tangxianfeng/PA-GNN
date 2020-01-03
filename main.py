"""
Code for robust GAT which transfer knowledge from multiple graphs
In addition to the original attention mechanism, we force the attention score between missing edges are higher than those on newly-added edges
"""

#%%
import os,sys
import tensorflow as tf
import numpy as np
import random
from metattack import utils
from utils import sample_reddit, load_graphs, GraphData, compute_diff
from GAT.utils.process import flat_adj_to_bias
from metattack import meta_gradient_attack as mtk
import scipy.sparse as sp
from tensorflow.contrib import slim
import argparse
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, desc=None: x


from ipdb import set_trace
from copy import copy
import pickle

from models import PAGNN




#%%


seed = 16
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


hidden_sizes = [16] # tune in [8, 16, 32]
head = [8]

# dataset
setting = 'pubmed'
# how to generate ptb for source graphs, so that we can train PAGNN
meta_ptb_method = 'metattack'
# how the target graph is ptb
ptb_method = 'metattack'
rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
if setting == 'reddit':
    head = [16]
if setting == 'yelp':
    hidden_sizes = [32]
if setting == 'yelp_large': # ptb with metattack is extremely slow, try fewer settings
    rates = [0.1, 0.2, 0.3]
if ptb_method == 'random':
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
data_loader = GraphData(acceptale_ptb_rate=rates)

# prepare data
ptb_graphs_dict = {}
added = {}
for rate in rates:
    cln_graphs, ptb_graphs = data_loader.load_graph(setting, ptb_method, rate)
    print('*****')
    ptb_graphs_dict[rate] = ptb_graphs
    
    added[rate] = [compute_diff(cln_graphs[i][0], ptb_graphs[i])[0] for i in range(len(cln_graphs))]


nb_task = len(cln_graphs) - 1
target = cln_graphs[0]
# 0.1 train, 0.2 val, remained test
target_train, target_val, target_test = np.split(np.random.permutation(target[0].shape[0]), [int(target[0].shape[0] * 0.1), int(target[0].shape[0] * 0.3)])

nway = target[2].shape[-1] # label dim
ndim = target[1].shape[-1] # ftr dim
beta = 1.0
dist = 200.0
is_train = True
K = 5




#%% pretrain PAGNN, try different rate for best oerfirna
if is_train:
    # train PAGNN, under different settings
    # meta_rate: ptb rate for source graphs
    for meta_rate in [0.2, 0.4, 0.6, 0.8, 1.0]:
        print(f'\n\n*** train PAGNN for {setting} under ptb rate {meta_rate}')
        perturb_graphs = ptb_graphs_dict[meta_rate]
        added_graphs = added[meta_rate]
        # nb_task x [cln_A, X, Y, added_A]
        subgraphs = [cln_graphs[i] + [added_graphs[i],] for i in range(1, nb_task + 1)]

        model = PAGNN(subgraphs, ndim, nway, hidden_sizes, head, beta=beta, dist=dist)
        model.build(K)
        model.train(f'{meta_ptb_method}_{setting}_{hidden_sizes[0]}_{head[0]}_{meta_rate}_{beta}_{dist}', max_iter=200, is_train=True)
        model.sess.close()
        tf.reset_default_graph()
    print(f'*** finish training PAGNN for {setting}\n\n')

#%% train on source graphs
meta_rate = 0.2 # change this rate to the best one
print(f'*** loading PAGNN for {setting} under ptb rate {meta_rate}')
perturb_graphs = ptb_graphs_dict[meta_rate]
added_graphs = added[meta_rate]
# nb_task x [cln_A, X, Y, added_A]
subgraphs = [cln_graphs[i] + [cln_graphs[i][0],] for i in range(1, nb_task + 1)]

model = PAGNN(subgraphs, ndim, nway, hidden_sizes, head, beta=beta, dist=dist)
model.build(K)
model.train(f'{meta_ptb_method}_{setting}_{hidden_sizes[0]}_{head[0]}_{meta_rate}_{beta}_{dist}', max_iter=200, is_train=False)
print(f'*** PAGNN for {setting} under ptb rate {meta_rate} loaded')

#%% finetune on target graph, report performance
target[0] = target[0].astype(np.float32)
origin_target = copy(target)

perturb_graph = copy(ptb_graphs_dict[rate][0]).astype(np.float32)
ptb_finetune = model.finetune((perturb_graph, target[1], target[2], added_graphs[0]), target_train, target_val, target_test, 200)
print(ptb_finetune)
