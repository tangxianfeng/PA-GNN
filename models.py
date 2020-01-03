"""
Code for PA-GNN which transfer knowledge from multiple graphs
In addition to the original attention mechanism, we force the attention score between missing edges are higher than those on newly-added edges
"""

#%%
import os,sys
import tensorflow as tf
import numpy as np
import random
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



class PAGNN:
    def __init__(self, graphs, ndim, nway, hidden_sizes = [16], head = [8], meta_lr=1e-2, train_lr=1e-2, beta = 0.0, dist = 100.0):

        self.ndim = ndim                    # feature dimension
        self.nway = nway                    # class number
        self.meta_lr = meta_lr
        self.train_lr = train_lr
        self.hidden_sizes = hidden_sizes + [self.nway,]
        self.head = head + [head[-1],]
        assert len(self.hidden_sizes) == len(self.head)
        self.raw_graphs = []
        self.graphs = []
        self.nb_task = len(graphs)
        self.beta = beta
        self.dist = dist
        self.coef_drop = 0.0
        self.in_drop = 0.0

        # debug
        self.attn_values = {}

        self.init_graphs(graphs) # import graphs

        print('Graph Maml setting:', '# of labels:', self.nway, 'meta-lr:', meta_lr, 'train-lr:', train_lr)

    def init_graphs(self, graphs):
        """Save graphs as SparseTensors
        
        Arguments:
            graphs {[list]} -- [list of graphs]
        """
        self.raw_graphs.extend(graphs)
        with tf.variable_scope('MAML', reuse= tf.AUTO_REUSE):
            for _A, _X, _Y, added_edges in graphs:
                # sparse adjacency
                # sparse adjacency
                _A = _A.astype(np.float32)
                added_edges = added_edges.astype(np.float32)
                N, D = _X.shape
                A = tf.SparseTensor(np.array(_A.nonzero()).T, _A[_A.nonzero()].A1, [N, N])
                ind = tf.SparseTensor(np.array(added_edges.nonzero()).T, added_edges[added_edges.nonzero()].A1, [N, N])
                is_sparse_attributes = sp.issparse(_X)
                if is_sparse_attributes:
                    _X = _X.todense()
                X = tf.constant(_X, dtype=tf.float32)
                Y = tf.constant(_Y, dtype=tf.float32) # one-hot label
                self.graphs.append((A, X, Y, ind))

    def build(self, K = 5):
        """
        For graphs, labels are not needed. Given the node ids of the testing set is enough.
        support + query = nodes of a graph
        :param 
        :param 
        :param K:           train update steps
        :param meta_batchsz:tasks number (# of graphs)
        :param mode:        train/eval/test, for training, we build train&eval network meanwhile.
        :return:
        """
        # create or reuse network variable, not including batch_norm variable, therefore we need extra reuse mechnism
        # to reuse batch_norm variables.
        self.weights = self.gnn_weights()
        # TODO: meta-test is sort of test stage.
        # training = True if mode is 'train' else False

        # support_idxes:   [b, None], nodes for updating theta_i
        self.support_idxes = [tf.placeholder(tf.int32, shape = [None], name = f'support_idx_task_{i}') for i in range(self.nb_task)]
        # query_idxes:   [b, None], nodes for computing meta gradient, and updating theta
        self.query_idxes = [tf.placeholder(tf.int32, shape = [None], name = f'query_idx_task_{i}') for i in range(self.nb_task)]
        
        def meta_task(input):
            """
            pass the graph idx to this function to avoid copy
            """
            support_idx, query_idx, task_id = input
            # to record the op in t update step.
            query_preds, query_losses, query_attn_losses, query_accs = [], [], [], []

            # ==================================
            # REUSE       True        False
            # Not exist   Error       Create one
            # Existed     reuse       Error
            # ==================================
            # That's, to create variable, you must turn off reuse
            _, _, labels, _ = self.graphs[task_id]

            logits, support_attn_loss = self.gnn_forward(self.weights, task_id)
            support_pred = tf.gather(logits, support_idx)
            support_y = tf.gather(labels, support_idx)
            
            support_loss = tf.nn.softmax_cross_entropy_with_logits(logits=support_pred, labels=support_y)
            support_acc = tf.reshape(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(support_pred, dim=1), axis=1), tf.argmax(support_y, axis=1)), (1,))
            # compute gradients
            grads = tf.gradients(support_loss + self.beta * support_attn_loss, list(self.weights.values()))
            # grad and variable dict
            gvs = dict(zip(self.weights.keys(), grads))

            # theta_pi = theta - alpha * grads
            fast_weights = dict(zip(self.weights.keys(), [self.weights[key] - self.train_lr * tf.clip_by_norm(gvs[key], 10) for key in self.weights.keys()]))
            # use theta_pi to forward meta-test
            logits, query_attn_loss = self.gnn_forward(fast_weights, task_id)
            query_pred = tf.gather(logits, query_idx)
            query_y = tf.gather(labels, query_idx)
            # meta-test loss
            query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
            # record T0 pred and loss for meta-test
            query_preds.append(query_pred)
            query_losses.append(query_loss)
            query_attn_losses.append(query_attn_loss)
            query_accs.append(tf.reshape(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(query_pred, dim=1), axis=1), tf.argmax(query_y, axis=1)), (1,)))

            # continue to build T1-TK steps graph
            for _ in range(1, K):
                # T_k loss on meta-train
                # we need meta-train loss to fine-tune the task and meta-test loss to update theta
                logits, _support_attn_loss = self.gnn_forward(fast_weights, task_id)
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf.gather(logits, support_idx), labels=tf.gather(labels, support_idx))
                # compute gradients
                grads = tf.gradients(loss + self.beta * _support_attn_loss, list(fast_weights.values()))
                # compose grad and variable dict
                gvs = dict(zip(fast_weights.keys(), grads))
                # update theta_pi according to varibles
                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.train_lr * tf.clip_by_norm(gvs[key], 10)
                                         for key in fast_weights.keys()]))
                # forward on theta_pi
                logits, query_attn_loss = self.gnn_forward(fast_weights, task_id)
                query_pred = tf.gather(logits, query_idx)
                query_y = tf.gather(labels, query_idx)
                # we need accumulate all meta-test losses to update theta
                query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
                query_preds.append(query_pred)
                query_losses.append(query_loss)
                query_attn_losses.append(query_attn_loss)
                query_accs.append(tf.reshape(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(query_pred, dim=1), axis=1), tf.argmax(query_y, axis=1)), (1,)))

            # # compute every steps' accuracy on query set
            # for i in range(K):
            #     query_accs.append(tf.reshape(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(query_preds[i], dim=1), axis=1), tf.argmax(query_y, axis=1)), (1,)))
            # we just use the first step support op: support_pred & support_loss, but igonre these support op
            # at step 1:K-1.
            # however, we return all pred&loss&acc op at each time steps.

            # the first accuracy on support set measures the performance for general theta on training of this specific task
            # the final accuracy on query set measures the performance of fast adaption of the general theta
            # 
            result = [support_pred, support_loss, support_attn_loss, support_acc, query_preds, query_losses, query_attn_losses, query_accs]

            return result

        # return: [support_pred, support_loss, support_acc, query_preds, query_losses, query_accs]
        # out_dtype = [tf.float32, tf.float32, tf.float32, [tf.float32] * K, [tf.float32] * K, [tf.float32] * K]

        result = []
        for task_id in range(self.nb_task):
            result.append(meta_task((self.support_idxes[task_id], self.query_idxes[task_id], task_id)))

        # result = tf.map_fn(meta_task, elems=(support_idxes, query_idxes, self.graphs_indices),
        #                    dtype=out_dtype, parallel_iterations=meta_batchsz, name='map_fn')
        support_pred_tasks, support_loss_tasks, support_attn_loss_tasks, support_acc_tasks, query_preds_tasks, query_losses_tasks, query_attn_loss_tasks, query_accs_tasks = list(zip(*result))


        support_pred_tasks, support_loss_tasks, support_attn_loss_tasks, support_acc_tasks = list(map(lambda x:tf.concat(x,axis = 0), [support_pred_tasks, support_loss_tasks, support_attn_loss_tasks, support_acc_tasks])) # become 1d vector, results of tasks are merged

        query_attn_loss_tasks = [tf.concat(K_step_query_attn_loss, axis = -1) for K_step_query_attn_loss in zip(*query_attn_loss_tasks)]

        query_losses_tasks = [tf.concat(K_step_query_loss, axis = -1) for K_step_query_loss in zip(*query_losses_tasks)]

        query_accs_tasks = [tf.concat(K_step_query_acc, axis = -1) for K_step_query_acc in zip(*query_accs_tasks)]


        meta_batchsz = self.nb_task
        # average loss
        self.support_loss = support_loss = tf.reduce_sum(support_loss_tasks) / meta_batchsz
        # average attn loss
        self.support_attn_loss = support_attn_loss = tf.reduce_sum(support_attn_loss_tasks) / meta_batchsz
        # [avgloss_t1, avgloss_t2, ..., avgloss_K]
        self.query_losses = query_losses = [tf.reduce_sum(query_losses_tasks[j]) / meta_batchsz
                                                for j in range(K)]
        self.query_attn_loss = query_attn_loss = [tf.reduce_sum(query_attn_loss_tasks[j]) / meta_batchsz
                                                for j in range(K)]
        # average accuracy
        self.support_acc = support_acc = tf.reduce_sum(support_acc_tasks) / meta_batchsz
        # average accuracies
        self.query_accs = query_accs = [tf.reduce_sum(query_accs_tasks[j]) / meta_batchsz
                                                for j in range(K)]

        # meta-train optim
        self.optimizer = tf.train.AdamOptimizer(self.meta_lr, name='meta_optim')
        # meta-train gradients, query_losses[-1] is the accumulated loss across over tasks.
        gvs = self.optimizer.compute_gradients(self.query_losses[-1] + self.beta * self.query_attn_loss[-1], var_list=list(self.weights.values()))
        # meta-train grads clipping
        gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
        # update theta
        self.meta_op = self.optimizer.apply_gradients(gvs)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(tf.global_variables_initializer())



    def gnn_weights(self):
        # single head gat
        w_init = slim.xavier_initializer
        weights = {}
        previous_size = self.ndim
        with tf.variable_scope('MAML', reuse= tf.AUTO_REUSE):
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
            # weight_final = tf.get_variable(f"W_{len(self.hidden_sizes) + 1}", shape=[previous_size, self.nway],
            #                                 dtype=tf.float32,
            #                                 initializer=w_init())
            # bias_final = tf.get_variable(f"b_{len(self.hidden_sizes) + 1}", shape=[self.nway], dtype=tf.float32,
            #                                 initializer=w_init())

            # weights[f'w_fin'] = weight_final
            # weights[f'b_fin'] = bias_final
            return weights

    def gnn_forward(self, weights, task_id):
        adj_mat, hidden, _, ind = self.graphs[task_id]
        # perturb_attn = []

        attn_common_edge = []
        attn_ptb_edge = []

        if not task_id in self.attn_values:
            self.attn_values[task_id] = []

        for ix in range(len(self.hidden_sizes)):
            new_hidden = []
            for hd in range(self.head[ix]):
                w = weights[f'w{ix}_{hd}']
                b = weights[f'b{ix}_{hd}']
                a1 = weights[f'a1{ix}_{hd}']
                a2 = weights[f'a2{ix}_{hd}']
                
                _hidden = hidden @ w
                
                f_1 = _hidden @ a1
                f_2 = _hidden @ a2

                # transfer to sparse first
                f1 = adj_mat * f_1
                f2 = adj_mat * tf.transpose(f_2, [1,0])

                raw_attn = tf.sparse_add(f1, f2)
                raw_attn = tf.SparseTensor(indices=raw_attn.indices, 
                        values=tf.nn.leaky_relu(raw_attn.values), 
                        dense_shape=raw_attn.dense_shape)

                # attention scores
                attn = tf.sparse_softmax(raw_attn)


                # attn_on_ptb_edge = tf.reduce_sum(attn.values * ind.values)
                # attn_on_ptb_edge = tf.reshape(attn_on_ptb_edge, [-1])

                origin_attn_avg = tf.reduce_sum(raw_attn.values) / tf.reduce_sum(adj_mat.values)
                attn_common_edge.append(tf.reshape(origin_attn_avg, [-1]))

                if self.coef_drop != 0.0:
                    attn = tf.SparseTensor(indices=attn.indices,
                            values=tf.nn.dropout(attn.values, 1.0 - self.coef_drop),
                            dense_shape=attn.dense_shape)
                if self.in_drop != 0.0:
                    hidden = tf.nn.dropout(hidden, 1.0 - self.in_drop)

                _hidden = tf.sparse_tensor_dense_matmul(attn, _hidden) + b
                _hidden = tf.nn.relu(_hidden)
                
                new_hidden.append(_hidden)

                # compute attention scores on perturbed edges
                # !! must ensure indicator matrix has the same indices with adj_mat
                _f1 = ind * f_1
                _f2 = ind * tf.transpose(f_2, [1,0])

                raw_attn_on_added_edge = tf.sparse_add(_f1, _f2)
                raw_attn_on_added_edge = tf.SparseTensor(indices=raw_attn_on_added_edge.indices, 
                    values=tf.nn.leaky_relu(raw_attn_on_added_edge.values), 
                    dense_shape=raw_attn_on_added_edge.dense_shape)

                ptb_attn_avg = tf.reduce_sum(raw_attn_on_added_edge.values) / tf.reduce_sum(ind.values)
                attn_ptb_edge.append(tf.reshape(ptb_attn_avg, [-1]))
                # add penalty as the distance, puterbuted edges should have smaller attention scores
                # attn_loss = tf.math.maximum(0.0, ptb_attn_avg - origin_attn_avg)
                
                # debug
                # self.attn_values[task_id].extend([raw_attn.values, raw_attn_on_added_edge.values])

                # attn_loss = tf.reshape(attn_loss, [-1])


                # perturb_attn.append(attn_loss)

            hidden = tf.concat(new_hidden, axis=-1)
        
        # compute average attention loss on all attentions, there are sum(self.head) scores in total
        attn_common_edge_avg = tf.reduce_mean(tf.concat(attn_common_edge, axis = -1))
        attn_ptb_edge_avg = tf.reduce_mean(tf.concat(attn_ptb_edge, axis = -1))
        attn_score = tf.math.maximum(-self.dist, attn_ptb_edge_avg - attn_common_edge_avg)
        attn_score = tf.reshape(attn_score, [-1])

        logits = tf.add_n(new_hidden) / self.head[-1]

        # debug
        # self.attn_values.append((raw_attn, attn, raw_attn_on_added_edge, ind, old_hidden, attn_score, ptb_attn_avg, origin_attn_avg, perturb_attn))


        return logits, attn_score

    def train(self, setting, max_iter = 1500, val_rate = 0.3, is_train = True):
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        sess = self.sess
        model = self

        self.model_path = model_path = os.path.join('models', setting)
        if not os.path.isdir(model_path):
            os.mkdir(model_path)


        if os.path.exists(os.path.join(model_path, 'checkpoint')):
            # alway load ckpt both train and test.
            model_file = tf.train.latest_checkpoint(model_path)
            print("Restoring model weights from ", model_file)
            saver.restore(sess, model_file)

        if not is_train:
            return
        # reserve some nodes for validation
        graph_node_for_train = []
        graph_node_reserved = []
        for graph in self.raw_graphs:
            nb_node = graph[0].shape[0]
            nodes = np.random.permutation(nb_node)
            split = int(nb_node * (1.0 - val_rate))
            graph_node_for_train.append(nodes[:split])
            graph_node_reserved.append(nodes[split:])

        feed = {}
        val_feed = {}

        prelosses, postlosses, preaccs, postaccs = [], [], [], []
        best_acc = 0
        val_go_worse_cnt = 0

        # train for meta_iteartion epoches
        for iteration in range(max_iter):
            # this is the main op
            ops = [self.meta_op]

            # add summary and print op
            if iteration % 10 == 0:
                ops.extend([self.query_losses[0], self.query_losses[-1],
                self.query_accs[0], self.query_accs[-1]])
            
            # sample train and test
            for task_id in range(self.nb_task):
                nodes = graph_node_for_train[task_id]
                split = nodes.shape[0] // 2
                feed[self.support_idxes[task_id]] = nodes[:split]
                feed[self.query_idxes[task_id]] = nodes[split:]

            # run all ops
            result = sess.run(ops, feed_dict= feed)

            # summary
            if iteration % 10 == 0:
                prelosses.append(result[1])
                postlosses.append(result[2])
                preaccs.append(result[3])
                postaccs.append(result[4])

                print(iteration, '\tloss:', np.mean(prelosses), '=>', np.mean(postlosses),
                    '\t\tacc:', np.mean(preaccs), '=>', np.mean(postaccs))
                prelosses, postlosses, preaccs, postaccs = [], [], [], []

            # evaluation
            if iteration % 10 == 0:
                # DO NOT write as a = b = [], in that case a=b
                # DO NOT use train variable as we have train func already.
                acc1s, acc2s = [], []
                for _ in range(1):
                    # sample train and test, but from the reserved set
                    for task_id in range(self.nb_task):
                        nodes = graph_node_reserved[task_id]
                        split = nodes.shape[0] // 2
                        val_feed[self.support_idxes[task_id]] = nodes[:split]
                        val_feed[self.query_idxes[task_id]] = nodes[split:]
                    
                    acc1, acc2 = sess.run([model.query_accs[0],
                                            model.query_accs[-1]], feed_dict= val_feed)
                    acc1s.append(acc1)
                    acc2s.append(acc2)

                acc = np.mean(acc2s)
                print('>>>>\tValidation accs: ', np.mean(acc1s), acc, 'best:', best_acc, '\t<<<<')

                if acc - best_acc > 0.005 and iteration != 0:
                    saver.save(sess, os.path.join(model_path, 'robust.gcn'))
                    best_acc = acc
                    print('saved into {}:'.format(setting), acc)
                if acc < best_acc:
                    val_go_worse_cnt+=1
        
                if val_go_worse_cnt > 10:
                    # stop training
                    break



    def finetune(self, graph, train_idx, val_idx, test_idx, n_iters=200):
        sess = self.sess
        nb_task = self.nb_task
        self.init_graphs([graph,])
        self.nb_task = nb_task + 1
        weights = self.gnn_weights()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        if os.path.exists(os.path.join(self.model_path, 'checkpoint')):
            # alway load ckpt both train and test.
            model_file = tf.train.latest_checkpoint(self.model_path)
            print("Restoring model weights from ", model_file)
            saver.restore(sess, model_file)

        adj_norm, hidden, Y, _ = self.graphs[-1]

        logits, _ = self.gnn_forward(weights, -1)

        # losses
        loss, val_loss, test_loss = map(lambda idx:tf.nn.softmax_cross_entropy_with_logits(labels=tf.gather(Y, idx), logits=tf.gather(logits, idx)), [train_idx, val_idx, test_idx])

        # accs
        acc, val_acc, test_acc = map(lambda idx:tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(tf.gather(logits, idx), dim=1), axis=1), tf.argmax(tf.gather(Y, idx), axis=1)), [train_idx, val_idx, test_idx])


        gvs = self.optimizer.compute_gradients(loss, var_list=list(self.weights.values()))
        gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
        train_op = self.optimizer.apply_gradients(gvs)

        self.finetune_model_path = os.path.join(self.model_path, 'finetune')
        if not os.path.isdir(self.finetune_model_path):
            os.mkdir(self.finetune_model_path)

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        test_losses = []
        test_accs = []

        best_acc = 0

        evaluation_results = []

        ops = [loss, acc, val_loss, val_acc, test_loss, test_acc, train_op]
        print('loss, acc, val_loss, val_acc, test_loss, test_acc:')
        for iter in range(n_iters):
            result = sess.run(ops)
            result[0] = np.sum(result[0])
            train_losses.append(result[0])
            train_accs.append(result[1])
            result[2] = np.sum(result[2])
            val_losses.append(result[2])
            val_accs.append(result[3])
            result[4] = np.sum(result[4])
            test_losses.append(result[4])
            test_accs.append(result[5])

            # set_trace()
            
            if val_accs[-1] > best_acc:
                print('{}: {}'.format(iter, ', '.join(["{:.2f}".format(x) for x in result[:6]])))

                best_acc = val_accs[-1]
                saver.save(sess, os.path.join(self.finetune_model_path, 'robust.gcn.finetune'))
                print('saved into ckpt:', best_acc)
                evaluation_results = [test_losses[-1], test_accs[-1]]

            if np.argmax(val_accs) < len(val_accs) - 100:
                pass
                # break
        return evaluation_results
