# -*- coding: utf-8 -*-
# /usr/bin/python3

from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import pickle
from scipy import special


def normalize(inputs, epsilon=1e-8, scope="layer_normalization", reuse=None):
    #layer normalize
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
        return outputs


def embedding(inputs, vocab_size, num_units, zero_pad=False, scale=True, scope="embedding", reuse=None,
              fine_tuning=False, pretrain_emb=None):
    with tf.variable_scope(scope, reuse=reuse):
        if fine_tuning:
            pretrain_emb = tf.concat([pretrain_emb, tf.zeros([1, num_units], dtype=tf.float32)], axis=0)
            lookup_table = tf.get_variable('lookup_table',
                                           dtype=tf.float32,
                                           # shape=[vocab_size + 1, num_units],
                                           initializer=pretrain_emb,)
        else:
            lookup_table = tf.get_variable('lookup_table',
                                           dtype=tf.float32,
                                           shape=[vocab_size + 1, num_units],
                                           initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    return outputs


def periodic_embedding(inputs, aggregate_method='average', scope='period_embedding', seqlen=None, num_unit=None,
                       scope_embed=None, reuse=None):
    """
    split the periodic token ids and aggregate the embeddings
    :param inputs: periodic string tensor
    :param aggregate_method:
    :param scope:
    :return:
    """
    assert aggregate_method in ['average', 'max']
    with tf.variable_scope(scope_embed, reuse=True):
        lookup_table = tf.get_variable('lookup_table')
        with tf.variable_scope(scope, reuse=reuse):
            ragged_input = tf.strings.split(inputs, sep='|', result_type='RaggedTensor')
            nested_row_lens = ragged_input.nested_row_lengths()
            tensor_input = ragged_input.to_tensor(default_value='-1')
            tensor_input = tf.strings.to_number(tensor_input, out_type=tf.int32)
            ragged_idx = tf.compat.v2.where(tf.not_equal(tensor_input, -1))
            ragged_input = tf.RaggedTensor.from_nested_row_lengths(tf.gather_nd(tensor_input, ragged_idx),
                                                                   nested_row_lens)
            ragged_embeddings = tf.ragged.map_flat_values(
                tf.nn.embedding_lookup, lookup_table, ragged_input
            )
            if aggregate_method == 'average':
                outputs = tf.reduce_mean(ragged_embeddings, axis=2)
            if aggregate_method == 'max':
                outputs = tf.reduce_max(ragged_embeddings, axis=2)
            outputs = tf.cast(outputs.to_tensor(), tf.float32)
            outputs = tf.reshape(outputs, [-1, seqlen, num_unit])
    return outputs  # [batch_size, period_len, hidden_unit]


def periodic_embedding_v2(inputs, scope='period_embedding', scope_embed=None, scope_intention_emb=None,
                          reuse=None):
    """
    conv1d align to generate the high-level embeddings
    :param inputs: periodic string tensor
    :param scope:
    :param scope_embed:
    :param scope_intention_emb:
    :param reuse:
    :return:
    """
    inputs = tf.cast(inputs, tf.int32)
    with tf.variable_scope(scope_embed, reuse=True):
        lookup_table = tf.get_variable('lookup_table')
        periodic_behavior_embed = tf.nn.embedding_lookup(lookup_table, inputs)
    periodic_behavior_embed = tf.reshape(periodic_behavior_embed, [-1, hp.maxlen, hp.hidden_units])
    intention_emb = convolution_align(periodic_behavior_embed, scope=scope_intention_emb, reuse=tf.AUTO_REUSE)
    with tf.variable_scope(scope, reuse=reuse):
        outputs = intention_emb
    return outputs  # [batch_size, period_len, hidden_unit]


def periodic_embedding_v3(inputs, scope='period_embedding', scope_embed=None, scope_intention_emb=None,
                          reuse=None):
    """
    conv1d align to generate the high-level embeddings
    :param inputs: periodic string tensor
    :param scope:
    :param scope_embed:
    :param scope_intention_emb:
    :param reuse:
    :return:
    """
    inputs = tf.cast(inputs, tf.int32)
    with tf.variable_scope(scope_embed, reuse=True):
        lookup_table = tf.get_variable('lookup_table')
        periodic_behavior_embed = tf.nn.embedding_lookup(lookup_table, inputs)
    periodic_behavior_embed = tf.reshape(periodic_behavior_embed, [-1, hp.maxlen, hp.hidden_units])
    intention_emb = gated_convolution_align(periodic_behavior_embed, scope=scope_intention_emb, reuse=tf.AUTO_REUSE)
    with tf.variable_scope(scope, reuse=reuse):
        outputs = intention_emb
    return outputs  # [batch_size, period_len, hidden_unit]


def positional_encoding(inputs, num_units, scale=True, scope="positional_encoding", reuse=None):
    inputs = tf.cast(inputs, tf.float32)
    N = tf.shape(inputs)[0]
    T = inputs.get_shape().as_list()[1]

    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc, np.float32)

        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5

    return outputs


def temporal_log_time_positional_encoding(inputs, num_units, time_stamp, scale=True,
                                          scope="temporal_log_positional_encoding", reuse=None):
    inputs = tf.cast(inputs, tf.float32)
    N = tf.shape(inputs)[0]
    T = inputs.get_shape().as_list()[1]
    st = tf.tile(tf.expand_dims(time_stamp[:, 0], 1), [1, T])  # [batch_size, max_len]
    ti = time_stamp - st  # [batch_size, max_len]
    ti = tf.math.log(ti+1)  # natural logarithm to deal with the skewed dist.
    ti = tf.tile(tf.expand_dims(ti, 2), [1, 1, num_units]) # [batch_size, max_len, num_units]
    ti = tf.cast(ti, tf.float32)

    with tf.variable_scope(scope, reuse=reuse):
        # First part of the PE function: sin and cos argument
        range_tensor = tf.range(num_units)
        mod_tensor = tf.mod(range_tensor, 2*tf.ones_like(range_tensor))
        expnt = range_tensor - mod_tensor
        expnt = tf.cast(expnt/num_units, tf.float32)
        base = tf.math.pow(20.0 * tf.ones_like(expnt, dtype=tf.float32), expnt)
        base = tf.cast(base, tf.float32)
        base = tf.expand_dims(tf.expand_dims(base, 0), 0)
        base = tf.tile(base, [N, T, 1])
        position_enc = ti / base

        # # Second part, apply the cosine to even columns and sin to odds.
        pos_sin = tf.sin(position_enc)
        pos_cos = tf.cos(position_enc)
        pos_ind = mod_tensor
        pos_ind = tf.tile(tf.expand_dims(pos_ind, 0), [T, 1])
        pos_ind = tf.tile(tf.expand_dims(pos_ind, 0), [N, 1, 1])
        pos_sin_ind = tf.cast(1-pos_ind, dtype=tf.float32)
        pos_cos_ind = tf.cast(pos_ind, dtype=tf.float32)
        position_enc = tf.multiply(pos_sin, pos_sin_ind) + tf.multiply(pos_cos, pos_cos_ind)

        outputs = position_enc

        if scale:
            outputs = outputs * num_units ** 0.5

        return outputs


def time2vec(inputs, num_units=None, periodic_fun='sin', scope='time2vec', reuse=None):
    assert periodic_fun in ['sin', 'cos']
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.cast(inputs, tf.float32)
        N = tf.shape(inputs)[0]
        T = inputs.get_shape().as_list()[1]
        st = tf.tile(tf.expand_dims(inputs[:, 0], 1), [1, T])  # [batch_size, max_len]
        ti = inputs - st  # [batch_size, max_len]
        ti = tf.math.log(ti + 1)  # natural logarithm to deal with the skewed dist.
        freq = tf.get_variable('freq',
                               dtype=tf.float32,
                               shape=[1, num_units],
                               initializer=tf.contrib.layers.xavier_initializer(),
                               )
        shift = tf.get_variable('shift',
                                dtype=tf.float32,
                                shape=[1, num_units],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                )

        # linear mapping
        F = tf.tile(tf.expand_dims(freq, 0), [N, 1, 1])  # [N, 1, C]
        S = tf.tile(tf.expand_dims(shift, 0), [N, T, 1])  # [N, T, C]
        ti = tf.expand_dims(ti, 2)               # [N, T, 1]
        linear_outputs = tf.matmul(ti, F) + S    # [N, T, C]

        # periodic activation
        periodic_outputs = tf.transpose(linear_outputs, [2, 0, 1])  # [C, N, T]
        if periodic_fun == 'sin':
            periodic_outputs = tf.concat([
                periodic_outputs[0:1, :, :],
                tf.sin(periodic_outputs[1:, :, :])
            ], axis=0)
        if periodic_fun == 'cos':
            periodic_outputs = tf.concat([
                periodic_outputs[0:1, :, :],
                tf.cos(periodic_outputs[1:, :, :])
            ], axis=0)
        outputs = tf.transpose(periodic_outputs, [1, 2, 0])  # [N, T, C]

        return outputs


def feedforward(inputs, num_units=None, scope="multihead_attention", reuse=None):
    assert len(num_units) == 2
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs  # (batch_size, maxlen, hidden_units)


def feedforward_sparse(inputs, num_units=[4 * hp.hidden_units, hp.hidden_units],
                       scope="multihead_attention_feedforward", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.elu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
    return outputs    #(batch_size, maxlen, hidden_units)


def label_smoothing(inputs, epsilon=0.1):
    inputs = tf.cast(inputs, tf.float32)
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)


def cross_entropy(logits, labels, class_num, isSmoothing = False):
    '''compute loss of cross_entropy'''
    if isSmoothing: # label smoothing
        one_hot_label = tf.one_hot(tf.cast(labels, tf.int32), depth=class_num)
        smoothed_label = label_smoothing(one_hot_label)
        loss_entropy_event = tf.nn.softmax_cross_entropy_with_logits_v2(labels=smoothed_label, logits=logits)
    else:
        loss_entropy_event = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.int32))
    # # wighted_loss
    # weight = tf.cast(tf.gather(self.weight_per_class, tf.cast(batch_target, tf.int32)), loss_entropy_event.dtype)
    # loss_entropy_event = tf.multiply(loss_entropy_event, weight)
    cost = tf.reduce_mean(loss_entropy_event)
    return cost


def cross_entropy_np(logits, labels, class_num):
    '''compute loss of cross_entropy'''
    prob = special.softmax(logits, -1) + 1e-12
    one_hot_label = np.zeros_like(prob, dtype=float)
    for i in range(prob.shape[0]):
        one_hot_label[i, int(labels[i])] = 1
    loss_entropy = np.multiply(one_hot_label, -np.log(prob))
    cost = np.mean(np.sum(loss_entropy, axis=-1))
    return cost


def cross_entropy_prob(prob, labels, class_num, isSmoothing = False):
    '''compute loss of cross_entropy'''
    prob = tf.clip_by_value(prob, 1e-3, 1)
    one_hot_label = tf.one_hot(tf.cast(labels, tf.int32), depth=class_num)
    if isSmoothing:  # label smoothing
        smoothed_label = label_smoothing(one_hot_label)
        loss_entropy = tf.multiply(smoothed_label, -tf.log(prob))
    else:
        loss_entropy = tf.multiply(one_hot_label, -tf.log(prob))
    return loss_entropy


def cross_entropy_loss(logits, labels, class_num, isSmoothing = False, one_hot_label=None):
    '''compute non-reduced loss of cross_entropy'''
    prob = tf.nn.softmax(logits)
    prob = tf.clip_by_value(prob, 1e-3, 1)
    if one_hot_label is None:
        one_hot_label = tf.one_hot(tf.cast(labels, tf.int32), depth=class_num)
    if isSmoothing:  # label smoothing
        one_hot_label = label_smoothing(one_hot_label)
    loss_entropy = tf.multiply(one_hot_label, -tf.log(prob))
    return loss_entropy


def cross_entropy_hierarchical(sub_logits, main_logits, sub_label, main_class, sub_class,
                               index_array=None, index_mat=None, isSmoothing=False, main_label=None):
    '''compute loss of hierarachical cross_entropy'''
    sub_loss = cross_entropy_loss(logits=sub_logits, labels=sub_label, class_num=sub_class,
                                  isSmoothing=isSmoothing)
    one_hot_main_label = tf.one_hot(tf.cast(main_label, tf.int32), depth=main_class)
    one_hot_main_label = tf.transpose(one_hot_main_label, [1, 0])
    one_hot_main_label = tf.gather(one_hot_main_label, index_array)
    one_hot_main_label = tf.transpose(one_hot_main_label, [1, 0])

    mask_mat = tf.where(tf.equal(one_hot_main_label, 1), tf.ones_like(sub_loss), 2*tf.ones_like(sub_loss))
    sub_loss = tf.multiply(sub_loss, mask_mat)
    cost = tf.reduce_sum(sub_loss, axis=1)
    cost = tf.reduce_mean(cost)
    return cost


def hirachical_violation(main_logit, sub_logit, index_array=None, _lambda=0.1):
    main_prob = tf.nn.softmax(logits=main_logit)
    # main_class_softmax_trans = tf.transpose(main_prob, [1, 0])  # [hp.output_unit, batch_size]
    # main_class_softmax_trans = tf.gather(main_class_softmax_trans, index_array)  # [hp.output_sub_unit, batch_size]
    # main_class_softmax_map = tf.transpose(main_class_softmax_trans, [1, 0])  # [batch_size, hp.output_sub_unit]
    # hv = tf.maximum(tf.subtract(sub_prob, main_class_softmax_map), 0)
    # hv = _lambda * tf.square(hv)
    # hv = tf.reduce_sum(hv, axis=1)
    idx_mat = [index_array == i for i in range(hp.output_unit)]
    idx_mat = np.asarray(idx_mat, dtype=int)
    idx_mat = tf.convert_to_tensor(idx_mat, dtype=tf.float32)
    sub_main_logits = tf.matmul(sub_logit, tf.transpose(idx_mat, [1, 0]))
    cos_sim = tf.multiply(tf.nn.l2_normalize(main_logit, axis=1), tf.nn.l2_normalize(sub_main_logits, axis=1))
    hv = tf.reduce_sum(-cos_sim, axis=1)
    cost = tf.reduce_mean(hv)
    return cost


def hirachical_violation_cost(sub_logit, main_label, main_class, sub_class, index_array=None, isSmoothing=False):
    # idx_mat = [index_array == i for i in range(main_class)]
    # idx_mat = np.asarray(idx_mat, dtype=int)
    # idx_mat = tf.convert_to_tensor(idx_mat, dtype=tf.float32)
    # main_logits = tf.matmul(sub_logit, tf.transpose(idx_mat, [1, 0]))
    # cost = cross_entropy(logits=sub_logit, labels=one_hot_sub_label,
    #                      class_num=sub_class, isSmoothing=isSmoothing)
    one_hot_main_label = tf.one_hot(tf.cast(main_label, tf.int32), depth=main_class)
    if isSmoothing:
        one_hot_main_label = label_smoothing(one_hot_main_label)
    one_hot_main_label = tf.transpose(one_hot_main_label, [1, 0])
    one_hot_sub_label = tf.gather(one_hot_main_label, index_array)
    one_hot_sub_label = tf.nn.softmax(tf.transpose(one_hot_sub_label, [1, 0]))
    sub_prob = tf.clip_by_value(tf.nn.softmax(sub_logit), 1e-3, 1)
    loss_entropy = tf.multiply(one_hot_sub_label, -tf.log(sub_prob))
    cost = tf.reduce_sum(loss_entropy, axis=-1)
    cost = tf.reduce_mean(cost)
    return cost


def time_loss(ts, time_list, time_label):
    '''
    compute time loss for time prediction
    :param ts:  [batch_size]
    :param time_list: [batch_size, maxlen]
    :param time_label: [batch_size]
    :return:
    '''
    ts = tf.cast(ts, tf.float32)
    time_list = tf.cast(time_list, tf.float32)
    time_label = tf.cast(time_label, tf.float32)
    base = time_list[:, 0]  # [batch_size]
    time_cost = tf.square(time_label-base+ts) / (hp.avg_time_gap/2)
    time_cost = tf.reduce_mean(time_cost)
    return time_cost


def focal_loss(logits, labels, class_num, alpha=1.0, gamma=2, isSmoothing=False):
    '''Focal loss: to balance the imbalanced multi-class classification'''
    labels = tf.one_hot(tf.cast(labels, tf.int32), depth=class_num)
    if isSmoothing:
        labels = label_smoothing(labels)
    y_pred = tf.nn.softmax(logits=logits) + 1e-10
    ce = tf.multiply(labels, -tf.log(y_pred))
    weight = alpha * tf.pow(tf.subtract(1., y_pred), gamma)
    fl = tf.multiply(weight, ce)
    cost = tf.reduce_max(fl, axis=1)
    # weighted_label = tf.multiply(labels, tf.pow(tf.subtract(1., y_pred), gamma))
    # ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=weighted_label, logits=logits)
    # fl = alpha * ce
    # cost = tf.reduce_max(fl, axis=1)
    cost = tf.reduce_mean(cost)
    return cost


def relative_multihead_attention(queries, keys, pos_enc=None, u_vec=None, v_vec=None,
                                 num_units=None, num_heads=4, causality=False,
                                 scope="relative_multihead_attention", reuse=None):
    """
    Compute the relative attention to fuse the temporal position information
    Inspired by relative position encoding in Transformer-XL
    :param queries: the input embedding of the token seq
    :param keys: the input embedding of the token seq
    :param u_vec:
    :param v_vec:
    :param num_units:
    :param num_heads:
    :param causality:
    :param scope:
    :param reuse:
    :param pos_enc: the (temporal) position encoding
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        queries = tf.cast(queries, tf.float32)
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        N = tf.shape(queries)[0]
        T = tf.shape(queries)[-2]
        C = num_units

        u_vec = tf.tile(u_vec, [T, 1])  # [max_len, num_units]
        v_vec = tf.tile(v_vec, [T, 1])  # [max_len, num_nuits]
        u_vec = tf.tile(tf.expand_dims(u_vec, 0), [N, 1, 1])  # [batch_size, max_len, num_units]
        v_vec = tf.tile(tf.expand_dims(v_vec, 0), [N, 1, 1])  # [batch_size, max_len, num_units]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        PE = tf.layers.dense(pos_enc,
                             num_units,
                             activation=tf.nn.relu,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
                             # [batch_size, max_len, num_units]

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        PE_ = tf.concat(tf.split(PE, num_heads, axis=2), axis=0)  # [num_heads*batch_size, max_len, num_units/num_heads]
        u_vec_ = tf.concat(tf.split(u_vec, num_heads, axis=2), axis=0)  # [num_heads*batch_size, max_len, num_units/num_heads]
        v_vec_ = tf.concat(tf.split(v_vec, num_heads, axis=2), axis=0)  # [num_heads*batch_size, max_len, num_units/num_heads]

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Relative temporal attention
        outputs += tf.matmul(Q_, tf.transpose(PE_, [0, 2, 1]))
        outputs += tf.matmul(u_vec_, tf.transpose(K_, [0, 2, 1]))
        outputs += tf.matmul(v_vec_, tf.transpose(PE_, [0, 2, 1]))

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys + pos_enc, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            # Causality=True #########
            # tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries+pos_enc, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        outputs2 = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs2 = tf.concat(tf.split(outputs2, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # post attention weighting
        outputs2 = tf.layers.dense(outputs2, units=num_units, use_bias=False)

    return outputs2


def multihead_attention(queries, keys, num_units=None, num_heads=4, causality=False,
                        scope="multihead_attention", reuse=None):
    """
    :param queries: the input embedding of the token seq
    :param keys: the input embedding of the token seq
    :param num_units:
    :param num_heads:
    :param causality:
    :param scope:
    :param reuse:
    :param pos_enc: the (temporal) position encoding
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        queries = tf.cast(queries, tf.float32)
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        N = tf.shape(queries)[0]
        T = tf.shape(queries)[-2]
        C = num_units

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)


        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            # Causality=True #########
            # tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        outputs2 = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs2 = tf.concat(tf.split(outputs2, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # post attention weighting
        outputs2 = tf.layers.dense(outputs2, units=num_units, use_bias=False)

    return outputs2


def fusion_attention(inputs, scope='fusion_attention', reuse=None, fusion_method='concat'):
    """
    fusion the input sequence with vanilla attention
    :param inputs:
    :param scope:
    :param reuse:
    :param fusion_method:
    :return:
    """
    assert fusion_method in ['concat', 'add']
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.cast(inputs, tf.float32)
        T = inputs.get_shape().as_list()[1]
        C = inputs.get_shape().as_list()[2]
        inputs = tf.transpose(inputs, [1, 0, 2])
        final = tf.gather(inputs, [T-1])  # [1, N, C]
        query = tf.tile(final, [T-1, 1, 1])
        keys = tf.gather(inputs, tf.range(T-1))  # [T-1, N, C]

        # vanilla attention
        alignment = tf.layers.dense(tf.concat([query, keys], axis=2), units=C)  # [T-1, N, C]
        attention = tf.nn.softmax(alignment, axis=0)  # [T-1, N, C]
        outputs = tf.reduce_sum(tf.multiply(attention, keys), axis=0)  # [N, C]
        if fusion_method == 'concat':
            outputs = tf.concat([tf.squeeze(final, axis=0), outputs], axis=1)  # [N, 2*C]
        if fusion_method == 'add':
            outputs = tf.layers.dense(outputs, units=C, use_bias=False)
            final = tf.layers.dense(tf.squeeze(final, axis=0), units=C, use_bias=False)
            outputs = tf.nn.relu(tf.add(outputs, final))
    return outputs


def temporal_conv1d_fusion(inputs, scope='temporal_conv1d_fusion', reuse=None):
    """
    fusion the input sequence with vanilla attention
    :param inputs:
    :param scope:
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.cast(inputs, tf.float32)  # [N, T, C]
        T = inputs.get_shape().as_list()[1]
        C = inputs.get_shape().as_list()[2]

        kernel = 3
        filters = C
        outputs = tf.layers.conv1d(inputs, filters=filters, kernel_size=kernel)  # [N, 1, C]
        outputs = normalize(outputs)
    return outputs


def fuse_markov_table(train_batch, logits, markov_table, is_reweight=False):
    '''
    fuse the info of markov table to eliminate/reweight the logits
    :param logits: the output logits of each class
    :param markov_table: the markov_table from training set
    :return:
    '''
    input_map_ids = np.asarray(train_batch, dtype=int)
    last_map_ids = np.squeeze(input_map_ids[:,-1])
    if not is_reweight:
        markov_table = markov_table > 0
    else:
        markov_table = np.log10(markov_table+1)
    logits = logits * markov_table[last_map_ids, :]
    return logits


def sub_class_logit(inputs, main_class_logit, is_fuse_main_logit=False,
                    index_array=None):
    inputs = tf.cast(inputs, tf.float32)  # [batch_size, ?]
    # inputs = tf.concat([inputs, main_class_logit], axis=1)
    outputs = tf.layers.dense(inputs, units=hp.output_sub_unit, use_bias=False)
    if is_fuse_main_logit:
        main_class_logit = tf.cast(main_class_logit, tf.float32)  # [batch_size, ?]
        main_output = tf.argmax(main_class_logit, axis=1)  # [batch_size]
        # main class masking
        idx_mat = [index_array == i for i in range(hp.output_unit)]  # [main_class, sub_class]
        idx_mat = np.asarray(idx_mat, dtype=int)
        idx_mat = tf.convert_to_tensor(idx_mat, dtype=tf.float32)
        mask_mat = tf.gather(idx_mat, main_output)  # [batch_size, sub_class]
        paddings = tf.ones_like(mask_mat) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(mask_mat, 1.0), outputs, paddings)
        # main_pred_emb = get_embedding(main_output, scope='enc_embed')
        # inputs = tf.concat([inputs, main_pred_emb], axis=1)
    # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
    return outputs


def construct_mapping():
    '''
    map main type to sub type
    :return: the index of main type in sub type
    e.g., [0, 1, 1, 1, 2, 2, ...] means the first position corresponds to main_type 0 and the subsequent three positions
    all belong to main_type 1
    '''
    map_id_table_file = hp.FILE_PATH + 'map_id_mapping_table.csv'
    map_id_coordinate_table_file = hp.FILE_PATH + 'map_id_coordinate_mapping_table.csv'
    map_id_dic = {}
    map_id_coordinate_dic = {}
    with open(map_id_table_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            strs = line.split(',')
            map_id_dic[int(strs[0])] = int(strs[1])
    map_id_dic[0] = 0
    with open(map_id_coordinate_table_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            strs = line.split(',')
            map_id_coordinate_dic[int(strs[1])] = int(strs[0])
    map_id_array = np.vectorize(map_id_coordinate_dic.__getitem__)(np.arange(len(map_id_coordinate_dic)))
    map_id_array = np.asarray(map_id_array/100, dtype=int)
    index_array = np.vectorize(map_id_dic.__getitem__)(map_id_array)
    # idx_mat = [index_array == i for i in range(hp.output_unit)]
    # idx_mat = np.asarray(idx_mat, dtype=int)
    return index_array  #, idx_mat


def get_pretrain_emb():
    """
    get the pretrained map_id embedding
    :return: embedding table
    """
    emb_pkl_file = hp.FILE_PATH + 'map_id_w2v_64.pkl'
    with open(emb_pkl_file, 'rb') as f:
        map_id_emb = pickle.load(f)
    return map_id_emb


def meta_multihead_attention(meta, inputs, num_units=None, num_heads=hp.num_heads, scope="portrait_multihead_attention",
                             reuse=None):
    # meta ==> query
    # inputs ==>  keys
    # inputs ==>  values
    with tf.variable_scope(scope, reuse=reuse):
        keys = tf.cast(inputs, tf.float32)  # [batch_size, max_len, hidden_unit]
        values = keys
        queries = tf.cast(meta, tf.float32)  # [batch_size, portrait_size]
        shapes = tf.shape(keys)
        num_units = shapes[-1] if num_units is None else num_units
        num_seq = shapes[-2]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu,)  # (N, C)
        # Q = tf.tile(tf.expand_dims(Q, 1), [1, num_seq, 1])  # (N, T_q, C)
        Q = tf.expand_dims(Q, 1)  # (N, 1, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(values, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, 1, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, 1, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, 1, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, 1, T_k)
        outputs = tf.transpose(outputs, [0, 2, 1])  # (h*N, T_k, 1)
        outputs = tf.tile(outputs, [1, 1, K_.get_shape().as_list()[-1]])  # (h*N, T_k, C/h)

        outputs2 = tf.multiply(outputs, V_)  # ( h*N, T_k, C/h)

        # Restore shape
        outputs2 = tf.concat(tf.split(outputs2, num_heads, axis=0), axis=2)  # (N, T_k, C)

        # post-attention weighting
        outputs2 = tf.layers.dense(outputs2, num_units, use_bias=False)

    return outputs2


def portrait_multihead_embedding(portrait, inputs, num_units=None, num_heads=hp.num_heads, dropout_rate=0,
                                 is_training=True, scope="portrait_multihead_embedding", reuse=None):
    """
    inspired by [ICLR2017] a structured self-attentive sentence embedding and
    [EMNLP2018]  A Hierarchical Neural Attention-based Text Classifier
    :param portrait: the protrait vector   [batch_size, portrait_vec_size]
    :param inputs: the inputs  [batch_size, maxlen, num_unit]
    :param num_units:
    :param num_heads:
    :param dropout_rate:
    :param is_training:
    :param scope:
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.cast(inputs, tf.float32)
        shapes = inputs.get_shape().as_list()
        N = shapes[0]
        T = shapes[1]
        C = shapes[2]

        # concat portrait vector
        portrait = tf.tile(tf.expand_dims(portrait, 1), [1, T, 1])
        inputs = tf.concat([inputs, portrait], axis=2)

        # attention calculation
        attn1 = tf.layers.dense(inputs, units=hp.embedding_attention_size,
                                use_bias=False, activation=tf.nn.tanh)  # [batch_size, maxlen, embed_attn_size]
        attn2 = tf.layers.dense(attn1, units=num_heads,
                                use_bias=False, activation=None)  # [batch_size, maxlen, num_heads]
        attn = tf.nn.softmax(tf.transpose(attn2, [0, 2, 1]))  # [batch_size, num_heads, maxlen]

        # embedding generation
        seq_emb = tf.matmul(attn, inputs)  # [batch_size, num_heads, num_unit + portrait_vec_size]

        # normalize
        seq_emb = normalize(seq_emb)

        return seq_emb, attn


def sub_class_prob(inputs, main_class_logit, is_training=True, dropout_rate=hp.dropout_rate, is_fuse_main_logit=False,
                    index_array=None, beta=0.5):
    inputs = tf.cast(inputs, tf.float32)  # [batch_size, ?]
    main_class_logit = tf.cast(main_class_logit, tf.float32)  # [batch_size, ?]
    # inputs = tf.concat([inputs, main_class_logit], axis=1)
    outputs = tf.layers.dense(inputs, units=hp.output_sub_unit)
    if is_fuse_main_logit:
        main_class_softmax = tf.nn.softmax(logits=main_class_logit)
        # index_array = construct_mapping()  # [hp.output_sub_unit]
        main_class_softmax_trans = tf.transpose(main_class_softmax, [1, 0])  # [hp.output_unit, batch_size]
        main_class_softmax_trans = tf.gather(main_class_softmax_trans, index_array)  # [hp.output_sub_unit, batch_size]
        main_class_softmax_map = tf.transpose(main_class_softmax_trans, [1, 0])  # [batch_size, hp.output_sub_unit]
        outputs = beta * main_class_softmax_map + (1 - beta) * tf.nn.softmax(logits=outputs)
        # outputs = tf.multiply(outputs, main_class_softmax_map)
    # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
    return outputs


def disagreement_regularization(inputs, num_heads = hp.num_heads):
    inputs = tf.cast(inputs, tf.float32)  # (h*N, T_q, C/h)
    outputs = tf.zeros([hp.batch_size, hp.maxlen])
    for i in range(num_heads):
        i_index = tf.range(i, hp.batch_size * num_heads, num_heads)
        for j in range(num_heads):
            j_index = tf.range(i, hp.batch_size * num_heads, num_heads)
            tensor_i = tf.gather(inputs, i_index)  # (N, T_q, C/h)
            tensor_j = tf.gather(inputs, j_index)  # (N, T_q, C/h)
            outputs += tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(tensor_i, dim=2),
                                                 tf.nn.l2_normalize(tensor_j, dim=2)), axis=2)
    outputs = - outputs / (num_heads * num_heads)
    reg = tf.reduce_mean(outputs)
    return reg


def attention_penalty(inputs, num_heads = hp.num_heads):
    inputs = tf.cast(inputs, tf.float32)  # (batch_size, num_heads, maxlen)
    inputs = tf.matmul(inputs, tf.transpose(inputs, [0, 2, 1]))  # (batch_size, num_heads, num_heads)
    outputs = tf.square(tf.subtract(inputs, tf.eye(num_heads)))
    outputs = tf.reduce_sum(tf.reduce_sum(outputs, axis=-1), axis=-1)
    outputs = tf.sqrt(outputs)
    penalty = tf.reduce_mean(outputs)
    return penalty


def get_embedding(inputs, scope):
    with tf.variable_scope(scope, reuse=True):
        lookup_table = tf.get_variable('lookup_table')
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
    return outputs


def get_current_embedding(x_embed):
    x_embed = tf.cast(x_embed, tf.float32)  # [N, T, C]
    x_embed_trans = tf.transpose(x_embed, [1, 0, 2])  # [T, N, C]
    x_embed_cur = x_embed_trans[-1:, :, :]  # [1, N, C]
    return tf.transpose(x_embed_cur, [1, 0, 2])  # [N, 1, C]


def gated_convolution(inputs, num_unit, scope='gated_convolution', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.cast(inputs, tf.float32)  # [N, T, C]
        inputs_norm = normalize(inputs)
        params_A = {"inputs": inputs_norm, "filters": num_unit, "kernel_size": 3, "padding": "same",
                    "activation": None, "use_bias": True}
        params_B = {"inputs": inputs_norm, "filters": num_unit, "kernel_size": 3, "padding": "same",
                    "activation": tf.nn.sigmoid, "use_bias": True}
        A = tf.layers.conv1d(**params_A)
        B = tf.layers.conv1d(**params_B)
        outputs = tf.multiply(A, B)
        outputs += inputs
    return outputs


def conv1d_align(inputs, align_unit, scope='conv1d_align', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.cast(inputs, tf.float32)
        N = tf.shape(inputs)[0]
        T = inputs.get_shape().as_list()[1]
        C = inputs.get_shape().as_list()[-1]
        assert align_unit <= align_unit

        kernel_size = T - align_unit + 1
        params = {"inputs": inputs, "filters": C, "kernel_size": kernel_size, "padding": "valid",
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
    return outputs


def convolution_align(inputs, align_len=hp.lastlen, scope='convolution_align', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.cast(inputs, tf.float32)  # [N, T, C]
        stride_size = align_len * 2
        kernel_size = hp.maxlen - (align_len - 1) * stride_size
        assert kernel_size >= 1
        params = {"inputs": inputs, "filters": hp.hidden_units, "kernel_size": kernel_size, "padding": "valid",
                  "strides": stride_size, "activation": tf.nn.relu, "use_bias": False}
        aligned_outputs = tf.layers.conv1d(**params)
        outputs = aligned_outputs
    return outputs


def gated_convolution_align(inputs, align_len=hp.lastlen, scope='gated_convolution_align', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.cast(inputs, tf.float32)  # [N, T, C]
        inputs = normalize(inputs)
        stride_size = align_len * 2
        kernel_size = hp.maxlen - (align_len - 1) * stride_size
        assert kernel_size >= 1
        params_a = {"inputs": inputs, "filters": hp.hidden_units, "kernel_size": kernel_size, "padding": "valid",
                    "strides": stride_size, "activation": tf.nn.sigmoid, "use_bias": True}
        params_b = {"inputs": inputs, "filters": hp.hidden_units, "kernel_size": kernel_size, "padding": "valid",
                    "strides": stride_size, "activation": None, "use_bias": True}
        aligned_outputs_a = tf.layers.conv1d(**params_a)
        aligned_outputs_b = tf.layers.conv1d(**params_b)
        aligned_outputs = tf.multiply(aligned_outputs_a, aligned_outputs_b)
        outputs = aligned_outputs
    return outputs


def get_last_emb(inputs, time, lastlen=hp.lastlen, scope='get_last_emb', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.cast(inputs, tf.float32)  # [N, T, C]
        T = hp.maxlen
        _, last_inputs = tf.split(inputs, [T-lastlen, lastlen], axis=1)
        _, last_time_inputs = tf.split(time, [T-lastlen, lastlen], axis=1)
    return last_inputs, last_time_inputs


def construct_pos_mask(maxlen=hp.maxlen, stride=32, c=2):
    width = int(stride + (maxlen - stride) / stride * c)
    pos_mask_bool = np.zeros([maxlen, maxlen], dtype=bool)  # [T, T]
    pos_mask = -np.ones([maxlen, width], dtype=int)  # [T, width]
    for i in range(maxlen):
        for j in range(maxlen):
            if j <= i:
                if j // stride == i // stride:
                    pos_mask_bool[i, j] = True
                else:
                    if j % stride >= stride - c:
                        pos_mask_bool[i, j] = True
            else:
                break
        cur_pos_bool = pos_mask_bool[i, :]
        for jj, each in enumerate(np.where(cur_pos_bool)[0]):
            pos_mask[i, jj] = each
    return pos_mask   # [T, width]


def get_attend_inputs(inputs, pos_mask):
    inputs = tf.cast(inputs, tf.float32)  # [N, T, C]
    N = tf.shape(inputs)[0]
    T = inputs.get_shape().as_list()[1]
    pos_mask_mat = tf.one_hot(pos_mask, depth=T, dtype=tf.float32)  # [T, width, T]
    pos_mask_mat = tf.tile(tf.expand_dims(pos_mask_mat, 0), [N, 1, 1, 1])  # [N, T, width, T]
    inputs = tf.tile(tf.expand_dims(inputs, 1), [1, T, 1, 1])  # [N, T, T, C]
    outputs = tf.matmul(pos_mask_mat, inputs)  # [N, T, width, C]
    return outputs


def sparse_attention(inputs, num_unit=hp.hidden_units, num_heads = hp.num_heads,
                     scope='sparse_attention', reuse=None, pos_mask=None):
    """
    Fixed factorized attention of sparse transformer
    :param inputs:
    :param num_unit:
    :param num_heads:
    :param scope:
    :param reuse:
    :param pos_mask:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.cast(inputs, tf.float32)  # [N, T, C]
        pos_mask = tf.convert_to_tensor(pos_mask, tf.int32)  # [T, width]
        attend_inputs = get_attend_inputs(inputs, pos_mask)  # [N, T, width, C] as key and value
        inputs_exp = tf.expand_dims(inputs, 2)  # [N, T, 1, C]  as query

        query = tf.layers.dense(inputs_exp, units=num_unit, activation=tf.nn.relu)
        key = tf.layers.dense(attend_inputs, units=num_unit, activation=tf.nn.relu)
        value = tf.layers.dense(attend_inputs, units=num_unit, activation=tf.nn.relu)

        # split with multi heads
        q_ = tf.concat(tf.split(query, num_heads, axis=-1), axis=0)  # [h*N, T, 1, C/h]
        k_ = tf.concat(tf.split(key, num_heads, axis=-1), axis=0)  # [h*N, T, width, C/h]
        v_ = tf.concat(tf.split(value, num_heads, axis=-1), axis=0)  # [h*N, T, width, C/h]

        # Multiplication
        outputs = tf.matmul(q_, tf.transpose(k_, [0, 1, 3, 2]))  # (h*N, T, 1, width)

        # Scale
        outputs = outputs / (k_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(key, axis=-1)))  # (N, T, width)
        key_masks = tf.tile(key_masks, [num_heads, 1, 1])  # (h*N, T, width)
        key_masks = tf.expand_dims(key_masks, 2)  # (h*N, T, 1, width)
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)

        # Multiplication
        outputs = tf.matmul(outputs, v_)  # [h*N, T, 1, C/h]

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)  # [N, T, 1, C]
        outputs = tf.reshape(outputs, tf.shape(inputs))

        # post attention weight
        outputs = tf.layers.dense(outputs, num_unit, use_bias=False)

        return outputs
    
    
def dense_interpolation(inputs, dense_factor):
    '''
    dense interpolation module: condense the raw concat of flattened output of the encoder
    See [AAAI 2018] Attend and Diagnose Clinical Time Series Analysis Using Attention Models
    :param inputs: output of the encoder  [batch_size, max_len, hidden_unit]
    :param dense_factor: determines dimension of the final interpolation output after the encoder layer. [1]
    Usually, dense factor is greatly less than the length of input sequence.
    :return: output, the dense interpolation of the final layer of encoder
    '''
    inputs = tf.cast(inputs, tf.float32)  # [batch_size, max_len, hidden_unit]
    N = inputs.get_shape().as_list()[0]  # batch_size
    T = hp.maxlen  # max_len
    M = dense_factor  # dense_factor
    s = 1.0*M * tf.range(1, T+1, 1.0, dtype=tf.float32) / T
    m = tf.range(1, M+1, 1.0, dtype=tf.float32)
    W = tf.transpose(tf.tile(tf.expand_dims(s, 1), [1, M])) - tf.tile(tf.expand_dims(m, 1), [1, T])  # [dense_factor, max_len]
    W = tf.pow(1 - tf.abs(W)/M, 2)
    W = tf.tile(tf.expand_dims(W, 0), [N, 1, 1])  # [batch_size, dense_factor, max_len]
    output = tf.matmul(W, inputs)  # [batch_size, dense_factor, hidden_unit]
    return output


def get_dense_interpolation():
    N = hp.batch_size
    T = hp.maxlen
    M = hp.dense_factor
    s = 1.0 * M * tf.range(1, T + 1, 1.0, dtype=tf.float32) / T
    m = tf.range(1, M + 1, 1.0, dtype=tf.float32)
    W = tf.transpose(tf.tile(tf.expand_dims(s, 1), [1, M])) - tf.tile(tf.expand_dims(m, 1),
                                                                      [1, T])  # [dense_factor, max_len]
    W = tf.pow(1 - tf.abs(W) / M, 2)
    W = tf.tile(tf.expand_dims(W, 0), [N, 1, 1])  # [batch_size, dense_factor, max_len]
    return W


