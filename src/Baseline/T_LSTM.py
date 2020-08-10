# Version 0.1
# 2019-09-02
# Deng Hao

from modules import *
import tensorflow as tf

class TLSTM(object):
    def init_weights(self, input_dim, output_dim, name, std=0.1, reg=None):
        return tf.get_variable(name, shape=[input_dim, output_dim], initializer=tf.random_normal_initializer(0.0, std),
                               regularizer=reg)

    def init_bias(self, output_dim, name):
        return tf.get_variable(name, shape=[output_dim], initializer=tf.constant_initializer(1.0))

    def no_init_weights(self, input_dim, output_dim, name):
        return tf.get_variable(name, shape=[input_dim, output_dim])

    def no_init_bias(self, output_dim, name):
        return tf.get_variable(name, shape=[output_dim])

    def __init__(self, train=1):
        # self.graph = tf.Graph()
        self.vocab_size = hp.vocab_size
        self.num_unit = hp.hidden_units # self.hidden_dim = hidden_dim
        self.maxlen = hp.maxlen
        self.daily_periodic_len = hp.daily_period_len
        self.weekly_periodic_len = hp.weekly_period_len
        self.portrait_vec_len = hp.portrait_vec_len
        self.class_num = hp.output_unit # self.output_dim = output_dim
        # self.input_dim = 1, remember to expend one dim out.
        self.input_dim = 1

        fc_dim = 64

        # [batch size x seq length x input dim]
        self.input = tf.placeholder('float', shape=[None, None, self.input_dim], name='input')
        # todo: - label is numeric number, transverse to oneHotCode in fn "cross_entropy"
        self.labels = tf.placeholder('float', shape=[None], name='labels')
        self.time = tf.placeholder('float', shape=[None, None], name='time')
        self.portrait = tf.placeholder('float', shape=[None, hp.portrait_vec_len], name='portrait')
        self.keep_prob = tf.placeholder_with_default(1.0, shape=[], name='keep_prob')

        if train == 1:

            self.Wi = self.init_weights(self.input_dim, self.num_unit, name='Input_Hidden_weight', reg=None)
            self.Ui = self.init_weights(self.num_unit, self.num_unit, name='Input_State_weight', reg=None)
            self.bi = self.init_bias(self.num_unit, name='Input_Hidden_bias')

            self.Wf = self.init_weights(self.input_dim, self.num_unit, name='Forget_Hidden_weight', reg=None)
            self.Uf = self.init_weights(self.num_unit, self.num_unit, name='Forget_State_weight', reg=None)
            self.bf = self.init_bias(self.num_unit, name='Forget_Hidden_bias')

            self.Wog = self.init_weights(self.input_dim, self.num_unit, name='Output_Hidden_weight', reg=None)
            self.Uog = self.init_weights(self.num_unit, self.num_unit, name='Output_State_weight', reg=None)
            self.bog = self.init_bias(self.num_unit, name='Output_Hidden_bias')

            self.Wc = self.init_weights(self.input_dim, self.num_unit, name='Cell_Hidden_weight', reg=None)
            self.Uc = self.init_weights(self.num_unit, self.num_unit, name='Cell_State_weight', reg=None)
            self.bc = self.init_bias(self.num_unit, name='Cell_Hidden_bias')

            self.W_decomp = self.init_weights(self.num_unit, self.num_unit, name='Decomposition_Hidden_weight',
                                              reg=None)
            self.b_decomp = self.init_bias(self.num_unit, name='Decomposition_Hidden_bias_enc')

            self.Wo = self.init_weights(self.num_unit, fc_dim, name='Fc_Layer_weight',
                                        reg=None)  # tf.contrib.layers.l2_regularizer(scale=0.001)
            self.bo = self.init_bias(fc_dim, name='Fc_Layer_bias')

            self.W_softmax = self.init_weights(fc_dim, self.class_num, name='Output_Layer_weight',
                                               reg=None)  # tf.contrib.layers.l2_regularizer(scale=0.001)
            self.b_softmax = self.init_bias(self.class_num, name='Output_Layer_bias')

        else:

            self.Wi = self.no_init_weights(self.input_dim, self.num_unit, name='Input_Hidden_weight')
            self.Ui = self.no_init_weights(self.num_unit, self.num_unit, name='Input_State_weight')
            self.bi = self.no_init_bias(self.num_unit, name='Input_Hidden_bias')

            self.Wf = self.no_init_weights(self.input_dim, self.num_unit, name='Forget_Hidden_weight')
            self.Uf = self.no_init_weights(self.num_unit, self.num_unit, name='Forget_State_weight')
            self.bf = self.no_init_bias(self.num_unit, name='Forget_Hidden_bias')

            self.Wog = self.no_init_weights(self.input_dim, self.num_unit, name='Output_Hidden_weight')
            self.Uog = self.no_init_weights(self.num_unit, self.num_unit, name='Output_State_weight')
            self.bog = self.no_init_bias(self.num_unit, name='Output_Hidden_bias')

            self.Wc = self.no_init_weights(self.input_dim, self.num_unit, name='Cell_Hidden_weight')
            self.Uc = self.no_init_weights(self.num_unit, self.num_unit, name='Cell_State_weight')
            self.bc = self.no_init_bias(self.num_unit, name='Cell_Hidden_bias')

            self.W_decomp = self.no_init_weights(self.num_unit, self.num_unit, name='Decomposition_Hidden_weight')
            self.b_decomp = self.no_init_bias(self.num_unit, name='Decomposition_Hidden_bias_enc')

            self.Wo = self.no_init_weights(self.num_unit, fc_dim, name='Fc_Layer_weight')
            self.bo = self.no_init_bias(fc_dim, name='Fc_Layer_bias')

            self.W_softmax = self.no_init_weights(fc_dim, self.class_num, name='Output_Layer_weight')
            self.b_softmax = self.no_init_bias(self.class_num, name='Output_Layer_bias')

    def TLSTM_Unit(self, prev_hidden_memory, concat_input):
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

        batch_size = tf.shape(concat_input)[0]
        x = tf.slice(concat_input, [0, 1], [batch_size, self.input_dim])
        t = tf.slice(concat_input, [0, 0], [batch_size, 1])

        # Dealing with time irregularity

        # Map elapse time in days or months
        T = self.map_elapse_time(t)

        # Decompose the previous cell if there is a elapse time
        C_ST = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        C_ST_dis = tf.multiply(T, C_ST)
        # if T is 0, then the weight is one
        prev_cell = prev_cell - C_ST + C_ST_dis

        # Input gate
        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(prev_hidden_state, self.Ui) + self.bi)

        # Forget Gate
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(prev_hidden_state, self.Uf) + self.bf)

        # Output Gate
        o = tf.sigmoid(tf.matmul(x, self.Wog) + tf.matmul(prev_hidden_state, self.Uog) + self.bog)

        # Candidate Memory Cell
        C = tf.nn.tanh(tf.matmul(x, self.Wc) + tf.matmul(prev_hidden_state, self.Uc) + self.bc)

        # Current Memory cell
        Ct = f * prev_cell + i * C

        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.stack([current_hidden_state, Ct])

    def get_states(self):  # Returns all hidden states for the samples in a batch
        batch_size = tf.shape(self.input)[0]
        scan_input_ = tf.transpose(self.input, perm=[2, 0, 1])
        scan_input = tf.transpose(scan_input_)  # scan input is [seq_length x batch_size x input_dim]
        scan_time = tf.transpose(self.time)  # scan_time [seq_length x batch_size]
        initial_hidden = tf.zeros([batch_size, self.num_unit], tf.float32)
        ini_state_cell = tf.stack([initial_hidden, initial_hidden])
        # make scan_time [seq_length x batch_size x 1]
        scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0], tf.shape(scan_time)[1], 1])
        concat_input = tf.concat([scan_time, scan_input], 2)  # [seq_length x batch_size x input_dim+1]
        packed_hidden_states = tf.scan(self.TLSTM_Unit, concat_input, initializer=ini_state_cell, name='states')
        all_states = packed_hidden_states[:, 0, :, :]
        return all_states

    def map_elapse_time(self, t):

        c1 = tf.constant(1, dtype=tf.float32)
        c2 = tf.constant(2.7183, dtype=tf.float32)

        # T = tf.multiply(self.wt, t) + self.bt

        T = tf.div(c1, tf.log(t + c2), name='Log_elapse_time')

        Ones = tf.ones([1, self.num_unit], dtype=tf.float32)

        T = tf.matmul(T, Ones)

        return T

    def meta_fusion(self, inputs, meta_data, is_training=False, scope='meta_fusion', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # flatten
            inputs = tf.reshape(inputs, [-1, hp.maxlen * self.num_unit])
            input_flat = normalize(inputs)

            # concat meta_data
            input_concat = tf.concat([input_flat, meta_data], axis=1)

            # feedforward
            input_mlp = tf.layers.dense(input_concat, units=1024, activation=tf.nn.elu)
            input_mlp = tf.layers.dropout(input_mlp, rate=hp.dropout_rate, training=tf.convert_to_tensor(is_training))
            input_mlp = tf.layers.dense(input_mlp, units=hp.maxlen * hp.hidden_units, activation=None)
            input_mlp = tf.layers.dropout(input_mlp, rate=hp.dropout_rate, training=tf.convert_to_tensor(is_training))

            # Residual connection and Normalize
            input_mlp += inputs
            input_mlp = normalize(input_mlp)
            return input_mlp

    def get_loss(self):

        # with self.graph.as_default():
        logits = self.get_states()

        # todo: - meta_data fusion with residual mlp
        dec_mlp = self.meta_fusion(logits, self.portrait, is_training=self.keep_prob < 1.0)

        # output
        logits = tf.layers.dense(dec_mlp, units=self.class_num, use_bias=False)

        cost = cross_entropy(logits=logits, labels=self.labels, class_num=self.class_num, isSmoothing=True)
        return logits, cost
