# Version 0.1
# 2019-07-17
# Runze Wu

from modules import *
import tensorflow as tf


class LSTM:
    def __init__(self):
        self.graph = tf.Graph()
        self.vocab_size = hp.vocab_size
        self.num_unit = hp.hidden_units
        self.maxlen = hp.maxlen
        self.daily_periodic_len = hp.daily_period_len
        self.weekly_periodic_len = hp.weekly_period_len
        self.portrait_vec_len = hp.portrait_vec_len
        self.class_num = hp.output_unit

    def forward(self, x_input, time_input, portrait_input, daily_periodic_input, weekly_periodic_input, role_id_seq,
                is_training=True, scope='forward'):
        with tf.variable_scope(scope, reuse=None):
            N = tf.shape(x_input)[0]

            # recent embedding
            x_embed = embedding(x_input, vocab_size=self.vocab_size, num_units=self.num_unit, scale=False,
                                scope='input_embed')

            cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_unit)
            init_state = cell.zero_state(N, tf.float32)
            output_rnn, output_state = tf.nn.dynamic_rnn(cell=cell, inputs=x_embed, initial_state=init_state)
            dec = output_rnn

            # meta_data fusion with residual mlp
            dec_mlp = self.meta_fusion(dec, portrait_input, is_training=is_training)

            # output
            logits = tf.layers.dense(dec_mlp, units=self.class_num, use_bias=False)
        return logits

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

    def get_loss(self, x_input, time_input, portrait_input, daily_periodic_input, weekly_periodic_input, role_id_seq,
                 y_output, is_training):
        with self.graph.as_default():
            logits = self.forward(x_input=x_input,
                                  time_input=time_input,
                                  portrait_input=portrait_input,
                                  daily_periodic_input=daily_periodic_input,
                                  weekly_periodic_input=weekly_periodic_input,
                                  role_id_seq=role_id_seq,
                                  is_training=is_training,
                                  )
            cost = cross_entropy(logits=logits, labels=y_output, class_num=self.class_num, isSmoothing=True)
        return logits, cost
