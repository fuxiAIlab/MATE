# Version 0.1
# 2019-07-17
# Runze Wu

from modules import *
import tensorflow as tf


class PeriodicSelfAttention:
    def __init__(self):
        self.graph = tf.Graph()
        self.vocab_size = hp.vocab_size
        self.num_unit = hp.hidden_units
        self.maxlen = hp.maxlen
        self.daily_periodic_len = hp.daily_period_len
        self.weekly_periodic_len = hp.weekly_period_len
        self.portrait_vec_len = hp.portrait_vec_len
        self.class_num = hp.output_unit
        self.dropout_rate = hp.dropout_rate
        self.num_block = hp.num_blocks
        self.num_head = hp.num_heads

    def forward(self, x_input, time_input, portrait_input, daily_periodic_input, weekly_periodic_input, role_id_seq,
                is_training=True, scope='forward'):
        with tf.variable_scope(scope, reuse=None):
            N = tf.shape(x_input)[0]

            # recent embedding
            x_embed = embedding(x_input, vocab_size=self.vocab_size, num_units=self.num_unit, scale=False,
                                scope='input_embed')

            # get current intention embedding
            current_intention_emb = convolution_align(x_embed, scope='intention_emb')

            # get lastlen embedding, time
            last_emb, last_time = get_last_emb(x_embed, time_input)

            # daily periodic intention embedding
            daily_periodic_embed = periodic_embedding_v2(inputs=daily_periodic_input, scope='daily_embed',
                                                         scope_embed='input_embed',
                                                         scope_intention_emb='intention_emb')

            # weekly periodic embedding
            weekly_periodic_embed = periodic_embedding_v2(inputs=weekly_periodic_input, scope='weekly_embed',
                                                          scope_embed='input_embed',
                                                          scope_intention_emb='intention_emb')

            # combine intention module
            intention_emb = tf.concat([current_intention_emb, daily_periodic_embed, weekly_periodic_embed], axis=0)
            # [N*(1+daily_len+weekly_len), lastlen, hidden_unit]

            # intention encoder
            with tf.variable_scope("intention_encoder"):
                pe = positional_encoding(intention_emb, num_units=hp.hidden_units)
                enc = intention_emb + pe
                for i in range(hp.num_encoder_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        enc = self.residual_attention_block(queries=enc, keys=enc, is_training=is_training)
                enc_intention = enc
            enc_intention = tf.reshape(enc_intention, [-1, 1+self.daily_periodic_len+self.weekly_periodic_len,
                                                       hp.lastlen, hp.hidden_units])

            # behavior decoder
            with tf.variable_scope("behavior_decoder"):
                temporal_log_pe = temporal_log_time_positional_encoding(last_emb, hp.hidden_units, last_time)
                enc = last_emb
                # Initialization for decoupled multihead attention module
                u_vec = tf.get_variable('u_vec',
                                        dtype=tf.float32,
                                        shape=[1, self.num_unit],
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        )
                v_vec = tf.get_variable('v_vec',
                                        dtype=tf.float32,
                                        shape=[1, self.num_unit],
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        )
                for i in range(hp.num_decoder_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        enc = self.residual_de_attention_block(queries=enc, keys=enc, time_encoding=temporal_log_pe,
                                                               u_vec=u_vec, v_vec=v_vec, is_training=is_training)
                dec_behavior = enc

            # intention alignment
            current_intention_enc, periodic_intention_enc = tf.split(enc_intention,
                                                                     [1, self.daily_periodic_len+self.weekly_periodic_len],
                                                                     axis=1)
            current_intention_enc = tf.reshape(current_intention_enc, [-1, hp.lastlen, hp.hidden_units])
            periodic_intention_enc = tf.reshape(periodic_intention_enc,
                                                [-1, hp.lastlen*(self.daily_periodic_len+self.weekly_periodic_len),
                                                 hp.hidden_units])
            with tf.variable_scope("intention_alignment"):
                # periodic_intention_enc += positional_encoding(periodic_intention_enc, num_units=hp.hidden_units)
                enc = self.residual_attention_block(queries=current_intention_enc, keys=periodic_intention_enc,
                                                    is_training=is_training)
                enc_intention_align = enc

            # behavior alignment
            with tf.variable_scope("behavior_alignment"):
                # enc_intention_align += positional_encoding(enc_intention_align, num_units=hp.hidden_units)
                enc = self.residual_attention_block(queries=dec_behavior, keys=enc_intention_align,
                                                    is_training=is_training)
                dec = enc

            # meta_data fusion with residual mlp
            dec_mlp = self.meta_fusion(dec, portrait_input, is_training=is_training)

            # output
            logits = tf.layers.dense(dec_mlp, units=self.class_num, use_bias=False)
        return logits

    def residual_de_attention_block(self, queries, keys, time_encoding, u_vec, v_vec, scope='decoupled_residual_block',
                                    reuse=None, is_training=False):
        with tf.variable_scope(scope, reuse=reuse):
            enc_q = queries
            enc_k = keys
            enc_a = relative_multihead_attention(queries=normalize(enc_q),
                                                 keys=normalize(enc_k),
                                                 num_units=hp.hidden_units,
                                                 num_heads=self.num_head,
                                                 pos_enc=time_encoding,
                                                 u_vec=u_vec,
                                                 v_vec=v_vec,
                                                 causality=True,
                                                 reuse=tf.AUTO_REUSE)
            enc_a = tf.layers.dropout(enc_a, rate=hp.dropout_rate,
                                      training=tf.convert_to_tensor(is_training))
            enc_b = normalize(enc_q + enc_a)
            enc_b = feedforward_sparse(enc_b)
            enc_b = tf.layers.dropout(enc_b, rate=hp.dropout_rate,
                                      training=tf.convert_to_tensor(is_training))
            enc_q += enc_a + enc_b
            outputs = enc_q
        return outputs

    def residual_attention_block(self, queries, keys, scope='residual_block', reuse=None, is_training=False):
        with tf.variable_scope(scope, reuse=reuse):
            enc_q = queries
            enc_k = keys

            enc_a = multihead_attention(queries=normalize(enc_q),
                                        keys=normalize(enc_k),
                                        num_units=hp.hidden_units,
                                        num_heads=self.num_head,
                                        causality=True,
                                        reuse=tf.AUTO_REUSE)
            enc_a = tf.layers.dropout(enc_a, rate=hp.dropout_rate,
                                      training=tf.convert_to_tensor(is_training))
            enc_b = normalize(enc_q + enc_a)
            enc_b = feedforward_sparse(enc_b)
            enc_b = tf.layers.dropout(enc_b, rate=hp.dropout_rate,
                                      training=tf.convert_to_tensor(is_training))
            enc_q += enc_a + enc_b
            outputs = enc_q
        return outputs

    def meta_fusion(self, inputs, meta_data, is_training=False, scope='meta_fusion', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # flatten
            inputs = tf.reshape(inputs, [-1, hp.lastlen * self.num_unit])
            input_flat = normalize(inputs)

            # concat meta_data
            input_concat = tf.concat([input_flat, meta_data], axis=1)

            # feedforward
            input_mlp = tf.layers.dense(input_concat, units=1024, activation=tf.nn.elu)
            input_mlp = tf.layers.dropout(input_mlp, rate=hp.dropout_rate, training=tf.convert_to_tensor(is_training))
            input_mlp = tf.layers.dense(input_mlp, units=hp.lastlen * hp.hidden_units, activation=None)
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
