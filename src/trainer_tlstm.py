
import tensorflow as tf
from hyperparams import Hyperparams as hp
from mate import PeriodicSelfAttention
from Baseline.LSTM import LSTM
from batch_geneartor import BatchGenerator
import numpy as np
import os
from Baseline.T_LSTM import TLSTM


def dir_check(path):
    if not os.path.exists(path):
        os.makedirs(path)


class MetricTracker:
    def __init__(self, class_num, file_stream=None):
        self.file_stream = file_stream
        self.class_num = class_num
        self.acc_count = 0
        self.acc_true = 0
        self.loss_sum = 0.0
        self.mse = 0.0
        self.acc_count_prec_per_class = [0 + 1e-10] * class_num
        self.acc_per_class = [0] * class_num
        self.acc_count_recl_per_class = [0 + 1e-10] * class_num
        self.confusion_matrix = np.zeros([class_num, class_num], dtype=int)
        if file_stream is not None and os.path.exists(file_stream):
            os.remove(file_stream)

    def reset_zero(self):
        self.acc_count = 0
        self.acc_true = 0
        self.loss_sum = 0.0
        self.mse = 0
        self.acc_count_prec_per_class = [0 + 1e-10] * self.class_num
        self.acc_per_class = [0] * self.class_num
        self.acc_count_recl_per_class = [0 + 1e-10] * self.class_num
        self.confusion_matrix = np.zeros([self.class_num, self.class_num], dtype=int)

    def update(self, logits, label, loss):
        y_pred = np.argmax(logits, axis=1)
        y_real = label
        n_sample = len(y_real)
        self.loss_sum += loss*n_sample
        self.acc_count += n_sample
        for i in range(n_sample):
            self.acc_count_prec_per_class[y_pred[i]] += 1
            self.acc_count_recl_per_class[y_real[i]] += 1
            self.confusion_matrix[y_real[i], y_pred[i]] += 1
            if y_pred[i] == y_real[i]:
                self.acc_true += 1
                self.acc_per_class[y_pred[i]] += 1

    def print_info(self, name, epoch_i, batch_i, max_epoch, max_batch, is_acc=True, is_rmse=False,
                   is_each_class=False, is_confusion_mat=False):
        print_str = 'Epoch {:>3}/{} Batch {:>4}/{} - '.format(epoch_i, max_epoch, batch_i, max_batch)
        print_str += 'Loss: {:>6.3f} - '.format(self.loss_sum/self.acc_count)
        if is_acc:
            print_str += '{} Acc: {:>6.3f} - '.format(name, self.acc_true / self.acc_count)
        if is_rmse:
            print_str += '{} RMSE: {:>6.3f} - '.format(name, np.sqrt(self.mse / self.acc_count))
        if is_each_class:  # output results for each class
            for i in range(self.class_num):
                print_str += '\n{} Precision of Class {:>3d}: {:d}/{:d}={:>6.3f}'\
                    .format(name, i, self.acc_per_class[i], self.acc_count_prec_per_class,
                            self.acc_per_class[i]/self.acc_count_prec_per_class[i])
            for i in range(self.class_num):
                print_str += '\n{} Recall of Class {:>3d}: {:d}/{:d}={:>6.3f}'\
                    .format(name, i, self.acc_per_class[i], self.acc_count_recl_per_class,
                            self.acc_per_class[i]/self.acc_count_recl_per_class[i])
        if is_confusion_mat:
            print_str += '\n{} Confusion Matrix:'.format(name)
            for i in range(self.class_num):
                print_str += '\n' + '\t'.join(str(each) for each in self.confusion_matrix[i])
        print(print_str)
        if self.file_stream is not None:
            with open(self.file_stream, 'a') as f:
                print(print_str, file=f)


class ModelTrainer:
    def __init__(self, log_dir=hp.log_file, model_file=hp.model_file, output_file=None):
        self.lrate = hp.learning_rate
        self.max_epoch = hp.epochs
        self.batch_size = hp.batch_size
        self.train_size = hp.train_size
        self.test_size = hp.test_size
        self.max_batchsize = self.train_size // self.batch_size
        self.max_batchsize_test = self.test_size // self.batch_size

        self.model_file = model_file if model_file is not None else hp.model_file
        self.checkpoint = self.model_file + "best_model.ckpt"
        self.log_file = log_dir if log_dir is not None else hp.log_file

        self.model = TLSTM(train=1) # PeriodicSelfAttention()  # LSTM()  #
        self.batch_generator = BatchGenerator()
        self.train_metric_tracker = MetricTracker(class_num=self.model.class_num, file_stream=output_file)
        self.test_metric_tracker = MetricTracker(class_num=self.model.class_num, file_stream=output_file)

    def __input_placeholder(self):
        with tf.name_scope('input'):
            # input
            x_input = tf.placeholder(dtype=tf.int32, shape=[None, self.model.maxlen], name='x_input')
            time_input = tf.placeholder(dtype=tf.float32, shape=[None, self.model.maxlen], name='time_input')
            portrait_input = tf.placeholder(dtype=tf.float32, shape=[None, self.model.portrait_vec_len],
                                            name='portrait_input')
            daily_periodic_input = tf.placeholder(dtype=tf.int32,
                                                  shape=[None, self.model.daily_periodic_len, self.model.maxlen],
                                                  name='daily_periodic_input')
            weekly_periodic_input = tf.placeholder(dtype=tf.int32,
                                                   shape=[None, self.model.weekly_periodic_len, self.model.maxlen],
                                                   name='weekly_periodic_input')
            role_id_seq = tf.placeholder(tf.int32, [None], name='role_id_seq')
            is_training = tf.placeholder(tf.bool, name='is_training')

            # label
            y_output = tf.placeholder(tf.int32, [None], name='y_output')
            time_label = tf.placeholder(tf.float32, [None], name='time_label')

        return x_input, time_input, portrait_input, daily_periodic_input, weekly_periodic_input, role_id_seq,\
            is_training, y_output, time_label

    def model_train(self):
        # Check dir
        dir_check(self.log_file)
        dir_check(self.model_file)
        # with self.model.graph.as_default():
        # Get placeholder
        x_input, time_input, portrait_input = self.model.input, self.model.time, self.model.portrait
        keep_prob = self.model.keep_prob
        y_output = self.model.labels

        # Get loss
        # logits, cost = self.model.get_loss(x_input, time_input, portrait_input, daily_periodic_input,
        #                                    weekly_periodic_input, role_id_seq, y_output, is_training)
        logits, cost = self.model.get_loss()
        tf.summary.scalar('loss', cost)

        # Optimizer
        global_steps = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.train.AdamOptimizer(self.lrate, beta1=0.9,
                                          beta2=0.999, epsilon=1e-8).minimize(cost, global_step=global_steps)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            # config = tf.ConfigProto(inter_op_parallelism_threads=hp.inter_op_parallelism_threads,
            #                         intra_op_parallelism_threads=hp.intra_op_parallelism_threads)
            # sess = tf.Session(config=config, graph=self.model.graph)
            sess.run(init)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.log_file + 'train', sess.graph)
            test_writer = tf.summary.FileWriter(self.log_file + 'test')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            epoch_i = 1
            for batch_i, (x_batch, y_batch, time_batch, time_label_batch, portrait_batch, role_id_batch,
                          daily_period_batch, weekly_period_batch) \
                    in enumerate(self.batch_generator.get_batch(datatype='training')):
                if batch_i % self.max_batchsize == 0 and batch_i > 0:
                    epoch_i += 1
                    self.train_metric_tracker.reset_zero()
                    self.test_metric_tracker.reset_zero()
                    if epoch_i > self.max_epoch:
                        break

                # validation
                x_batch_ = np.expand_dims(x_batch, -1)
                time_batch_ = np.append(
                    np.zeros((time_batch.shape[0], 1)),
                    np.diff(time_batch, axis=1),
                    axis=1
                )
                summary, _, logits_, loss_ = sess.run(
                    [merged, train_op, logits, cost],
                    {x_input              : x_batch_,
                     y_output             : y_batch,
                     keep_prob            : 0.8,
                     time_input           : time_batch_,
                     portrait_input       : portrait_batch,
                     })

                self.train_metric_tracker.update(logits_, y_batch, loss_)
                train_writer.add_summary(summary, batch_i)

                # Testing
                if (batch_i % self.max_batchsize) + 1 == self.max_batchsize:
                    for test_batch_i in range(self.max_batchsize_test):
                        x_test_batch, y_test_batch, time_test_batch, time_label_test_batch, \
                        portrait_test_batch, \
                        role_id_test_batch, daily_period_test_batch, weekly_period_test_batch \
                            = next(self.batch_generator.get_batch(datatype='testing'))

                        x_test_batch_ = np.expand_dims(x_test_batch, -1)
                        time_test_batch_ = np.append(
                            np.zeros((time_test_batch.shape[0], 1)),
                            np.diff(time_test_batch, axis=1),
                            axis=1
                        )
                        summary, logits_, loss_ = sess.run(
                            [merged, logits, cost],
                            {x_input              : x_test_batch_,
                             y_output             : y_batch,
                             time_input           : time_test_batch_,
                             portrait_input       : portrait_test_batch,
                             })

                        self.test_metric_tracker.update(logits_, y_test_batch, loss_)
                        test_writer.add_summary(summary, test_batch_i + (epoch_i - 1) * self.max_batchsize_test)

                    # save model per epoch
                    saver = tf.train.Saver()
                    saver.save(sess, self.model_file + "epoch" + str(epoch_i) + "batch" + str(
                        (batch_i % self.max_batchsize) + 1) + ".ckpt")

                    # print performance info per epoch
                    self.train_metric_tracker.print_info('Train', epoch_i, (batch_i % self.max_batchsize) + 1,
                                                         self.max_epoch,
                                                         self.max_batchsize)
                    self.test_metric_tracker.print_info('Test', epoch_i,
                                                        (test_batch_i % self.max_batchsize_test) + 1,
                                                        self.max_epoch,
                                                        self.max_batchsize_test)


            # save results after training
            saver = tf.train.Saver()
            saver.save(sess, self.checkpoint)