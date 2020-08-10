
from hyperparams import Hyperparams as hp
import pickle
import numpy as np
from datetime import datetime, timedelta
import keras.preprocessing.sequence as sequence
import pytz


def read_pkl(src_file):
    with open(src_file, 'rb') as f:
        return pickle.load(f)


def get_phase(ts, n_phase=12):
    cur_time = datetime.fromtimestamp(float(ts))
    cur_hour = int(cur_time.hour)
    return int(cur_hour // (24/n_phase))


def get_dates(ts, period_type='daily', n_period=6, shift_hour=6):
    assert period_type in ['daily', 'weekly']
    tzname = pytz.timezone('Asia/Shanghai')
    cur_time = datetime.fromtimestamp(float(ts), tz=tzname)
    cur_time = cur_time - timedelta(hours=shift_hour)  # MMORPG player cycle starts at 6AM everyday
    dates = []
    _period = 1
    if period_type == 'weekly':
        _period = 7
    for i in range(n_period):
        date = (cur_time - timedelta(days=(i+1)*_period)).strftime('%F')
        dates = [date] + dates
    return dates


class BatchGenerator:
    def __init__(self):
        # data holder
        self.x_train = read_pkl(hp.X_file_train)  # map_id input
        self.x_test = read_pkl(hp.X_file_test)  # map_id input
        self.y_train = read_pkl(hp.y_file_train)    # map_id output
        self.y_test = read_pkl(hp.y_file_test)  # map_id output
        self.time_train = read_pkl(hp.Time_file_train)  # timestamp input
        self.time_test = read_pkl(hp.Time_file_test)    # timestamp input
        self.time_label_train = read_pkl(hp.Time_file_train_label)  # timestamp output
        self.time_label_test = read_pkl(hp.Time_file_test_label)  # timestamp output
        self.portrait_train = read_pkl(hp.portrait_train)   # portrait input
        self.portrait_test = read_pkl(hp.portrait_test)  # portrait input
        self.role_id_train = read_pkl(hp.role_id_train)  # role_id input
        self.role_id_test = read_pkl(hp.role_id_test)   # role_id input
        self.periodic_dict = read_pkl(hp.periodic_dict)  # periodic dict table

        # parameter
        self.batch_size = hp.batch_size
        self.daily_period_len = hp.daily_period_len
        self.weekly_period_len = hp.weekly_period_len
        self.shift_hour = hp.shift_hour

    def __get_period_list(self, role_id_array, ts_array, period_type, period_len):
        periods_list = []
        if period_len == 0:
            return None
        for each_role_id, each_ts in zip(role_id_array, ts_array):
            dates = get_dates(each_ts, period_type=period_type, n_period=period_len, shift_hour=self.shift_hour)
            session = []
            for each_date in dates:
                cur_ids = self.periodic_dict[int(each_role_id)][each_date]
                session.append(cur_ids)
            # period padding
            session = sequence.pad_sequences(session, maxlen=hp.maxlen, dtype='int32',
                                             padding='pre', truncating='pre', value=0)
            periods_list.append(session)
        return np.array(periods_list)

    def get_batch(self, datatype):
        if datatype == 'training':  # generate training batch
            x_ndarray = np.array(self.x_train)
            y_ndarray = np.asarray(self.y_train, dtype=int)
            time_ndarray = np.array(self.time_train)
            time_label_ndarray = np.array(self.time_label_train)
            portrait_ndarray = np.array(self.portrait_train)
            role_id = np.asarray(self.role_id_train, dtype=int)
        else:   # generate test batch
            x_ndarray = np.array(self.x_test)
            y_ndarray = np.asarray(self.y_test, dtype=int)
            time_ndarray = np.array(self.time_test)
            time_label_ndarray = np.array(self.time_label_test)
            portrait_ndarray = np.array(self.portrait_test)
            role_id = np.asarray(self.role_id_test, dtype=int)

        pos = 0
        n_records = len(y_ndarray)
        shuffle_index0 = np.random.permutation(n_records)  # global shuffling

        while True:  # generator shuffled batch data
            st = pos
            ed = pos + self.batch_size
            x_batch = x_ndarray[shuffle_index0][st:ed]
            y_batch = y_ndarray[shuffle_index0][st:ed]
            time_batch = time_ndarray[shuffle_index0][st:ed]
            time_label_batch = time_label_ndarray[shuffle_index0][st:ed]
            portrait_batch = portrait_ndarray[shuffle_index0][st:ed]
            role_id_batch = role_id[shuffle_index0][st:ed]
            daily_period_batch = self.__get_period_list(role_id_batch, time_label_batch, 'daily', self.daily_period_len)
            weekly_period_batch = self.__get_period_list(role_id_batch, time_label_batch, 'weekly', self.weekly_period_len)

            # padding
            x_batch = sequence.pad_sequences(x_batch, maxlen=hp.maxlen, dtype='int32',
                                             padding='pre', truncating='pre', value=0)
            time_batch = sequence.pad_sequences(time_batch, maxlen=hp.maxlen, dtype='int32',
                                                padding='pre', truncating='pre', value=0)
            pos += self.batch_size
            if pos >= n_records:
                pos = 0
                shuffle_index0 = np.random.permutation(n_records)  # global reshuffling
            shuffle_index1 = np.random.permutation(len(y_batch))  # local shuffling
            if weekly_period_batch is None:
                weekly_period_batch = daily_period_batch
            yield x_batch[shuffle_index1], y_batch[shuffle_index1], time_batch[shuffle_index1], \
                time_label_batch[shuffle_index1], portrait_batch[shuffle_index1], role_id_batch[shuffle_index1], \
                daily_period_batch[shuffle_index1], weekly_period_batch[shuffle_index1]
