from modules import *


def table_mapping(keys, dic, pop):
    if keys in dic:
        return dic[keys]
    else:
        return pop


class MarkovChain:
    def __init__(self, order, class_num, lastlen):
        self.order = order
        self.class_num = class_num
        self.lastlen = lastlen
        self.markov_table = {}
        self.popular_array = np.zeros(self.class_num)

    def forward(self, x_input):
        prior = x_input[:, -self.order:]  # [N, order]
        keys = np.zeros(prior.shape[0], dtype=int)
        for i in range(self.order):
            keys *= 10000
            keys += prior[:, i]
        logits = np.vstack(np.vectorize(table_mapping, otypes=[object],
                                        excluded=['dic', 'pop', 'class_num'])(keys=keys, dic=self.markov_table,
                                                                              pop=self.popular_array))
        return logits

    def get_loss(self, x_input, y_output):
        logits = self.forward(x_input=x_input)
        cost = cross_entropy_np(logits=logits, labels=y_output, class_num=self.class_num)
        return logits, cost

    def update_table(self, x_input, y_output):
        y_output = np.expand_dims(y_output, 1)
        seq = np.hstack([x_input, y_output])
        for each_row in seq:
            key = 0
            index = 0
            for i in range(hp.maxlen+1-self.lastlen-self.order, hp.maxlen+1):
                key = int(key)
                self.popular_array[each_row[i]] += 1
                if index >= self.order:
                    if key not in self.markov_table:
                        self.markov_table[key] = np.zeros(self.class_num)
                    self.markov_table[key][each_row[i]] += 1
                key = np.mod(key, 10000)
                key = key * 10000 * (self.order - 1) + each_row[i]
                index += 1

    def reset_zero(self):
        self.markov_table = {}
        self.popular_array = np.zeros(self.class_num)



