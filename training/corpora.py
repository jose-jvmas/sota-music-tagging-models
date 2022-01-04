import os
import numpy as np


class SingleCorpus():

    def read_partition(self, config, partition):
        # Source path:
        read_path = os.path.join('embeddings', config.dataset, config.model_type)

        # Embeddings:
        with open(os.path.join(read_path, partition + '_emb.csv')) as fin:
            FileIn = fin.readlines()
        x = np.array([[float(u) for u in line.split(",")] for line in FileIn])


        # GT:
        with open(os.path.join(read_path, partition + '_gt.csv')) as fin:
            FileIn = fin.readlines()
        y = np.array([[float(u) for u in line.split(",")] for line in FileIn])

        return x, y

    def __init__(self, config):
        self.x_train_NC, self.y_train_onehot = self.read_partition(config, partition = 'train')
        self.train_indexes = list(range(len(self.x_train_NC)))
        
        self.x_valid_NC, self.y_valid_onehot = self.read_partition(config, partition = 'valid')
        self.valid_indexes = list(range(len(self.x_valid_NC)))
        
        self.x_test_NC, self.y_test_onehot = self.read_partition(config, partition = 'test')
        self.test_indexes = list(range(len(self.x_test_NC)))