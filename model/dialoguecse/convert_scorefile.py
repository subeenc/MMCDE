import os
import numpy as np
import torch
import argparse
import logging
import pickle

from metrics import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--feature_file', default=None, type=str)
    parser.add_argument('--feature_checkpoint', default=None, type=str)
    args = parser.parse_args()

    # logging configuration
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.feature_file) as f:
        lines = f.readlines()

    features, labels = [], []
    topic_mapper = {}
    for line in lines:
        label, feature = line.strip('\n').split('\t')
        if label not in topic_mapper:
            topic_mapper[label] = len(topic_mapper)
        features.append(np.array([float(c) for c in feature.split(',')]))
        labels.append(topic_mapper[label])
    cpu_features = np.stack(features, axis=0)

    pickle.dump(obj={'final_features': cpu_features},
                file=open(args.feature_checkpoint, 'wb'))

    # cpu_features = torch.tensor(cpu_features)
    # features = cpu_features.to(device)
    #
    # feature_based_evaluation_at_once(features=features,
    #                                  labels=labels,
    #                                  gpu_features=features,
    #                                  n_average=1)

