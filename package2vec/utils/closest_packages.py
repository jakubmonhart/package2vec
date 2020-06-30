import torch
import pandas as pd
import numpy as np

def print_closest_packages(file_path=None, word='torch', k=50):
    if file_path == None:
        file_path = '/Users/kuba/Dropbox/work/package2vec/results/pytorch/' \
                    'less_cleaning_lr0.005_neg5_mc2/'

    df = pd.read_csv(file_path + 'vectors.tsv', sep='\t', header=None)
    vectors = torch.Tensor(df.values)

    df = pd.read_csv(file_path + 'labels.tsv', sep='\t', header=None)
    labels = df.values

    # Compute l2 norm (euclidean distance) of input word with all other words
    word_id = np.where(labels == word)[0]

    dist = torch.sum(torch.pow(vectors-vectors[word_id], 2),dim=1)
    topk = torch.topk(-dist, k+1).indices

    print('{} closest packages to {}:\n---'.format(k, word))

    for label in labels[topk][1:].squeeze():
        print(label)

    # return topk, labels[topk].squeeze()
    return 0
