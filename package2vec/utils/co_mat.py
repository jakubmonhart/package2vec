import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict


# def get_co_mat(file_path):
#     if file_path == None:
#         print('specify file path to data!')
#         exit(0)

#     df = pd.read_csv(file_path)

#     # Vocab and word2id
#     vocab = set()
#     word2id = dict()
#     for i, w in enumerate(df['dependency'].unique()):
#         vocab.add(w)
#         word2id[w] = i

#     # Loop through packages and their dependencies - add to corresponding location in matrix
#     packages = df['package'].unique()
#     co_mat = np.zeros(shape=(len(vocab), len(vocab)), dtype=int)
#     print(co_mat.shape)

#     for p in tqdm(packages):
#         dependencies = df.loc[df['package'] == p]['dependency'].values
#         # Word
#         for w in dependencies:
#             # Context
#             for c in dependencies:
#                 if c != w:
#                     co_mat[word2id[w]][word2id[c]] += 1

#     return co_mat, word2id


def create_co_mat(file_path):
    if file_path == None:
        print('specify file path to data!')
        exit(0)

    df = pd.read_csv(file_path)

    # Vocab and word2id
    vocab = set()
    word2id = dict()
    for i, w in enumerate(df['dependency'].unique()):
        vocab.add(w)
        word2id[w] = i

    # Loop through packages and their dependencies - add to corresponding location in matrix
    packages = df['package'].unique()
    co_mat = defaultdict(Counter)

    for p in tqdm(packages):
        dependencies = df.loc[df['package'] == p]['dependency'].values
        for w in dependencies:
            for c in dependencies:
                if c != w:
                    # Should I normalize the value somehow?
                    co_mat[word2id[w]][word2id[c]] += 1

    return co_mat, word2id

def print_common_packages(co_mat, word2id, package, n):
    mc = co_mat[word2id[package]].most_common(n)
    
    print('{} packages co-occuring most with {}:\n---'.format(n, package))

    for i, cnt in mc:
        print('{}'.format(list(word2id.keys())[i]))
