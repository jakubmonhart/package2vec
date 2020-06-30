import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np

from collections import Counter, defaultdict

from tqdm import tqdm

import time
import sys
import yaml
from pathlib import Path

class Options(object):
    """docstring for options"""
    def __init__(self, file_path):
        super(Options, self).__init__()
        y = self.parse_options(file_path)

        self.data_path = y['data_path']
        self.output_path = y['output_path']

        self.X_max = y['X_max']
        self.alpha = y['alpha']
        self.vocab_len = None
        self.embed_dim = y['embed_dim']
        self.learning_rate = y['learning_rate']
        self.n_epochs = y['n_epochs']
        self.batch_size = y['batch_size']
        self.sparse = y['sparse']
        self.min_count = y['min_count']

        # GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('INFO | Using device: {}'.format(self.device))

        self.y = y

    def parse_options(self, file_path):
        with open(file_path, 'r') as stream:
            try:
                y = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return y

class GloveDataset(Dataset):
    def __init__(self, options):

        # Data is already preprocessed (using filter2_requirements.py)
        df = pd.read_csv(options.data_path)

        # Use only dependencies which occur more then or equal to min_count
        counts = df['dependency'].value_counts()
        mask = df['dependency'].replace(counts)
        df = df.loc[mask.ge(options.min_count)]

        # Word2id
        self.word2id, self.vocab = create_word2id(df)

        # Co-ocurrence matrix
        co_mat = create_co_mat(df, self.word2id)
        
        # Create X, i_idx and j_idx vectors
        X = list()
        self.i_idx = list()
        self.j_idx = list()

        # This should create the dataset I am looping through
        for w_id, cnt in tqdm(co_mat.items()):
            for c_id, v in cnt.items():
                if w_id != c_id:
                    X.append(v)
                    self.i_idx.append(w_id)
                    self.j_idx.append(c_id)


        # Compute log(X) and weight(X) beforehand?
        self.i_idx = torch.LongTensor(self.i_idx)
        self.j_idx = torch.LongTensor(self.j_idx)
        X = torch.FloatTensor(X)
        self.logX = torch.log(X)
        self.weightX =  torch.clamp((X/options.X_max)**options.alpha, max=1)


    def get_batches(self, batch_size, device):
        idxs = torch.randperm(len(self.i_idx))
        for l in range(0, len(idxs), batch_size):
            batch_idxs = idxs[l:l+batch_size]
            yield self.logX[batch_idxs].to(device=device), \
                self.weightX[batch_idxs].to(device=device), \
                self.i_idx[batch_idxs].to(device=device), \
                self.j_idx[batch_idxs].to(device=device)


class GloveModel(nn.Module):
    def __init__(self, options):
        super(GloveModel, self).__init__()
        '''
        Should I set the embeddings as sparse?
        '''

        # Weights - left and right
        self.L_vecs = nn.Embedding(options.vocab_len, options.embed_dim, 
            sparse = options.sparse)
        self.R_vecs = nn.Embedding(options.vocab_len, options.embed_dim, 
            sparse = options.sparse)

        # Biases - left and right
        self.L_bias = nn.Embedding(options.vocab_len, 1, 
            sparse = options.sparse)
        self.R_bias = nn.Embedding(options.vocab_len, 1, 
            sparse = options.sparse)


    def forward(self):
        pass

    def loss(self, logX, weightX, i_idx, j_idx):
        l_vecs = self.L_vecs(i_idx)
        r_vecs = self.R_vecs(j_idx)
        l_bias = self.L_bias(i_idx).squeeze()
        r_bias = self.R_bias(j_idx).squeeze()

        x = (torch.sum(l_vecs*r_vecs, dim=1) + l_bias + r_bias - logX)**2
        loss = weightX * x
        return torch.mean(loss)

    def save_embbedings(self, path, word2id):
        df = pd.DataFrame(self.L_vecs.weight.cpu().detach().numpy())
        df.to_csv(path + 'vectors1.tsv', sep='\t', header=False, index=False)

        df = pd.DataFrame(self.R_vecs.weight.cpu().detach().numpy())
        df.to_csv(path + 'vectors2.tsv', sep='\t', header=False, index=False)

        vectors = self.L_vecs.weight.cpu().detach().numpy() + self.R_vecs.weight.cpu().detach().numpy()

        df = pd.DataFrame(vectors)
        df.to_csv(path + 'vectors.tsv', sep='\t', header=False, index=False)

        df = pd.DataFrame(list(word2id.keys()))
        df.to_csv(path + 'labels.tsv', sep='\t', header=False, index=False)

def create_word2id(df):
    # Vocab and word2id
    vocab = set()
    word2id = dict()
    for i, w in enumerate(df['dependency'].unique()):
        vocab.add(w)
        word2id[w] = i

    print('DATA: vocab and word2id done')

    return word2id, vocab


def create_co_mat(df, word2id):
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

    print('DATA: Co-ocurrence matrix done')
    return co_mat


def print_batch(logX, weightX, i_idx, j_idx, word2id):
    print('logX: {}'.format(logX))
    print('weightX: {}'.format(weightX))
    print('i_idx: {}'.format(i_idx))
    print('j_idx: {}'.format(j_idx))
    print('i words: {}'.format(np.array(list(word2id))[i_idx]))
    print('j words: {}'.format(np.array(list(word2id))[j_idx]))


def train(options_path, options=None):
    if options is None:
        options = Options(options_path)
    

    dataset = GloveDataset(options)
    options.vocab_len = len(dataset.vocab)

    # Create directory for results
    Path(options.output_path).mkdir(parents=True)

    # Tensorboard writer
    writer = SummaryWriter(log_dir=options.output_path)
    
    model = GloveModel(options).to(device=options.device)
    optimizer = optim.Adagrad(model.parameters(), lr=options.learning_rate)
    loss_per_epoch = []

    for epoch in range(options.n_epochs):
        t_start = time.time()
        losses = []
        for batch in dataset.get_batches(options.batch_size, options.device):
            optimizer.zero_grad()

            batch_loss = model.loss(*batch)
            losses.append(batch_loss)
            batch_loss.backward()
            optimizer.step()

        loss = torch.mean(torch.Tensor(losses))
        t_ellapsed = time.time() - t_start
        print('INFO: Epoch {} done | loss: {}'.format(epoch, loss))

        writer.add_scalar('Loss', loss, epoch+1)
        writer.add_scalar('Time per epoch', t_ellapsed/60, epoch+1)

        loss_per_epoch.append(loss.numpy())


    # Save losses
    df = pd.DataFrame(loss_per_epoch)
    df.to_csv(options.output_path + 'train_losses.csv',
                   sep = ',', index = False, header = False)

    # Save model
    model.save_embbedings(options.output_path, dataset.word2id)

    # Save model options
    with open(options.output_path + 'options.yaml', 'w') as outfile:
        yaml.dump(options.y, outfile, default_flow_style=False)



if __name__ == "__main__":
    train(sys.argv[1])


    