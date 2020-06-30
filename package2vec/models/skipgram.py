import yaml
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from collections import Counter
import numpy as np
import time
from pathlib import Path
import sys
from torch.utils.tensorboard import SummaryWriter
import git

class PackagesDataset(Dataset):
    '''
        Object containing dataset, word to indices map and vocabulary. Creates
        dataset from raw data during intialization.
    '''

    def __init__(self, options):
        '''
        Args:
            options: instance of Options object containing data, model and
                training configurations.
        '''

        super(PackagesDataset, self).__init__()

        print('DATA | initializing dataset - start')

        # Load data
        self.data_path = options.data_path
        df = pd.read_csv(self.data_path)

        # Use only dependencies which occur more then or equal to min_count
        counts = df['dependency'].value_counts()
        mask = df['dependency'].replace(counts)
        df = df.loc[mask.ge(options.min_count)]

        # Create vocab and word2id
        vocab = set()
        word2id = dict()
        for i, w in enumerate(df['dependency'].unique()):
            vocab.add(w)
            word2id[w] = i

        # Create dataset - input and contex word pairs
        dataset = list()
        packages = df['package'].unique()
        print('INFO | Creating dataset')
        for i, p in enumerate(tqdm(packages)):
            dependencies = df.loc[df['package'] == p]['dependency'].values
            for w in dependencies:
                for c in dependencies:
                    if w != c:
                        dataset.append((word2id[w], word2id[c]))

        # Compute unigram distribution of packages in dataset
        # - used for negative subsampling
        df_dataset = pd.DataFrame(dataset)
        counts = df_dataset[1].value_counts() # Context (output)
        w_freq_neg_samp = np.array(counts.sort_index().values)
        w_freq_neg_samp = w_freq_neg_samp**0.75
        w_freq_neg_samp /= w_freq_neg_samp.sum()

        self.vocab = vocab
        self.w_freq_neg_samp = torch.DoubleTensor(w_freq_neg_samp)
        self.word2id = word2id
        self.dataset = torch.LongTensor(dataset)

        print('DATA | Size of dataset: {}'.format(len(dataset)))
        print('DATA | Size of vocab: {}'.format(len(vocab)))
        print('DATA | initializing dataset - done')

    def get_negative_samples(self, batch_size, n_samples):
        '''
        Returns indices of sampled negative samples based on unigram 
        distribution of packages in dataset.

        Args:
            batch_size: integer 
            n_samples: Number of negative samples to be sampled for each positive 
                sample in batch.
    
        Returns:
            Torch tensor with indeces of negative samples.
        '''

        neg_samples = torch.empty(size=(batch_size, n_samples), dtype = torch.long)
        for i in range(neg_samples.size(0)):
            neg_samples[i] = torch.multinomial(input = self.w_freq_neg_samp, num_samples=n_samples)
        return neg_samples.to(device=options.device)

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class Package2Vec(nn.Module):
    '''
    Skip-gram (word2vec) model using negative subsampling.
    '''
    
    def __init__(self, options):
        '''
        Args:
            options: instance of Options object containing data, model and
                training configurations.
        '''
        super(Package2Vec, self).__init__()

        self.embed_in = nn.Embedding(options.vocab_size, options.embed_dim,
                                     sparse = options.sparse)
        self.embed_out = nn.Embedding(options.vocab_size, options.embed_dim,
                                      sparse = options.sparse)

        # initialize weights of embedding layers to numbers between -1 and 1
        self.embed_in.weight.data.uniform_(-1, 1)
        self.embed_out.weight.data.uniform_(-1, 1)

    def neg_samp_loss(self, in_idxs, pos_out_idxs, neg_out_idxs):
        """
        Loss of model using negative sampling for optimization.
        Source: https://arxiv.org/abs/1310.4546v1
        
        Args:
            in_idx: index of input word
            pos_out_idxs: indices of positive (true) context words
            neg_out_idxs: indices of negative (true) context words

        Returns:
            Loss computed for batch specified by arguments.
        """

        # Vectors of words specified by indices
        in_emb = self.embed_in(in_idxs)
        pos_out_emb = self.embed_out(pos_out_idxs)

        # Loss given by positive samples
        pos_loss = torch.sum(in_emb*pos_out_emb, dim = 1)
        pos_loss = F.logsigmoid(pos_loss)

        # Loss given by negative samples
        neg_out_emb = self.embed_out(neg_out_idxs)
        neg_loss = torch.bmm(-neg_out_emb, in_emb.unsqueeze(2)).squeeze()
        neg_loss = F.logsigmoid(neg_loss)
        neg_loss = torch.sum(neg_loss, dim = 1)

        total_loss = torch.mean(pos_loss + neg_loss)

        return -total_loss

    def forward(self, in_idxs):
        return self.embed_in(in_idxs)

    def save_embbedings(self, path, word2id):
        '''
        Saves learned vectors and corresponding labels to specified path.

        Args:
            path: path to folder to which to save vectors and labels
            word2id: word to index map of dataset (for labels)
        '''
        df = pd.DataFrame(self.embed_in.weight.cpu().detach().numpy())
        df.to_csv(path + 'vectors.tsv', sep='\t', header=False, index=False)

        df = pd.DataFrame(list(word2id.keys()))
        df.to_csv(path + 'labels.tsv', sep='\t', header=False, index=False)


class Options(object):
    '''
    Object containing all data, model and training parameters.
    '''

    def __init__(self, file_path):
        '''
        Parses configurations from yaml file to attributes of this class.
        
        Args:
            file_path: path to yaml file containing congiguration.
        '''
        super(Options, self).__init__()
        y = self.parse_options(file_path)
        data = y['data']
        model = y['model']
        training = y['training']
        output = y['output']

        self.y = y

        self.data_path = data['data_path']
        self.min_count = data['min_count']
        self.vocab_size = None

        self.neg_samples = model['neg_samples']
        self.embed_dim = model['embed_dim']

        self.epochs = training['epochs']
        self.learning_rate = training['learning_rate']
        self.batch_size = training['batch_size']
        self.optimizer = training['optimizer']

        self.output_path = output['path']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('INFO | Using device: {}'.format(self.device))
        self.sparse = True

        r = git.Repo('.', odbt=git.GitCmdObjectDB)
        c = r.head.commit
        self.y['model']['commit_message'] = c.message
        self.y['model']['commit_hash'] = c.hexsha

    def parse_options(self, file_path):
        '''
        Parses yaml file to dictionary.

        Args:
            file_path: path to yaml file containing congiguration.
        '''
        with open(file_path, 'r') as stream:
            try:
                y = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return y


if __name__ == "__main__":

    # Parse options specified by yaml file
    options = Options(sys.argv[1])

    # Create directory for results.
    Path(options.output_path).mkdir(parents=True)

    # Set tensorboard writer for logginf
    writer = SummaryWriter(log_dir=options.output_path)

    # Create dataset from raw data
    dataset = PackagesDataset(options)
    options.vocab_size = len(dataset.vocab)

    # Create data loader for training
    dataloader = DataLoader(dataset, batch_size = options.batch_size, drop_last=True)

    print('TRAINING | Start')

    # Create Skip-gram model specified by parsed configurations
    model = Package2Vec(options).to(device=options.device)


    if options.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=options.learning_rate)
    elif options.optimizer == 'SparseAdam':
        optimizer = optim.SparseAdam(model.parameters(), lr=options.learning_rate)
    elif options.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=options.learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=options.learning_rate)

    # Measure duration of training
    t_training_start = time.time()

    loss_per_epoch = []

    for epoch in range(options.epochs):
        loss_values = []

        # Measure duration of epoch.
        t_start = time.time()

        print('TRAINING | Epoch {}/{}'.format(epoch+1, options.epochs))
        for batch in tqdm(dataloader):
            optimizer.zero_grad()

            # Move data to GPU (if available).
            in_idxs = batch[:, 0].to(options.device)
            out_idxs = batch[:, 1].to(options.device)
            neg_out_idxs = dataset.get_negative_samples(
                options.batch_size, options.neg_samples).to(options.device)

            # Compute loss and backward pass
            loss = model.neg_samp_loss(in_idxs, out_idxs, neg_out_idxs)
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())

        t_ellapsed = time.time() - t_start

        print('TRAINING | Finished epoch {}/{} | Training loss: {} | Ellapsed time: {:.2f} min'.format(
            epoch+1, options.epochs, np.mean(loss_values), t_ellapsed/60))

        # Logging
        writer.add_scalar('Loss/train', np.mean(loss_values), epoch+1)
        writer.add_scalar('Time per epoch', t_ellapsed/60, epoch+1)
        loss_per_epoch.append(np.mean(loss_values))

    t_training_ellapsed = time.time() - t_training_start
    print('TRAINING | Done | Ellapsed time: {:.2f} min'.format(t_training_ellapsed/60))

    # Save losses
    df = pd.DataFrame(loss_per_epoch)
    df.to_csv(options.output_path + 'train_losses.csv',
                   sep = ',', index = False, header = False)

    print('EVALUATION | Train losses saved')

    # Save info
    options.y['t_training_ellapsed[min]'] = t_training_ellapsed/60
    with open(options.output_path + 'config.yaml', 'w') as outfile:
        yaml.dump(options.y, outfile, default_flow_style=False)

    print('EVALUATION | Configs saved')

    # Save learned vectors
    model.save_embbedings(options.output_path, dataset.word2id)

    print('EVALUATION | Embeddings saved')
