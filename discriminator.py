from model import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder, hierEncoder
import os
import pickle
import torch
import random
import torch.nn as nn
from torch import optim
import time
from search_utils import tensorFromPair
from model import hierEncoder



class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)


def pretrainD(modelD, TrainSet, GenSet, EOS_token, learning_rate=0.01, batch_size=128, to_device=True):
    # prepare data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pos_data = [tensorFromPair(random.choice(TrainSet), EOS_Token=EOS_token, to_device=to_device) for _ in range(batch_size)]
    neg_data = [tensorFromPair(random.choice(GenSet), EOS_Token=EOS_token, to_device=to_device) for _ in range(batch_size)]


    # define optimizer & criterion
    discOptimizer = optim.SGD(modelD.parameters(), lr=learning_rate, momentum=0.8)
    criterion = nn.NLLLoss()
    discOptimizer.zero_grad()

    # some predefined variable
    # 注意：规定 Discriminator 的输出概率的含义为 [positive_probability, negative_probability]
    if to_device:
        posTag = torch.tensor([0]).to(device)
        negTag = torch.tensor([1]).to(device)
    else:
        posTag = torch.tensor([0])
        negTag = torch.tensor([1])

    loss = 0
    start_time = time.time()

    for iter in range(batch_size):
        # choose positive or negative pair randomly
        pick_positive_data = True if random.random() < 0.5 else False
        if pick_positive_data:
            output = modelD(pos_data[iter],to_device=to_device)
            loss += criterion(output, posTag)
        else:
            output = modelD(neg_data[iter],to_device=to_device)
            loss += criterion(output, negTag)

    # BPTT & params updating
    loss.backward()
    discOptimizer.step()

    print("Time consumed: {} Batch loss: {:.2f} ".format((time.time()-start_time),
                                                          loss.item()))

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token
    save_dir = './data/save'
    vocab_dir = os.path.join(save_dir, 'Vocabulary')
    pair_dir = os.path.join(save_dir, 'Preprocessed_data')
    gen_data_dir = os.path.join(save_dir, 'Generated_data')
    GenSet = pickle.load(open(gen_data_dir, 'rb'))
    vocab = pickle.load(open(vocab_dir, 'rb'))
    TrainSet = pickle.load(open(pair_dir, 'rb'))
    for pair in TrainSet:
        for j in range(2):
            pair[j] = pair[j].split()
            pair[j] = [vocab.word2index[word] for word in pair[j]]
    for pair in GenSet:
        for j in range(2):
            pair[j] = [vocab.word2index[word] for word in pair[j] if word != 'EOS' and word != 'PAD']
    embedding_size = 500

    Discriminator = hierEncoder(len(vocab.index2word), 500)
    Discriminator.to(device)
    n_iterations = 4000

    for i in range(n_iterations):
        pretrainD(Discriminator, TrainSet, GenSet, EOS_token)
        print('iteration', i)


