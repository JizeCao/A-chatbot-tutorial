from model import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder, hierEncoder
import os
import pickle
import torch
import random
import torch.nn as nn
from torch import optim
import time
from search_utils import tensorFromPair

save_dir = ''
vocab_dir = os.path.join(save_dir, 'Vocab')
pair_dir = os.path.join(save_dir, 'Processed_data')


vocab = pickle.load(open(vocab_dir, 'rb'))
pairs = pickle.load(open(pair_dir, 'rb'))


def pretrainD(modelD, TrainSet, GenSet, learning_rate=0.01, batch_size=128, to_device=True):
    # prepare data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pos_data = [tensorFromPair(random.choice(TrainSet), to_device=to_device) for _ in range(batch_size)]
    neg_data = [tensorFromPair(random.choice(GenSet), to_device=to_device) for _ in range(batch_size)]

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

def generate_sen():



