import sys
sys.path.append('..')

import numpy as np
from common.time_layers import *

class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype(np.float32)
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype(np.float32)
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype(np.float32)
        rnn_b = np.zeros(H).astype(np.float32)
        
