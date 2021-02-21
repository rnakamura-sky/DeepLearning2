import sys
sys.path.append('..')
import numpy as np
from common.base_model import BaseModel
from common.time_layers import TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss, TimeDropout
import pickle

class BetterRnnlm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=650, hidden_size=650, dropout_ratio=0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype(np.float32)
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype(np.float32)
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype(np.float32)
        lstm_b1 = np.zeros(4 * H).astype(np.float32)
        lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(D)).astype(np.float32)
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype(np.float32)
        lstm_b2 = np.zeros(4 * H).astype(np.float32)
        # affine_W = (rn(H, V) / np.sqrt(H)).astype(np.float32)
        affine_b = np.zeros(V).astype(np.float32)

        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b),
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def predict(self, xs, train_flg=False):
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs, ts, train_flg=True):
        score = self.predict(xs, train_flg=train_flg)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        for lstm_layer in self.lstm_layers:
            lstm_layer.reset_state()
    

