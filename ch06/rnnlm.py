import sys
sys.path.append('..')
import numpy as np
from common.base_model import BaseModel
from common.time_layers import TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss
import pickle

class Rnnlm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype(np.float32)
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype(np.float32)
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype(np.float32)
        lstm_b = np.zeros(4 * H).astype(np.float32)
        affine_W = (rn(H, V) / np.sqrt(H)).astype(np.float32)
        affine_b = np.zeros(V).astype(np.float32)

        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b),
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_later = self.layers[1]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        self.lstm_later.reset_state()
    
