# coding: utf-8
import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi

text = 'You say goodbye and I say hello.'

corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

U, S, V = np.linalg.svd(W)

idx = 0
xi = 0
yi = 2
print(C[idx])
print(W[idx])
print(U[idx])
print(U[idx, :2])

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id,xi], U[word_id,yi]))
plt.scatter(U[:, xi], U[:, yi], alpha=0.5)
plt.show()
