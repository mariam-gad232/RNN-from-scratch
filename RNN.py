import math
import random


sentence = ["I", "love", "deep", "learning"]
vocab = list(set(sentence))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

# Convert words to indices
inputs = [word_to_idx[w] for w in sentence[:3]]
target = word_to_idx[sentence[3]]

vocab_size = len(vocab)
hidden_size = 4 
learning_rate = 0.1


def one_hot(index, size):
    vec = [0] * size
    vec[index] = 1
    return vec

def softmax(x):
    exps = [math.exp(i) for i in x]
    sum_exps = sum(exps)
    return [j / sum_exps for j in exps]

def cross_entropy(pred, target_idx):
    return -math.log(pred[target_idx])


def init_matrix(rows, cols):
    return [[random.uniform(-0.1, 0.1) for _ in range(cols)] for _ in range(rows)]

Wxh = init_matrix(hidden_size, vocab_size)
Whh = init_matrix(hidden_size, hidden_size)
Why = init_matrix(vocab_size, hidden_size)
bh = [0] * hidden_size
by = [0] * vocab_size

for epoch in range(1000):
    hs = [[0] * hidden_size]  
    xs = []

    # Forward pass 
    for t in range(3):
        x = one_hot(inputs[t], vocab_size)
        xs.append(x)
        h_prev = hs[-1]
        
        h = []
        for i in range(hidden_size):
            total = sum(Wxh[i][j] * x[j] for j in range(vocab_size)) + \
                    sum(Whh[i][j] * h_prev[j] for j in range(hidden_size)) + bh[i]
            h.append(math.tanh(total))
        hs.append(h)


    y = []
    for i in range(vocab_size):
        total = sum(Why[i][j] * hs[-1][j] for j in range(hidden_size)) + by[i]
        y.append(total)
    
    probs = softmax(y)
    loss = cross_entropy(probs, target)

   
    
    dWhy = init_matrix(vocab_size, hidden_size)
    dWxh = init_matrix(hidden_size, vocab_size)
    dWhh = init_matrix(hidden_size, hidden_size)
    dbh = [0] * hidden_size
    dby = [0] * vocab_size
    dh_next = [0] * hidden_size

    # Output gradient
    dy = probs[:]
    dy[target] -= 1 

    for i in range(vocab_size):
        for j in range(hidden_size):
            dWhy[i][j] += dy[i] * hs[-1][j]
        dby[i] += dy[i]

    
    for t in reversed(range(3)):
        dh = [0] * hidden_size
        for i in range(hidden_size):
            for j in range(vocab_size):
                dh[i] += Why[j][i] * dy[j]
            dh[i] += dh_next[i]

        dh_raw = [dh[i] * (1 - hs[t+1][i] ** 2) for i in range(hidden_size)]

        for i in range(hidden_size):
            for j in range(vocab_size):
                dWxh[i][j] += dh_raw[i] * xs[t][j]
            for j in range(hidden_size):
                dWhh[i][j] += dh_raw[i] * hs[t][j]
            dbh[i] += dh_raw[i]

        dh_next = [sum(Whh[j][i] * dh_raw[j] for j in range(hidden_size)) for i in range(hidden_size)]

    
    for i in range(vocab_size):
        for j in range(hidden_size):
            Why[i][j] -= learning_rate * dWhy[i][j]
        by[i] -= learning_rate * dby[i]

    for i in range(hidden_size):
        for j in range(vocab_size):
            Wxh[i][j] -= learning_rate * dWxh[i][j]
        for j in range(hidden_size):
            Whh[i][j] -= learning_rate * dWhh[i][j]
        bh[i] -= learning_rate * dbh[i]

    if epoch % 100 == 0:
        pred_idx = probs.index(max(probs))
        pred_word = idx_to_word[pred_idx]
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Predicted: {pred_word}, Target: {sentence[3]}")

