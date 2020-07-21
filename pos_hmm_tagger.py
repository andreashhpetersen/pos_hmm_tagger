"""
POS tagger with HMM and Viterbi

Google Colab Link: https://colab.research.google.com/drive/1aBMTtCqy3MfanQ2QJpXqeqpLjjznOPH5?usp=sharing
"""

import numpy as np
import matplotlib.pyplot as plt

SOS_TOK, EOS_TOK, UNK_TOK = '<SOS>', '<EOS>', '<UNK>'

def get_data(filename):
    """
    Import data from filename and return a list of ([words], [tags]) tuples
    and the set of all words and all tags
    """
    data, words, tags = [], [SOS_TOK], [SOS_TOK]
    all_tags, all_words = { SOS_TOK, EOS_TOK }, { UNK_TOK, SOS_TOK, EOS_TOK }
    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip('\n')
        if line.startswith('#'):
            continue

        if line == '':
            words.append(EOS_TOK)
            tags.append(EOS_TOK)
            data.append((words, tags))
            words, tags = [SOS_TOK], [SOS_TOK]

        else:
            line = line.split('\t')
            words.append(line[1])
            all_words.add(line[1])
            tags.append(line[3])
            all_tags.add(line[3])

    return data, all_words, all_tags


def make_idx_mappings(elements):
    """
    Create mappings from elements in a list to indicies (and the other way
    around)
    """
    el2idx, idx2el = {}, {}
    for idx, el in enumerate(elements):
        el2idx[el] = idx
        idx2el[idx] = el
    return el2idx, idx2el


def plot_heatmap(accuracy_matrix, labels=None):
    """
    Plot an accuracy matrix as a heat map. If 'labels' is provided, use this
    for both x- and y-labels.
    """
    fig, ax = plt.subplots(1,1)
    ax.imshow(accuracy_matrix)

    if labels is not None:
        ticks = [i for i in range(len(labels))]
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation='vertical')
    plt.show()


def to_idx(X, idx_map, UNK_IDX=-1):
    """
    Convert a list X (eg. tags or words) to a list of indicies
    """
    return [idx_map.get(x, UNK_IDX) for x in X]


def normalize(matrix):
    """
    Normalize a matrix so all rows sum to one
    """
    return matrix / matrix.sum(axis=1).reshape(-1,1)


def generate_2d_matrix(dim1, dim2, EPS=0.00001):
    return np.zeros((dim1, dim2)) + EPS


def make_transition_matrix(data, tag2idx):
    """
    Create and train transition matrix
    """
    transitions = generate_2d_matrix(len(tag2idx), len(tag2idx))
    for _, tags in data:
        tags = to_idx(tags, tag2idx)
        for i in range(1, len(tags)):
            transitions[tags[i-1]][tags[i]] += 1
    return normalize(transitions)


def make_emission_matrix(data, word2idx, tag2idx):
    """
    Create and train emission matrix
    """
    emissions = generate_2d_matrix(len(word2idx), len(tag2idx))
    for words, tags in data:
        for word, tag in zip(words, tags):
            wix, tix = word2idx[word], tag2idx[tag]
            emissions[wix][tix] += 1
    return normalize(emissions)


def viterbi(observations, A, B, tag2idx):
    """
    Run the Viterbi algorithm, and return the best sequence. 'A' is the
    transition matrix, 'B' is the emission matrix
    """
    N, T = len(tag2idx), len(observations)
    SOS_IDX, EOS_IDX = tag2idx[SOS_TOK], tag2idx[EOS_TOK]

    viter = np.zeros((N, T+1))
    bptrs = np.zeros((N, T+1), dtype=int) - 1

    # Initialization step
    viter[:,0] = A[SOS_IDX] * B[observations[0]]
    bptrs[:,0] = SOS_IDX

    # Calculate all the probabilities in the network
    for t in range(1, T):
        wix = observations[t]
        viter[:,t] = np.max(viter[:,t-1].reshape(-1,1) * A * B[wix], axis=0)
        bptrs[:,t] = np.argmax(viter[:,t-1].reshape(-1,1) * A * B[wix], axis=0)

    # Termination step
    viter[EOS_IDX,T] = np.max(viter[:,T-1] * A[:,EOS_IDX])
    bptrs[EOS_IDX,T] = np.argmax(viter[:, T-1] * A[:,EOS_IDX])

    # Backtrack through the stored pointers
    states, s = [], EOS_IDX
    for c in range(T, 0, -1):
        s = bptrs[s][c]
        states.append(s)
    states.reverse()

    return states


def train(data, word2idx, tag2idx, use=100):
    """
    Train the model on 'data' and return the transition and the emission
    matricies. If 'use' is provided, only the corresponding percentage of the
    data will be used
    """
    data = data[:int((len(data)/100)*use)]
    transitions = make_transition_matrix(data, tag2idx)
    emissions = make_emission_matrix(data, word2idx, tag2idx)
    return transitions, emissions


def evaluate(data, transitions, emissions, word2idx, tag2idx):
    """
    Evaluate the model using 'data' and return tag and sentence accuracy as
    well as the accuracy matrix
    """
    correct_sequences = 0
    accuracy_matrix = np.zeros((len(tag2idx), len(tag2idx)), dtype=int)
    for words, tags in data:
        words = to_idx(words, word2idx, word2idx[UNK_TOK])
        tags = to_idx(tags, tag2idx)
        pred_tags = viterbi(words, transitions, emissions, tag2idx)
        if tags == pred_tags:
            correct_sequences += 1

        for corr_tag, pred_tag in zip(tags, pred_tags):
            accuracy_matrix[corr_tag][pred_tag] += 1

    correct = accuracy_matrix.diagonal().sum()
    total = accuracy_matrix.sum()
    tag_accuracy = round(correct/total * 100, 2)
    seq_accuracy = round(correct_sequences / len(data) * 100, 2)
    return tag_accuracy, seq_accuracy, accuracy_matrix


LANG = 'en'
BASE_PATH = f'data/{LANG}/'
TRAIN_SET = f'{BASE_PATH}train.conllu'
TEST_SET  = f'{BASE_PATH}test.conllu'

train_data, all_words, all_tags = get_data(TRAIN_SET)
test_data, _, _ = get_data(TEST_SET)
tag2idx, idx2tag = make_idx_mappings(all_tags)
word2idx, idx2word = make_idx_mappings(all_words)

print('Number of training sentences: ', len(train_data))
print('Number of test sentences: ', len(test_data))
print('Number of words: ', len(all_words))
print('Number of tags: ', len(all_tags))

for percentage in range(10, 101, 10):
    transitions, emissions = train(train_data, word2idx, tag2idx, use=percentage)
    tag_acc, seq_acc, accuracy_matrix = evaluate(
        test_data, transitions, emissions, word2idx, tag2idx
    )
    print(f'Language: {LANG}\tTrain data: {percentage}%')
    print(f'Tag accuracy: {tag_acc}\tSentence accuracy: {seq_acc}')
    print()

# Plot accuracy_matrix as heatmap
plot_heatmap(normalize(accuracy_matrix), labels=list(tag2idx.keys()))
