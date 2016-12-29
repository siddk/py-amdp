"""
neural_classifier.py

Core model definition code for Neural-Network Classifier Approach - Takes in a parallel corpus,
and trains/evaluates the model.

Architecture:
    - Embedding Layer
    - RNN (LSTM) => Fixed State Vector
    - ReLU Layer
    - Softmax Output over Commands (Enumerated 1 - N)
"""
import numpy as np
import tensorflow as tf
import tflearn

PAD, PAD_ID = "<<PAD>>", 0
UNK, UNK_ID = "<<UNK>>", 1


class NNClassifier():
    def __init__(self, parallel_corpus, commands, embedding_size=20, lstm_size=100, hidden_size=50,
                 epochs=20, batch_size=32):
        """
        Instantiates and Trains Model using the given parallel corpus.

        :param parallel_corpus: List of Tuples, where each tuple has two elements:
                    1) List of source sentence tokens
                    2) List of target sentence tokens
        :param commands: List of Lists, where each element is one of the possible commands (labels)
        """
        self.commands, self.labels = commands, {" ".join(x): i for (i, x) in enumerate(commands)}
        self.pc, self.epochs, self.bsz = parallel_corpus, epochs, batch_size
        self.embedding_sz, self.lstm_sz, self.hidden_sz = embedding_size, lstm_size, hidden_size

        # Build Vocabulary
        self.word2id, self.id2word = self.build_vocabulary()

        # Vectorize Parallel Corpus
        self.train_x, self.train_y = self.vectorize()
        self.lengths = [len(n) for (n, _) in self.pc]

        # Setup Placeholders
        self.X = tf.placeholder(tf.int32, shape=[None, self.train_x.shape[-1]], name='NL_Command')
        self.Y = tf.placeholder(tf.float32, shape=[None, len(self.commands)], name='ML_Command')
        self.X_len = tf.placeholder(tf.int32, shape=[None], name='NL_Length')
        self.keep_prob = tf.placeholder(tf.float32, shape=[None], name='Dropout_Prob')

        # Build and Compile Model
        self.model = self.compile_model()

        # Fit Model
        self.model.fit([self.train_x, self.lengths, [0.5] * len(self.lengths)], self.train_y,
                       n_epoch=self.epochs, batch_size=self.bsz, snapshot_epoch=False)

    def build_vocabulary(self):
        """
        Builds the vocabulary from the parallel corpus, adding the UNK ID.

        :return: Tuple of Word2Id, Id2Word Dictionaries.
        """
        vocab = set()
        for n, _ in self.pc:
            for word in n:
                vocab.add(word)

        id2word = [PAD, UNK] + list(vocab)
        word2id = {id2word[i]: i for i in range(len(id2word))}
        return word2id, id2word

    def vectorize(self):
        """
        Step through the Parallel Corpus, and convert each sequence to vectors.
        """
        x, y = [], []
        for nl, ml in self.pc:
            nl_vec, ml_label = [self.word2id[w] for w in self.word2id], self.labels[" ".join(ml)]
            x.append(nl_vec)
            y.append(ml_label)
        return tflearn.data_utils.pad_sequences(x), \
            tflearn.data_utils.to_categorical(y, len(self.commands))

    def compile_model(self):
        """
        Compile the LSTM Classifier, taking the input placeholder, generating the softmax
        distribution over all possible reward functions.
        """
        x, xlen = tflearn.input_data(placeholder=self.X), tflearn.input_data(placeholder=self.X_len)
        keep_prob = tflearn.input_data(placeholder=self.keep_prob)
        embedding = tflearn.embedding(x, len(self.word2id), self.embedding_sz)
        embedding = tflearn.dropout(embedding, keep_prob[0])
        lstm = tflearn.lstm(embedding, self.lstm_sz, sequence_length=xlen)
        hidden = tflearn.fully_connected(lstm, self.hidden_sz, activation='relu')
        hidden = tflearn.dropout(hidden, keep_prob[0])
        output = tflearn.fully_connected(hidden, len(self.commands), activation='softmax')
        net = tflearn.regression(output, placeholder=self.Y)
        return tflearn.DNN(net, tensorboard_dir='logs/')

    def score(self, nl_command):
        """
        Given a natural language command, return predicted output and score.

        :return: List of tokens representing predicted command, and score.
        """
        seq, seq_len = [self.word2id.get(w, UNK_ID) for w in nl_command], len(nl_command)
        seq = tflearn.data_utils.pad_sequences([seq], maxlen=self.train_x.shape[1])
        y = self.model.predict([seq, [seq_len], [1.0]])
        pred_command = np.argmax(y, axis=1)
        return self.commands[pred_command], y[0][pred_command]
