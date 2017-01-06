"""
nn_classifier.py

Core model definition code for Neural-Network Classifier Approach - Takes in a parallel corpus,
and trains/evaluates the model.

Architecture:
    - Input => Vocabulary Histogram
    - Embedding Layer
    - ReLU Layer
    - ReLU Layer
    - Softmax Output over Commands (Enumerated 1 - N)
"""
import numpy as np
import tensorflow as tf

PAD, PAD_ID = "<<PAD>>", 0
UNK, UNK_ID = "<<UNK>>", 1


class NNClassifier():
    def __init__(self, parallel_corpus, commands, embedding_size=30, h1_size=60, h2_size=50,
                 epochs=10, batch_size=16):
        """
        Instantiates and Trains Model using the given parallel corpus.

        :param parallel_corpus: List of Tuples, where each tuple has two elements:
                    1) List of source sentence tokens
                    2) List of target sentence tokens
        :param commands: List of Lists, where each element is one of the possible commands (labels)
        """
        self.commands, self.labels = commands, {" ".join(x): i for (i, x) in enumerate(commands)}
        self.pc, self.epochs, self.bsz = parallel_corpus, epochs, batch_size
        self.embedding_sz, self.h1_sz, self.h2_sz = embedding_size, h1_size, h2_size
        self.init = tf.truncated_normal_initializer(stddev=0.5)

        # Build Vocabulary
        self.word2id, self.id2word = self.build_vocabulary()

        # Vectorize Parallel Corpus
        self.train_x, self.train_y = self.vectorize()

        # Setup Placeholders
        self.X = tf.placeholder(tf.float32, shape=[None, len(self.word2id)], name='NL_Command')
        self.Y = tf.placeholder(tf.int32, shape=[None], name='ML_Command')
        self.keep_prob = tf.placeholder(tf.float32, name='Dropout_Prob')

        # Build Inference Graph
        self.logits = self.inference()
        self.probs = tf.nn.softmax(self.logits)

        # Build Loss Computation
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits,
                                                                                  self.Y))
        # Build Training Operation
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

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
            nvec, mlab = np.zeros((len(self.word2id)), dtype=np.int32), self.labels[" ".join(ml)]
            for w in nl:
                nvec[self.word2id.get(w, UNK_ID)] += 1
            x.append(nvec)
            y.append(mlab)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

    def inference(self):
        """
        Compile the LSTM Classifier, taking the input placeholder, generating the softmax
        distribution over all possible reward functions.
        """
        # Embedding
        E = tf.get_variable("Embedding", shape=[len(self.word2id), self.embedding_sz],
                            dtype=tf.float32, initializer=self.init)
        embedding = tf.matmul(self.X, E)
        embedding = tf.nn.dropout(embedding, self.keep_prob)

        # ReLU Layer
        H1_W = tf.get_variable("Hidden_W1", shape=[self.embedding_sz, self.h1_sz], dtype=tf.float32,
                               initializer=self.init)
        H1_B = tf.get_variable("Hidden_B1", shape=[self.h1_sz], dtype=tf.float32,
                               initializer=self.init)
        h1 = tf.nn.relu(tf.matmul(embedding, H1_W) + H1_B)

        # ReLU Layer
        H2_W = tf.get_variable("Hidden_W", shape=[self.h1_sz, self.h2_sz], dtype=tf.float32,
                               initializer=self.init)
        H2_B = tf.get_variable("Hidden_B", shape=[self.h2_sz], dtype=tf.float32,
                              initializer=self.init)
        hidden = tf.nn.relu(tf.matmul(h1, H2_W) + H2_B)
        hidden = tf.nn.dropout(hidden, self.keep_prob)

        # Output Layer
        O_W = tf.get_variable("Output_W", shape=[self.h2_sz, len(self.commands)],
                              dtype=tf.float32, initializer=self.init)
        O_B = tf.get_variable("Output_B", shape=[len(self.commands)], dtype=tf.float32,
                              initializer=self.init)
        output = tf.matmul(hidden, O_W) + O_B
        return output

    def fit(self, chunk_size):
        """
        Train the model, with the specified batch size and number of epochs.
        """
        # Initialize all variables
        tf.set_random_seed(42)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        # Run through epochs
        for e in range(self.epochs):
            curr_loss, batches = 0.0, 0.0
            for start, end in zip(range(0, len(self.train_x[:chunk_size]) - self.bsz, self.bsz),
                                  range(self.bsz, len(self.train_x[:chunk_size]), self.bsz)):
                loss, _ = self.session.run([self.loss, self.train_op],
                                           feed_dict={self.X: self.train_x[start:end],
                                                      self.keep_prob: 0.5,
                                                      self.Y: self.train_y[start:end]})
                curr_loss += loss
                batches += 1
            print 'Epoch %s Average Loss:' % str(e), curr_loss / batches

    def score(self, nl_command):
        """
        Given a natural language command, return predicted output and score.

        :return: List of tokens representing predicted command, and score.
        """
        seq = np.zeros((len(self.word2id)), dtype=np.int32)
        for w in nl_command:
            seq[self.word2id.get(w, UNK_ID)] += 1
        y = self.session.run(self.probs, feed_dict={self.X: [seq], self.keep_prob: 1.0})
        [pred_command] = np.argmax(y, axis=1)
        return self.commands[pred_command], y[0][pred_command]
