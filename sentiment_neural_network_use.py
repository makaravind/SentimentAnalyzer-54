import tensorflow as tf
import numpy as np
import pickle

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


IS_TRAINED_SAVED = False

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

from create_sentiment_featuresets import create_feature_sets_and_labels


train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
# train_x, train_y, test_x, test_y = pickle.load(open('sentiment_set.pickle', 'rb'))

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

n_classes = 2
batch_size = 100

hl1_nodes = 500
hl2_nodes = 500
hl3_nodes = 100


hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), hl1_nodes])),
                      'biases': tf.Variable(tf.random_normal([hl1_nodes]))}

hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([hl1_nodes, hl2_nodes])),
                  'biases': tf.Variable(tf.random_normal([hl2_nodes]))}

hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([hl2_nodes, hl3_nodes])),
                  'biases': tf.Variable(tf.random_normal([hl3_nodes]))}

output_layer = {'weights': tf.Variable(tf.random_normal([hl3_nodes, n_classes])),
                'biases': tf.Variable(tf.random_normal([n_classes]))}


def neural_network_model(data):

    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    return output

saver = tf.train.Saver()


def train_neural_network(x):

    prediction = neural_network_model(x)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdadeltaOptimizer(0.5).minimize(cross_entropy)

    # model has ended here
    #now, run the computation by creating a sesstion

    hm_epochs = 10

    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())
        for each_epoch in range(hm_epochs):

            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i+=batch_size

            print(each_epoch, 'is done out of', hm_epochs, 'loss:', epoch_loss)

        print('training is done!')
        saver.save(sess, 'sentiment_model.ckpt')

        is_correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
        accuracy_calculator = tf.reduce_mean(tf.cast(is_correct, 'float'))

        print('accuracy :', accuracy_calculator.eval({x: test_x, y: test_y}))

        print('length of train_X', len(train_x))
        # return prediction, test_x, test_y


# train_neural_network(x)

def use_neural_network(input_data):

    prediction = neural_network_model(x)

    with open('sentiment_lexicon.pickle', 'rb') as f:
        lexicon = pickle.load(f)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        try:
            saver.restore(sess, "sentiment_model.ckpt")
        except:
            print 'train the model firs, run train_neural_network method'
            return
        
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                # OR DO +=1, test both
                features[index_value] += 1

        features = np.array(list(features))
        # pos: [1,0] , argmax: 0
        # neg: [0,1] , argmax: 1
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [features[:423]]}), 1)))

        print prediction.eval(feed_dict={x: [features[:423]]})

        if result[0] == 0:
            print('Positive:', input_data)
        elif result[0] == 1:
            print('Negative:', input_data)

use_neural_network('i dont like this at all!')
use_neural_network('it is so awesome')
use_neural_network('i hate you')
use_neural_network('you are not a human')
use_neural_network("I'm not bad")




















